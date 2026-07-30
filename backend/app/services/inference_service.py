"""In-process inference for the chat playground and benchmarking.

Keeps at most one model loaded at a time (consumer GPU VRAM budget) with a
simple LRU-of-one cache; loading a different model evicts the current one.
Runs in the main API process (unlike training) since generation is comparatively
cheap and needs low-latency streaming back to the UI.
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

_lock = Lock()


@dataclass
class LoadedModel:
    repo_id: str
    model: Any
    tokenizer: Any
    load_kwargs: dict = field(default_factory=dict)


_current: LoadedModel | None = None


def _unload_current() -> None:
    global _current
    if _current is None:
        return
    import gc

    import torch

    del _current.model
    _current = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(repo_id: str, adapter_path: str | None = None, load_in_4bit: bool = False) -> None:
    global _current
    with _lock:
        if _current is not None and _current.repo_id == repo_id and _current.load_kwargs.get("adapter_path") == adapter_path:
            return

        _unload_current()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        _current = LoadedModel(repo_id=repo_id, model=model, tokenizer=tokenizer, load_kwargs={"adapter_path": adapter_path})


def unload_model() -> None:
    with _lock:
        _unload_current()


def get_current_model_id() -> str | None:
    return _current.repo_id if _current else None


def format_prompt(messages: list[dict], system_prompt: str | None = None) -> str:
    if _current is None:
        raise RuntimeError("No model loaded")
    chat = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    try:
        return _current.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\nassistant:"


def stream_generate(
    messages: list[dict],
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
):
    """Yields (token_text, is_final, stats) tuples. Runs generation in a background thread."""
    if _current is None:
        raise RuntimeError("No model loaded")

    import threading

    import torch
    from transformers import TextIteratorStreamer

    prompt = format_prompt(messages, system_prompt)
    inputs = _current.tokenizer(prompt, return_tensors="pt").to(_current.model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(_current.tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        pad_token_id=_current.tokenizer.pad_token_id,
    )
    if min_p > 0:
        gen_kwargs["min_p"] = min_p

    start = time.perf_counter()
    thread = threading.Thread(target=_current.model.generate, kwargs=gen_kwargs)
    thread.start()

    first_token_time = None
    token_count = 0
    for text in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        token_count += 1
        yield text, False, None

    thread.join()
    end = time.perf_counter()

    stats = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": token_count,
        "time_to_first_token_s": (first_token_time - start) if first_token_time else None,
        "total_time_s": end - start,
        "tokens_per_sec": token_count / (end - start) if end > start else 0,
    }
    yield "", True, stats


def benchmark_generation(prompt: str, max_new_tokens: int = 256) -> dict:
    if _current is None:
        raise RuntimeError("No model loaded")

    import torch

    inputs = _current.tokenizer(prompt, return_tensors="pt").to(_current.model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        output = _current.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=_current.tokenizer.pad_token_id
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    generated_tokens = output.shape[-1] - prompt_tokens
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_time_s": end - start,
        "tokens_per_sec": generated_tokens / (end - start) if end > start else 0,
        "peak_gpu_mem_gb": peak_mem_gb,
    }
