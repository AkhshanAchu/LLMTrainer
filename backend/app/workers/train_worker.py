"""Standalone training worker.

Runs in its own OS process (launched via subprocess by the training service) so
that CUDA memory, the GIL, and a crash/OOM are fully isolated from the FastAPI
server process. Communicates progress back via a JSONL file that the API tails.

Usage: python -m app.workers.train_worker <run_id> <config_json_path> <status_path>
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path


def emit(status_path: Path, event: str, **data) -> None:
    record = {"event": event, "ts": time.time(), **data}
    with open(status_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_bnb_config(cfg: dict):
    from transformers import BitsAndBytesConfig
    import torch

    if not (cfg.get("load_in_4bit") or cfg.get("load_in_8bit")):
        return None
    compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bf16"), torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_dataset_records(dataset_path: str, fmt: str) -> list[dict]:
    import json as _json

    path = Path(dataset_path)
    records = []
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_json.loads(line))
    elif path.suffix == ".json":
        data = _json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("data", [])
    else:
        raise ValueError(f"Unsupported dataset file for training: {path.suffix}")
    return records


def to_chat_messages(record: dict, fmt: str) -> list[dict]:
    if fmt == "sharegpt":
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        return [
            {"role": role_map.get(m.get("from"), m.get("from", "user")), "content": m.get("value", "")}
            for m in record.get("conversations", [])
        ]
    if fmt in ("chatml", "openai"):
        return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in record.get("messages", [])]
    if fmt == "alpaca":
        instruction = record.get("instruction", "")
        inp = record.get("input", "")
        user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record.get("output", "")},
        ]
    if fmt == "preference":
        prompt = record.get("prompt", "")
        return [{"role": "user", "content": prompt}]
    # raw_text / unknown
    return [{"role": "user", "content": record.get("text", "")}]


def run_sft(cfg: dict, status_path: Path) -> None:
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    emit(status_path, "stage", stage="loading_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_repo_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    emit(status_path, "stage", stage="loading_model")
    bnb_config = build_bnb_config(cfg)
    model_kwargs = dict(
        pretrained_model_name_or_path=cfg["base_model_repo_id"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    if cfg.get("use_flash_attention_2"):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    method = cfg.get("method", "qlora")
    if method in ("lora", "qlora", "dora"):
        from peft import prepare_model_for_kbit_training

        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        target_modules = cfg.get("lora_target_modules") or find_linear_module_names(model)
        peft_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=target_modules,
            use_dora=cfg.get("use_dora", False) or method == "dora",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        trainable, total = model.get_nb_trainable_parameters()
        emit(status_path, "info", trainable_params=trainable, total_params=total)

    emit(status_path, "stage", stage="loading_dataset")
    records = load_dataset_records(cfg["dataset_path"], cfg["dataset_format"])
    texts = []
    for r in records:
        messages = to_chat_messages(r, cfg["dataset_format"])
        if not messages or not any(m["content"] for m in messages):
            continue
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        texts.append(text)

    if not texts:
        raise RuntimeError("No trainable examples produced from dataset after formatting.")

    eval_ratio = cfg.get("eval_split_ratio", 0.05)
    split_idx = max(1, int(len(texts) * (1 - eval_ratio))) if len(texts) > 20 else len(texts)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:] if split_idx < len(texts) else None

    train_ds = HFDataset.from_dict({"text": train_texts})
    eval_ds = HFDataset.from_dict({"text": eval_texts}) if eval_texts else None

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None
            emit(
                status_path,
                "metric",
                step=state.global_step,
                epoch=state.epoch or 0.0,
                loss=logs.get("loss"),
                eval_loss=logs.get("eval_loss"),
                learning_rate=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
                gpu_mem_used_gb=gpu_mem,
                total_steps=state.max_steps,
            )

        def on_save(self, args, state, control, **kwargs):
            emit(status_path, "checkpoint", step=state.global_step, epoch=state.epoch or 0.0)

    max_steps = cfg.get("max_steps", -1)
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=cfg.get("num_train_epochs", 3.0),
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        optim=cfg.get("optimizer", "paged_adamw_8bit"),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", 50) if eval_ds is not None else None,
        max_length=cfg.get("max_seq_length", 2048),
        packing=cfg.get("packing", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        bf16=torch.cuda.is_available(),
        seed=cfg.get("seed", 42),
        report_to=[],
        dataset_text_field="text",
        neftune_noise_alpha=cfg.get("neftune_noise_alpha"),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[ProgressCallback()],
    )

    emit(status_path, "stage", stage="training")
    trainer.train()

    emit(status_path, "stage", stage="saving_final")
    final_dir = str(Path(cfg["output_dir"]) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    emit(status_path, "done", final_dir=final_dir)


def find_linear_module_names(model) -> list[str]:
    import torch.nn as nn

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf not in ("lm_head",):
                names.add(leaf)
    return sorted(names) or ["q_proj", "k_proj", "v_proj", "o_proj"]


def run_preference(cfg: dict, status_path: Path) -> None:
    """DPO / ORPO / KTO via TRL. Shares model/tokenizer loading with SFT path."""
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

    objective = cfg["objective"]
    emit(status_path, "stage", stage="loading_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_repo_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    emit(status_path, "stage", stage="loading_model")
    bnb_config = build_bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_repo_id"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = cfg.get("lora_target_modules") or find_linear_module_names(model)
    peft_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    emit(status_path, "stage", stage="loading_dataset")
    records = load_dataset_records(cfg["dataset_path"], cfg["dataset_format"])
    prompts, chosen, rejected = [], [], []
    for r in records:
        p, c, rj = r.get("prompt"), r.get("chosen"), r.get("rejected")
        if p is None or c is None or rj is None:
            continue
        prompts.append(p)
        chosen.append(c)
        rejected.append(rj)

    if not prompts:
        raise RuntimeError("No valid prompt/chosen/rejected triples found for preference training.")

    ds = HFDataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None
            emit(
                status_path,
                "metric",
                step=state.global_step,
                epoch=state.epoch or 0.0,
                loss=logs.get("loss"),
                eval_loss=logs.get("eval_loss"),
                learning_rate=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
                gpu_mem_used_gb=gpu_mem,
                total_steps=state.max_steps,
            )

        def on_save(self, args, state, control, **kwargs):
            emit(status_path, "checkpoint", step=state.global_step, epoch=state.epoch or 0.0)

    max_steps = cfg.get("max_steps", -1)

    if objective == "dpo":
        from trl import DPOConfig, DPOTrainer

        args = DPOConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            num_train_epochs=cfg.get("num_train_epochs", 3.0),
            max_steps=max_steps if max_steps and max_steps > 0 else -1,
            learning_rate=cfg.get("learning_rate", 2e-4),
            beta=cfg.get("beta", 0.1),
            logging_steps=cfg.get("logging_steps", 5),
            save_steps=cfg.get("save_steps", 50),
            save_total_limit=cfg.get("save_total_limit", 3),
            bf16=torch.cuda.is_available(),
            seed=cfg.get("seed", 42),
            report_to=[],
            max_length=cfg.get("max_seq_length", 2048),
        )
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer, callbacks=[ProgressCallback()])
    elif objective == "orpo":
        raise ValueError(
            "ORPO is not supported by the installed TRL version (ORPOTrainer/ORPOConfig were removed upstream). "
            "Use DPO or KTO instead, or install an older trl release that still provides ORPOTrainer."
        )
    elif objective == "kto":
        from trl import KTOConfig, KTOTrainer

        # KTO expects {"prompt","completion","label"} — split chosen/rejected into binary-labeled rows.
        kto_prompts = prompts + prompts
        kto_completions = chosen + rejected
        kto_labels = [True] * len(chosen) + [False] * len(rejected)
        ds = HFDataset.from_dict({"prompt": kto_prompts, "completion": kto_completions, "label": kto_labels})
        args = KTOConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            num_train_epochs=cfg.get("num_train_epochs", 3.0),
            max_steps=max_steps if max_steps and max_steps > 0 else -1,
            learning_rate=cfg.get("learning_rate", 2e-4),
            beta=cfg.get("beta", 0.1),
            logging_steps=cfg.get("logging_steps", 5),
            save_steps=cfg.get("save_steps", 50),
            save_total_limit=cfg.get("save_total_limit", 3),
            bf16=torch.cuda.is_available(),
            seed=cfg.get("seed", 42),
            report_to=[],
        )
        trainer = KTOTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer, callbacks=[ProgressCallback()])
    else:
        raise ValueError(f"Unsupported preference objective: {objective}")

    emit(status_path, "stage", stage="training")
    trainer.train()

    emit(status_path, "stage", stage="saving_final")
    final_dir = str(Path(cfg["output_dir"]) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    emit(status_path, "done", final_dir=final_dir)


def main() -> None:
    run_id, config_path, status_path_str = sys.argv[1], sys.argv[2], sys.argv[3]
    status_path = Path(status_path_str)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    emit(status_path, "started", run_id=run_id)
    try:
        if cfg.get("objective", "sft") == "sft":
            run_sft(cfg, status_path)
        else:
            run_preference(cfg, status_path)
    except Exception as e:  # noqa: BLE001
        emit(status_path, "error", message=str(e), traceback=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
