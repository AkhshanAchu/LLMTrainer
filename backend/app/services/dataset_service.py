"""Dataset ingestion, format detection, normalization, validation, statistics,
tokenizer-aware analysis, auto-repair and runtime estimation for FineTune Studio.

This module contains pure/CPU-bound logic only - no FastAPI/DB imports - so it
can be unit tested and called from routes via `run_in_threadpool`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.models.db import DatasetFormat

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MAX_ROWS_FOR_STATS = 200_000
DETECTION_SAMPLE_SIZE = 200
DEFAULT_FALLBACK_TOKENIZER = "gpt2"
DEFAULT_THROUGHPUT_TOK_S = 2000.0  # rough single consumer GPU QLoRA throughput

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INST_TAG_RE = re.compile(r"\[/?INST\]|<\|.*?\|>|<<SYS>>|<</SYS>>")

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt", ".parquet"}

try:
    import pyarrow.parquet as _pq  # noqa: F401

    _HAS_PARQUET = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_PARQUET = False

try:
    from langdetect import DetectorFactory, detect as _langdetect_detect

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_LANGDETECT = False


# --------------------------------------------------------------------------
# File ingestion
# --------------------------------------------------------------------------


class UnsupportedFileTypeError(ValueError):
    pass


def save_upload(uploads_dir: Path, original_filename: str, content: bytes) -> Path:
    """Persist raw upload bytes to disk under a uuid-prefixed name."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(original_filename).name
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    stored_path = uploads_dir / stored_name
    stored_path.write_bytes(content)
    return stored_path


def parse_file(path: Path, max_rows: int = MAX_ROWS_FOR_STATS) -> list[dict]:
    """Parse a dataset file into a normalized list-of-dict records.

    Supported: .json (list of objects, or single object), .jsonl, .csv, .txt
    (one example per line -> {"text": line}), .parquet (if pyarrow available).
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".jsonl":
        return _parse_jsonl(path, max_rows)
    if suffix == ".json":
        return _parse_json(path, max_rows)
    if suffix == ".csv":
        return _parse_csv(path, max_rows)
    if suffix == ".txt":
        return _parse_txt(path, max_rows)
    if suffix == ".parquet":
        if not _HAS_PARQUET:
            raise UnsupportedFileTypeError(
                "Parquet support requires 'pyarrow', which is not installed."
            )
        return _parse_parquet(path, max_rows)
    raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'")


def _parse_jsonl(path: Path, max_rows: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(records) >= max_rows:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                records.append({"value": obj})
    return records


def _parse_json(path: Path, max_rows: int) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    records: list[dict] = []
    if isinstance(data, list):
        for item in data[:max_rows]:
            records.append(item if isinstance(item, dict) else {"value": item})
    elif isinstance(data, dict):
        # Could be {"data": [...]} wrapper, or a single record.
        for key in ("data", "examples", "records", "rows"):
            if key in data and isinstance(data[key], list):
                for item in data[key][:max_rows]:
                    records.append(item if isinstance(item, dict) else {"value": item})
                return records
        records.append(data)
    return records


def _parse_csv(path: Path, max_rows: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            records.append(dict(row))
    return records


def _parse_txt(path: Path, max_rows: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if len(records) >= max_rows:
                break
            records.append({"text": line})
    return records


def _parse_parquet(path: Path, max_rows: int) -> list[dict]:
    table = _pq.read_table(path)
    n = min(max_rows, table.num_rows)
    table = table.slice(0, n)
    return table.to_pylist()


# --------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------


@dataclass
class FormatDetectionResult:
    format: DatasetFormat
    confidence: float
    signals: dict = field(default_factory=dict)


def detect_format(records: list[dict]) -> FormatDetectionResult:
    """Sample the first ~200 records and score each known format."""
    if not records:
        return FormatDetectionResult(DatasetFormat.unknown, 0.0, {"reason": "no records"})

    sample = records[: min(len(records), DETECTION_SAMPLE_SIZE)]
    n = len(sample)

    scores = {fmt: 0 for fmt in DatasetFormat}

    for rec in sample:
        if not isinstance(rec, dict):
            continue
        keys = set(rec.keys())

        if "conversations" in keys and isinstance(rec.get("conversations"), list):
            convs = rec["conversations"]
            if convs and all(
                isinstance(c, dict) and ("from" in c or "role" in c) for c in convs
            ):
                scores[DatasetFormat.sharegpt] += 1
                continue

        if "messages" in keys and isinstance(rec.get("messages"), list):
            msgs = rec["messages"]
            if msgs and all(isinstance(m, dict) and "role" in m for m in msgs):
                scores[DatasetFormat.chatml] += 1
                continue

        if "chosen" in keys and "rejected" in keys:
            scores[DatasetFormat.preference] += 1
            continue

        if "instruction" in keys and ("output" in keys or "response" in keys):
            scores[DatasetFormat.alpaca] += 1
            continue

        if "text" in keys and len(keys) <= 2:
            text_val = rec.get("text")
            if isinstance(text_val, str) and _INST_TAG_RE.search(text_val):
                scores[DatasetFormat.llama] += 1
            elif isinstance(text_val, str):
                scores[DatasetFormat.raw_text] += 1
            continue

    best_fmt = max(scores, key=lambda f: scores[f])
    best_count = scores[best_fmt]

    if best_count == 0:
        return FormatDetectionResult(
            DatasetFormat.unknown, 0.0, {"scores": {k.value: v for k, v in scores.items()}}
        )

    confidence = round(best_count / n, 4)
    return FormatDetectionResult(
        best_fmt, confidence, {"scores": {k.value: v for k, v in scores.items()}, "sampled": n}
    )


# --------------------------------------------------------------------------
# Normalization -> canonical chat messages
# --------------------------------------------------------------------------

_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "bot": "assistant",
}


def to_chat_messages(record: dict, fmt: DatasetFormat) -> list[dict]:
    """Convert a single record of any supported format into canonical
    `[{"role": ..., "content": ...}]` messages."""
    if not isinstance(record, dict):
        return []

    if fmt == DatasetFormat.sharegpt or "conversations" in record:
        convs = record.get("conversations") or []
        out = []
        for c in convs:
            if not isinstance(c, dict):
                continue
            role_raw = str(c.get("from") or c.get("role") or "user").lower()
            role = _SHAREGPT_ROLE_MAP.get(role_raw, "user")
            content = c.get("value", c.get("content", ""))
            out.append({"role": role, "content": str(content)})
        return out

    if fmt in (DatasetFormat.chatml, DatasetFormat.openai) or "messages" in record:
        msgs = record.get("messages") or []
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user")).lower()
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            out.append({"role": role, "content": str(m.get("content", ""))})
        return out

    if fmt == DatasetFormat.preference or ("chosen" in record and "rejected" in record):
        # Represent as a prompt + chosen response for chat-shaped previews.
        prompt = record.get("prompt") or record.get("instruction") or record.get("input") or ""
        chosen = record.get("chosen", "")
        out = []
        if prompt:
            out.append({"role": "user", "content": str(prompt)})
        # `chosen`/`rejected` may themselves be message lists (common DPO shape)
        if isinstance(chosen, list):
            for m in chosen:
                if isinstance(m, dict):
                    out.append(
                        {"role": str(m.get("role", "assistant")), "content": str(m.get("content", ""))}
                    )
        else:
            out.append({"role": "assistant", "content": str(chosen)})
        return out

    if fmt == DatasetFormat.alpaca or "instruction" in record:
        instruction = str(record.get("instruction", ""))
        inp = str(record.get("input", "") or "")
        output = str(record.get("output", record.get("response", "")) or "")
        user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction
        out = [{"role": "user", "content": user_content}]
        if output:
            out.append({"role": "assistant", "content": output})
        return out

    if fmt == DatasetFormat.llama or ("text" in record and _INST_TAG_RE.search(str(record.get("text", "")))):
        text = str(record.get("text", ""))
        return _parse_llama_text(text)

    if "text" in record:
        return [{"role": "user", "content": str(record.get("text", ""))}]

    # Fallback: dump whatever fields exist as a single user message.
    if record:
        content = json.dumps(record, ensure_ascii=False)
        return [{"role": "user", "content": content}]
    return []


def _parse_llama_text(text: str) -> list[dict]:
    """Best-effort split of `[INST] ... [/INST] response` style text into turns."""
    out: list[dict] = []
    # Split on [INST]...[/INST] pairs
    pattern = re.compile(r"\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|$)", re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        for m in matches:
            user_part = m.group(1).strip()
            # strip a leading <<SYS>>...<</SYS>> block into a system message
            sys_match = re.match(r"<<SYS>>(.*?)<</SYS>>\s*(.*)", user_part, re.DOTALL)
            if sys_match:
                out.append({"role": "system", "content": sys_match.group(1).strip()})
                user_part = sys_match.group(2).strip()
            if user_part:
                out.append({"role": "user", "content": user_part})
            assistant_part = m.group(2).strip()
            if assistant_part:
                out.append({"role": "assistant", "content": assistant_part})
        return out
    # No recognizable tags matched despite detection; treat as raw text.
    return [{"role": "user", "content": text}]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _hash_content(messages: list[dict]) -> str:
    normalized = json.dumps(
        [{"role": m.get("role"), "content": m.get("content", "").strip()} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def validate_dataset(records: list[dict], fmt: DatasetFormat) -> dict:
    """Run structural / quality checks over the dataset and return a JSON-safe report."""
    categories = {
        "missing_fields": [],
        "role_mismatch": [],
        "invalid_conversation": [],
        "duplicate_samples": [],
        "empty_responses": [],
        "broken_utf8": [],
        "unsupported_tokens": [],
    }

    seen_hashes: dict[str, int] = {}

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict) or not rec:
            categories["invalid_conversation"].append(idx)
            continue

        # missing fields per format
        if fmt == DatasetFormat.alpaca and "instruction" not in rec:
            categories["missing_fields"].append(idx)
        elif fmt == DatasetFormat.sharegpt and "conversations" not in rec:
            categories["missing_fields"].append(idx)
        elif fmt in (DatasetFormat.chatml, DatasetFormat.openai) and "messages" not in rec:
            categories["missing_fields"].append(idx)
        elif fmt == DatasetFormat.preference and ("chosen" not in rec or "rejected" not in rec):
            categories["missing_fields"].append(idx)

        try:
            messages = to_chat_messages(rec, fmt)
        except Exception:
            categories["invalid_conversation"].append(idx)
            continue

        if not messages:
            categories["invalid_conversation"].append(idx)
            continue

        # role mismatch: first message should be system/user, no consecutive same roles
        if messages[0]["role"] not in ("system", "user"):
            categories["role_mismatch"].append(idx)
        else:
            for a, b in zip(messages, messages[1:]):
                if a["role"] == b["role"] and a["role"] != "system":
                    categories["role_mismatch"].append(idx)
                    break

        # empty responses: any assistant message with blank content, or no assistant turn at all
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs or any(not m["content"].strip() for m in assistant_msgs):
            categories["empty_responses"].append(idx)

        # broken utf-8 / control chars
        joined = "\n".join(m.get("content", "") for m in messages)
        if "�" in joined:
            categories["broken_utf8"].append(idx)
        if _CONTROL_CHAR_RE.search(joined) or "\x00" in joined:
            categories["unsupported_tokens"].append(idx)

        # duplicates
        h = _hash_content(messages)
        if h in seen_hashes:
            categories["duplicate_samples"].append(idx)
        else:
            seen_hashes[h] = idx

    issues_found = sum(len(v) for v in categories.values())

    report: dict[str, Any] = {"issues_found": issues_found, "num_examples_checked": len(records)}
    for cat, indices in categories.items():
        report[cat] = {
            "count": len(indices),
            "sample_indices": indices[:5],
        }
    return report


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1.0
    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


def _detect_language(sample_texts: list[str]) -> dict:
    if not _HAS_LANGDETECT:
        return {"available": False, "note": "langdetect not installed; skipped"}
    lang_counts: Counter = Counter()
    checked = 0
    for text in sample_texts[:100]:
        text = text.strip()
        if len(text) < 8:
            continue
        try:
            lang_counts[_langdetect_detect(text)] += 1
            checked += 1
        except Exception:
            continue
    if checked == 0:
        return {"available": True, "note": "insufficient text to detect language"}
    top = lang_counts.most_common(5)
    return {
        "available": True,
        "distribution": {lang: count for lang, count in top},
        "sampled": checked,
    }


def _bucket_histogram(values: list[int], buckets: list[tuple[int, int | None]]) -> dict:
    hist = {}
    for lo, hi in buckets:
        label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
        hist[label] = 0
    for v in values:
        for lo, hi in buckets:
            if v >= lo and (hi is None or v < hi):
                label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
                hist[label] += 1
                break
    return hist


TOKEN_BUCKETS = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, None)]
TURN_BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, None)]


def compute_stats(records: list[dict], fmt: DatasetFormat) -> dict:
    """Compute dataset-level statistics without requiring a tokenizer.

    Word-level token approximation is used here; real tokenizer-based counts
    are computed separately via `analyze_with_tokenizer`.
    """
    n = len(records)
    if n == 0:
        return {"num_examples": 0, "note": "empty dataset"}

    turn_counts: list[int] = []
    approx_token_counts: list[int] = []
    role_counter: Counter = Counter()
    word_counter: Counter = Counter()
    all_texts: list[str] = []
    hashes: Counter = Counter()
    empty_response_count = 0
    role_mismatch_count = 0

    for rec in records:
        try:
            messages = to_chat_messages(rec, fmt)
        except Exception:
            messages = []
        turn_counts.append(len(messages))

        joined = " ".join(m.get("content", "") for m in messages)
        all_texts.append(joined)
        words = joined.split()
        approx_token_counts.append(len(words))
        word_counter.update(w.lower() for w in words[:5000])

        for m in messages:
            role_counter[m["role"]] += 1

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs or any(not m["content"].strip() for m in assistant_msgs):
            empty_response_count += 1

        if messages:
            if messages[0]["role"] not in ("system", "user"):
                role_mismatch_count += 1
            else:
                for a, b in zip(messages, messages[1:]):
                    if a["role"] == b["role"] and a["role"] != "system":
                        role_mismatch_count += 1
                        break

        if messages:
            hashes[_hash_content(messages)] += 1

    duplicate_groups = [c for c in hashes.values() if c > 1]
    duplicate_examples = sum(c - 1 for c in duplicate_groups)
    duplicate_ratio = round(duplicate_examples / n, 4) if n else 0.0

    user_count = role_counter.get("user", 0)
    assistant_count = role_counter.get("assistant", 0)
    ratio = round(assistant_count / user_count, 4) if user_count else None

    empty_frac = empty_response_count / n
    dup_frac = duplicate_ratio
    role_mismatch_frac = role_mismatch_count / n

    quality_score = 100.0
    quality_score -= empty_frac * 40
    quality_score -= dup_frac * 30
    quality_score -= role_mismatch_frac * 30
    quality_score = max(0.0, min(100.0, round(quality_score, 2)))

    entropy_score = _shannon_entropy(word_counter)

    stats = {
        "num_examples": n,
        "conversation_turns": {
            "avg": round(sum(turn_counts) / n, 3),
            "min": min(turn_counts),
            "max": max(turn_counts),
        },
        "role_counts": dict(role_counter),
        "assistant_user_ratio": ratio,
        "duplicate_ratio": duplicate_ratio,
        "duplicate_examples": duplicate_examples,
        "language": _detect_language(all_texts),
        "token_histogram": {
            "method": "whitespace_approx",
            "note": "Approximate word-split counts; use /tokenize-preview for real tokenizer counts.",
            "buckets": _bucket_histogram(approx_token_counts, TOKEN_BUCKETS),
            "avg": round(sum(approx_token_counts) / n, 2),
            "min": min(approx_token_counts),
            "max": max(approx_token_counts),
        },
        "conversation_length_histogram": {
            "buckets": _bucket_histogram(turn_counts, TURN_BUCKETS),
        },
        "quality_score": quality_score,
        "entropy_score": entropy_score,
    }
    return stats


# --------------------------------------------------------------------------
# Tokenizer-aware analysis
# --------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(repo_id: str):
    if repo_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[repo_id]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    _TOKENIZER_CACHE[repo_id] = tok
    return tok


def _format_prompt(tokenizer, messages: list[dict]) -> tuple[str, list[int]]:
    """Return (formatted_prompt_string, token_ids) using chat template if available."""
    if not messages:
        return "", []
    try:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return prompt_str, token_ids
    except Exception:
        pass
    # Fallback: simple role-tagged concatenation
    prompt_str = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    token_ids = tokenizer.encode(prompt_str)
    return prompt_str, token_ids


def analyze_with_tokenizer(
    records: list[dict],
    fmt: DatasetFormat,
    tokenizer_repo_id: str | None = None,
    max_seq_len: int = 2048,
    sample_size: int | None = None,
) -> dict:
    """Real tokenizer-based analysis: token counts, truncation preview, epoch estimate."""
    repo_id = tokenizer_repo_id or DEFAULT_FALLBACK_TOKENIZER
    used_fallback = False
    try:
        tokenizer = _get_tokenizer(repo_id)
    except Exception as e:
        if repo_id != DEFAULT_FALLBACK_TOKENIZER:
            try:
                tokenizer = _get_tokenizer(DEFAULT_FALLBACK_TOKENIZER)
                repo_id = DEFAULT_FALLBACK_TOKENIZER
                used_fallback = True
            except Exception:
                return {"error": f"Failed to load tokenizer '{tokenizer_repo_id}' and fallback: {e}"}
        else:
            return {"error": f"Failed to load tokenizer '{repo_id}': {e}"}

    sample = records if sample_size is None else records[:sample_size]
    token_counts: list[int] = []
    first_prompt_str = None
    first_token_ids: list[int] | None = None
    truncated_count = 0

    for i, rec in enumerate(sample):
        try:
            messages = to_chat_messages(rec, fmt)
        except Exception:
            messages = []
        prompt_str, token_ids = _format_prompt(tokenizer, messages)
        n_tokens = len(token_ids)
        token_counts.append(n_tokens)
        if n_tokens > max_seq_len:
            truncated_count += 1
        if i == 0:
            first_prompt_str = prompt_str
            first_token_ids = token_ids

    n = len(token_counts)
    total_tokens_sample = sum(token_counts)
    avg_tokens = round(total_tokens_sample / n, 2) if n else 0.0
    # Extrapolate to full dataset if we only analyzed a sample
    scale = (len(records) / n) if n and n < len(records) else 1.0
    estimated_total_tokens = int(round(total_tokens_sample * scale))

    result = {
        "tokenizer_repo_id": repo_id,
        "used_fallback_tokenizer": used_fallback,
        "sampled_examples": n,
        "total_examples": len(records),
        "avg_tokens_per_example": avg_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "estimated_total_tokens": estimated_total_tokens,
        "token_histogram": _bucket_histogram(token_counts, TOKEN_BUCKETS),
        "max_seq_len": max_seq_len,
        "truncation_preview": {
            "sampled_examples_over_limit": truncated_count,
            "fraction_over_limit": round(truncated_count / n, 4) if n else 0.0,
            "estimated_total_over_limit": int(round((truncated_count / n) * len(records))) if n else 0,
        },
        "first_example": {
            "formatted_prompt": first_prompt_str,
            "token_count": len(first_token_ids) if first_token_ids is not None else 0,
        },
    }
    return result


# --------------------------------------------------------------------------
# Auto-repair
# --------------------------------------------------------------------------


def auto_repair(records: list[dict], fmt: DatasetFormat, options: dict | None = None) -> tuple[list[dict], dict]:
    """Repair common dataset issues. Returns (repaired_records_in_original_shape, summary).

    Options (all optional, default True/None unless noted):
      - fix_roles: bool = True
      - merge_consecutive_same_role: bool = True
      - remove_duplicates: bool = True
      - remove_empty_responses: bool = True
      - max_turn_chars: int | None = None   (truncate each turn's content to this many chars)
      - split_oversized: int | None = None  (max turns per example; overflow spawns new examples)
      - shuffle: bool = False
      - shuffle_seed: int = 42
    """
    options = options or {}
    fix_roles = options.get("fix_roles", True)
    merge_same_role = options.get("merge_consecutive_same_role", True)
    remove_duplicates = options.get("remove_duplicates", True)
    remove_empty_responses = options.get("remove_empty_responses", True)
    max_turn_chars = options.get("max_turn_chars")
    split_oversized = options.get("split_oversized")
    shuffle = options.get("shuffle", False)
    shuffle_seed = options.get("shuffle_seed", 42)

    summary = {
        "input_count": len(records),
        "roles_fixed": 0,
        "merged_turns": 0,
        "duplicates_removed": 0,
        "empty_responses_removed": 0,
        "turns_truncated": 0,
        "examples_split": 0,
        "shuffled": False,
    }

    repaired: list[dict] = []
    seen_hashes: set[str] = set()

    for rec in records:
        try:
            messages = to_chat_messages(rec, fmt)
        except Exception:
            messages = []
        if not messages:
            continue

        # fix roles: drop leading turns that aren't system/user
        if fix_roles:
            original_len = len(messages)
            while messages and messages[0]["role"] not in ("system", "user"):
                messages.pop(0)
            if len(messages) != original_len:
                summary["roles_fixed"] += 1

        # merge consecutive same-role messages
        if merge_same_role and messages:
            merged: list[dict] = [dict(messages[0])]
            merged_happened = False
            for m in messages[1:]:
                if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
                    merged[-1]["content"] = (merged[-1]["content"] + "\n" + m["content"]).strip()
                    merged_happened = True
                else:
                    merged.append(dict(m))
            messages = merged
            if merged_happened:
                summary["merged_turns"] += 1

        # remove empty assistant responses
        if remove_empty_responses:
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            if not assistant_msgs or any(not m["content"].strip() for m in assistant_msgs):
                summary["empty_responses_removed"] += 1
                continue

        # truncate long turns
        if max_turn_chars:
            for m in messages:
                if len(m["content"]) > max_turn_chars:
                    m["content"] = m["content"][:max_turn_chars]
                    summary["turns_truncated"] += 1

        # split oversized examples into chunks of `split_oversized` turns
        chunks = [messages]
        if split_oversized and len(messages) > split_oversized:
            chunks = [
                messages[i : i + split_oversized] for i in range(0, len(messages), split_oversized)
            ]
            summary["examples_split"] += len(chunks) - 1

        for chunk in chunks:
            if not chunk:
                continue
            if remove_duplicates:
                h = _hash_content(chunk)
                if h in seen_hashes:
                    summary["duplicates_removed"] += 1
                    continue
                seen_hashes.add(h)
            repaired.append({"messages": chunk})

    if shuffle:
        import random

        rng = random.Random(shuffle_seed)
        rng.shuffle(repaired)
        summary["shuffled"] = True

    summary["output_count"] = len(repaired)
    return repaired, summary


# --------------------------------------------------------------------------
# Runtime / VRAM-adjacent estimates (token/time based; VRAM itself handled
# elsewhere per the GPU estimate schema, this focuses on dataset throughput)
# --------------------------------------------------------------------------


def estimate_training_runtime(
    total_tokens: int,
    epochs: float = 1.0,
    tokens_per_sec: float = DEFAULT_THROUGHPUT_TOK_S,
) -> dict:
    """Heuristic wall-clock estimate given a fixed tokens/sec throughput assumption.

    This is intentionally simple: real throughput depends heavily on model size,
    sequence length, batch size, GPU, and quantization. Callers should treat this
    as a ballpark, not a guarantee.
    """
    tokens_per_sec = max(tokens_per_sec, 1e-6)
    total_training_tokens = total_tokens * epochs
    seconds = total_training_tokens / tokens_per_sec
    hours = seconds / 3600.0
    return {
        "total_dataset_tokens": total_tokens,
        "epochs": epochs,
        "total_training_tokens": int(round(total_training_tokens)),
        "assumed_tokens_per_sec": tokens_per_sec,
        "estimate_runtime_hours": round(hours, 3),
        "note": "Heuristic estimate assuming constant token throughput; actual speed varies with model size, seq_len, batch size, GPU, and quantization.",
    }


def estimate_epochs_for_target_tokens(total_tokens_per_epoch: int, target_tokens: int) -> float:
    if total_tokens_per_epoch <= 0:
        return 0.0
    return round(target_tokens / total_tokens_per_epoch, 4)


def estimate_dataset_runtime(
    records: list[dict],
    fmt: DatasetFormat,
    seq_len: int = 2048,
    batch_size: int = 1,
    epochs: float = 1.0,
    tokenizer_repo_id: str | None = None,
    tokens_per_sec: float = DEFAULT_THROUGHPUT_TOK_S,
    target_tokens: int | None = None,
) -> dict:
    """Combine tokenizer analysis + runtime heuristic into one estimate payload."""
    analysis = analyze_with_tokenizer(
        records, fmt, tokenizer_repo_id=tokenizer_repo_id, max_seq_len=seq_len
    )
    if "error" in analysis:
        return analysis

    total_tokens = analysis["estimated_total_tokens"]
    runtime = estimate_training_runtime(total_tokens, epochs=epochs, tokens_per_sec=tokens_per_sec)
    result = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        **runtime,
        "tokenizer_repo_id": analysis["tokenizer_repo_id"],
        "avg_tokens_per_example": analysis["avg_tokens_per_example"],
        "truncation_preview": analysis["truncation_preview"],
    }
    if target_tokens:
        result["estimated_epochs_for_target_tokens"] = estimate_epochs_for_target_tokens(
            total_tokens, target_tokens
        )
    return result
