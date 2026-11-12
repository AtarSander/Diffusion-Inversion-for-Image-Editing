"""Prepare a deterministic, non-photographic DiffusionDB prompt manifest."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPTokenizerFast

_STYLE_ORDER = (
    "illustration_vector",
    "cartoon_comic_anime",
    "painting_digital_art",
    "render_3d",
    "graphic_design",
)


def normalize_prompt(prompt: str) -> str:
    """Normalize a prompt only for duplicate detection; keep the source text for generation."""
    normalized = unicodedata.normalize("NFKC", prompt).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.strip(".,;:!?\\\"'`()[]{}<>")


def non_latin_dominated(prompt: str, threshold: float) -> bool:
    """Return true when more than ``threshold`` of Unicode letters are non-Latin.

    The heuristic deliberately considers letters only: punctuation, emoji, and digits do not
    make a mostly Latin prompt look non-Latin. Prompts with fewer than four letters pass.
    """
    letters = [char for char in prompt if unicodedata.category(char).startswith("L")]
    if len(letters) < 4:
        return False
    latin_letters = sum("LATIN" in unicodedata.name(char, "") for char in letters)
    return (1.0 - latin_letters / len(letters)) > threshold


def contains_repeated_token_spam(prompt: str, min_tokens: int, fraction: float) -> bool:
    tokens = re.findall(r"\b\w+\b", normalize_prompt(prompt))
    if len(tokens) < min_tokens:
        return False
    return max(Counter(tokens).values(), default=0) / len(tokens) >= fraction


def _phrase_matches(prompt: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", prompt.lower()) is not None


def matched_style_labels(prompt: str, style_keywords: Mapping[str, Sequence[str]]) -> list[str]:
    return [
        label
        for label in _STYLE_ORDER
        if any(_phrase_matches(prompt, phrase) for phrase in style_keywords.get(label, ()))
    ]


def _is_obvious_boilerplate(prompt: str, exact_phrases: Sequence[str]) -> bool:
    return normalize_prompt(prompt) in {normalize_prompt(phrase) for phrase in exact_phrases}


def _is_bad_text(prompt: str, cfg: Mapping[str, Any]) -> bool:
    lowered = prompt.lower()
    if any(marker in lowered for marker in cfg["url_markers"]):
        return True
    if any(marker in lowered for marker in cfg["discord_markers"]):
        return True
    if _is_obvious_boilerplate(prompt, cfg["boilerplate_exact"]):
        return True
    return contains_repeated_token_spam(
        prompt,
        min_tokens=int(cfg["repeated_token_min_tokens"]),
        fraction=float(cfg["repeated_token_fraction"]),
    )


def _has_photographic_target(prompt: str, terms: Sequence[str]) -> bool:
    return any(_phrase_matches(prompt, term) for term in terms)


def select_artificial_prompts(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    token_length: Callable[[str], int] | None = None,
    token_lengths: Callable[[Sequence[str]], Sequence[int]] | None = None,
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Filter, stratify, and deterministically sample DiffusionDB prompt rows.

    ``token_lengths`` keeps the SD1.5 tokenizer cutoff exact while batching the expensive
    tokenizer work. ``token_length`` remains available for small callers and unit tests.
    """
    if token_lengths is None:
        if token_length is None:
            raise ValueError("Provide token_length or token_lengths.")
        def token_lengths(prompts: Sequence[str]) -> Sequence[int]:
            return [token_length(prompt) for prompt in prompts]

    seed = int(cfg["selection_seed"])
    count_per_stratum = int(cfg["count_per_stratum"])
    seen: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_row_count = 0
    pending: list[tuple[int, Mapping[str, Any], str]] = []

    def classify_pending() -> None:
        if not pending:
            return
        lengths = token_lengths([prompt for _, _, prompt in pending])
        if len(lengths) != len(pending):
            raise ValueError("Batched tokenizer returned an unexpected number of lengths.")
        for (source_row_id, row, prompt), length in zip(pending, lengths):
            if int(length) > int(cfg["max_clip_tokens"]):
                rejection_counts["too_many_clip_tokens"] += 1
                continue
            if non_latin_dominated(prompt, float(cfg["non_latin_letter_fraction_threshold"])):
                rejection_counts["non_latin_dominated"] += 1
                continue
            if _is_bad_text(prompt, cfg):
                rejection_counts["boilerplate_url_or_spam"] += 1
                continue
            if _has_photographic_target(prompt, cfg["photographic_terms"]):
                rejection_counts["photographic_target"] += 1
                continue
            labels = matched_style_labels(prompt, cfg["style_keywords"])
            if not labels:
                rejection_counts["no_artificial_style"] += 1
                continue
            primary_label = labels[0]
            eligible[primary_label].append(
                {
                    "prompt": prompt,
                    "source_dataset": "diffusiondb",
                    "style_stratum": primary_label,
                    "style_matched_labels": labels,
                    "source_row_id": int(source_row_id),
                    "selection_seed": seed,
                    "width": row.get("width"),
                    "height": row.get("height"),
                }
            )
        pending.clear()

    for source_row_id, row in rows:
        input_row_count += 1
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            rejection_counts["empty_prompt"] += 1
            continue
        if len(prompt.split()) < int(cfg["min_whitespace_words"]):
            rejection_counts["too_few_words"] += 1
            continue
        normalized = normalize_prompt(prompt)
        if not normalized:
            rejection_counts["empty_after_normalization"] += 1
            continue
        if normalized in seen:
            rejection_counts["duplicate"] += 1
            continue
        seen.add(normalized)

        prompt_nsfw = row.get("prompt_nsfw")
        image_nsfw = row.get("image_nsfw")
        if prompt_nsfw is None or image_nsfw is None:
            rejection_counts["nsfw_missing"] += 1
            continue
        try:
            prompt_nsfw_value = float(prompt_nsfw)
            image_nsfw_value = float(image_nsfw)
            if not math.isfinite(prompt_nsfw_value) or not math.isfinite(image_nsfw_value):
                rejection_counts["nsfw_missing"] += 1
                continue
            if (
                prompt_nsfw_value >= float(cfg["nsfw_threshold"])
                or image_nsfw_value >= float(cfg["nsfw_threshold"])
            ):
                rejection_counts["nsfw"] += 1
                continue
        except (TypeError, ValueError):
            rejection_counts["nsfw_missing"] += 1
            continue
        pending.append((source_row_id, row, prompt))
        if len(pending) >= 1_024:
            classify_pending()
    classify_pending()

    selected: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(_STYLE_ORDER):
        candidates = eligible[stratum]
        if len(candidates) < count_per_stratum:
            raise ValueError(
                f"Only {len(candidates)} eligible prompts for {stratum}; "
                f"need {count_per_stratum}."
            )
        chosen = random.Random(seed + stratum_index).sample(candidates, count_per_stratum)
        selected.extend(sorted(chosen, key=lambda record: record["source_row_id"]))

    review_records: list[dict[str, Any]] = []
    review_count = int(cfg["review_count_per_stratum"])
    for stratum_index, stratum in enumerate(_STYLE_ORDER):
        stratum_selected = [record for record in selected if record["style_stratum"] == stratum]
        if len(stratum_selected) < review_count:
            raise ValueError(f"Selected records for {stratum} are fewer than review_count.")
        review_records.extend(
            sorted(
                random.Random(seed + 10_000 + stratum_index).sample(stratum_selected, review_count),
                key=lambda record: record["source_row_id"],
            )
        )

    summary = {
        "input_row_count": input_row_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "duplicate_count": int(rejection_counts["duplicate"]),
        "eligible_count_per_stratum": {stratum: len(eligible[stratum]) for stratum in _STYLE_ORDER},
        "selected_count_per_stratum": {
            stratum: sum(record["style_stratum"] == stratum for record in selected)
            for stratum in _STYLE_ORDER
        },
        "selected_count": len(selected),
        "selection_seed": seed,
        "filter_policy": {
            "clip_token_count": "CLIP input_ids with add_special_tokens=False",
            "non_latin_heuristic": (
                "reject when more than "
                f"{cfg['non_latin_letter_fraction_threshold']:.0%} of Unicode letter characters "
                "are non-Latin; prompts with fewer than four letters pass"
            ),
        },
    }
    return selected, summary, review_records

def _iter_parquet_rows(metadata_path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError("pyarrow is required to read DiffusionDB metadata.parquet.") from exc

    columns = ["prompt", "prompt_nsfw", "image_nsfw", "width", "height"]
    parquet = pq.ParquetFile(metadata_path)
    row_id = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=16_384):
        for row in batch.to_pylist():
            yield row_id, row
            row_id += 1


def _write_jsonl(records: Sequence[Mapping[str, Any]], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def _token_lengths_fn(tokenizer: CLIPTokenizerFast) -> Callable[[Sequence[str]], Sequence[int]]:
    def token_lengths(prompts: Sequence[str]) -> Sequence[int]:
        encoded = tokenizer(
            list(prompts),
            add_special_tokens=False,
            truncation=False,
            padding=False,
            verbose=False,
        )
        return [len(input_ids) for input_ids in encoded["input_ids"]]

    return token_lengths


@hydra.main(
    config_path="../../config",
    config_name="prepare_diffusiondb_artificial_prompts",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    metadata_path = Path(to_absolute_path(str(cfg.data.metadata_parquet)))
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"DiffusionDB metadata parquet is missing: {metadata_path}. "
            "Download only metadata.parquet from poloclub/diffusiondb; do not download image ZIPs."
        )

    tokenizer = CLIPTokenizerFast.from_pretrained(str(cfg.tokenizer_model_id))
    selected, summary, review_records = select_artificial_prompts(
        _iter_parquet_rows(metadata_path),
        token_lengths=_token_lengths_fn(tokenizer),
        cfg=OmegaConf.to_container(cfg.selection, resolve=True),
    )
    output_path = Path(to_absolute_path(str(cfg.data.prepared_prompts_jsonl)))
    review_path = Path(to_absolute_path(str(cfg.data.manual_review_jsonl)))
    summary_path = Path(to_absolute_path(str(cfg.data.summary_json)))
    summary["manifest_sha256"] = _write_jsonl(selected, output_path)
    _write_jsonl(review_records, review_path)
    summary["manual_review_sample_count"] = len(review_records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} prompts: {output_path}")
    print(f"Review exactly {len(review_records)} prompts before generation: {review_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
