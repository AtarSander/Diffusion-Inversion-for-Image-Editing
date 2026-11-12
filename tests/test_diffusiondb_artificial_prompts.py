from __future__ import annotations

from diff_inversion.data.prepare_diffusiondb_artificial_prompts import select_artificial_prompts


def _cfg() -> dict:
    return {
        "selection_seed": 42,
        "count_per_stratum": 2,
        "review_count_per_stratum": 1,
        "min_whitespace_words": 4,
        "max_clip_tokens": 75,
        "nsfw_threshold": 0.1,
        "non_latin_letter_fraction_threshold": 0.5,
        "repeated_token_min_tokens": 8,
        "repeated_token_fraction": 0.6,
        "url_markers": ["http://", "https://", "www."],
        "discord_markers": ["discord.gg/", "<@"],
        "boilerplate_exact": ["test"],
        "photographic_terms": ["photograph", "dslr", "35mm"],
        "style_keywords": {
            "illustration_vector": ["illustration", "vector art"],
            "cartoon_comic_anime": ["cartoon", "anime"],
            "painting_digital_art": ["painting", "digital art"],
            "render_3d": ["3d render", "cgi"],
            "graphic_design": ["poster", "diagram"],
        },
    }


def _row(prompt: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": prompt,
        "prompt_nsfw": 0.0,
        "image_nsfw": 0.0,
        "width": 512,
        "height": 512,
    }
    row.update(overrides)
    return row


def test_selection_is_deterministic_and_has_exact_stratum_counts() -> None:
    rows = []
    styles = ["illustration", "cartoon", "painting", "3d render", "poster"]
    for style in styles:
        for index in range(3):
            rows.append((len(rows), _row(f"a detailed {style} scene with subject {index}")))

    selected_a, summary_a, review_a = select_artificial_prompts(
        rows, token_length=lambda _: 4, cfg=_cfg()
    )
    selected_b, summary_b, review_b = select_artificial_prompts(
        rows, token_length=lambda _: 4, cfg=_cfg()
    )

    assert selected_a == selected_b
    assert summary_a == summary_b
    assert review_a == review_b
    assert len(selected_a) == 10
    assert summary_a["selected_count_per_stratum"] == {
        "illustration_vector": 2,
        "cartoon_comic_anime": 2,
        "painting_digital_art": 2,
        "render_3d": 2,
        "graphic_design": 2,
    }
    assert len(review_a) == 5


def test_precedence_makes_strata_disjoint_and_records_all_matches() -> None:
    rows = []
    for style in ["illustration", "cartoon", "painting", "3d render", "poster"]:
        for index in range(2):
            prompt = f"a detailed {style} scene with a character {index}"
            rows.append((len(rows), _row(prompt)))
    rows[0] = (0, _row("a detailed illustration cartoon scene with a character zero"))

    selected, _, _ = select_artificial_prompts(rows, token_length=lambda _: 4, cfg=_cfg())
    mixed = next(record for record in selected if record["source_row_id"] == 0)
    assert mixed["style_stratum"] == "illustration_vector"
    assert mixed["style_matched_labels"] == ["illustration_vector", "cartoon_comic_anime"]
    assert len({record["source_row_id"] for record in selected}) == len(selected)


def test_filters_reject_duplicates_nsfw_photographic_and_long_prompts() -> None:
    rows = [
        (0, _row("a detailed illustration scene with trees")),
        (1, _row("A detailed illustration scene with trees!!!")),
        (2, _row("a detailed cartoon scene with hero", prompt_nsfw=None)),
        (3, _row("a detailed painting photograph scene with hero")),
        (4, _row("a detailed 3d render scene with hero")),
        (5, _row("a detailed poster scene with hero")),
    ]
    # Add enough valid rows for each stratum after filtering.
    rows.extend(
        [
            (6, _row("a second illustration scene with trees")),
            (7, _row("a second cartoon scene with hero")),
            (8, _row("a third cartoon scene with hero")),
            (9, _row("a second painting scene with hero")),
            (10, _row("a third painting scene with hero")),
            (11, _row("a second 3d render scene with hero")),
            (12, _row("a third 3d render scene with hero")),
            (13, _row("a second poster scene with hero")),
            (14, _row("a third poster scene with hero")),
        ]
    )
    selected, summary, _ = select_artificial_prompts(
        rows,
        token_length=lambda prompt: 76 if "third poster" in prompt else 4,
        cfg=_cfg(),
    )
    assert len(selected) == 10
    assert summary["rejection_counts"]["duplicate"] == 1
    assert summary["rejection_counts"]["nsfw_missing"] == 1
    assert summary["rejection_counts"]["photographic_target"] == 1
    assert summary["rejection_counts"]["too_many_clip_tokens"] == 1
