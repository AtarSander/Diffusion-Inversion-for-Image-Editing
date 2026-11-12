import json

from diff_inversion.data.prepare_pie_bench_ood_prompts import prepare_prompts


def test_prepare_pie_bench_prompts_is_sorted_and_preserves_provenance(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "b": {"original_prompt": "a [blue] bicycle", "editing_type_id": "2"},
                "a": {"original_prompt": "a red car", "editing_type_id": "0"},
                "c": {"original_prompt": "   "},
            }
        ),
        encoding="utf-8",
    )

    records, summary = prepare_prompts(mapping_path)

    assert [record["pie_bench_id"] for record in records] == ["a", "b"]
    assert records[1]["prompt"] == "a blue bicycle"
    assert records[1]["editing_type_id"] == "2"
    assert summary["rejected"] == {"empty_original_prompt": 1}
