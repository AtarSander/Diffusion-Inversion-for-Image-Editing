import pytest

from diff_inversion.eval.prepare_pie_bench_source_manifest import create_template, freeze_manifest


def test_template_and_frozen_manifest_cover_each_mapping_image_once() -> None:
    mapping = {
        "a": {"image_path": "one/image.jpg"},
        "b": {"image_path": "one/image.jpg"},
        "c": {"image_path": "two/image.jpg"},
    }
    assert create_template(mapping) == [
        {"image_path": "one/image.jpg", "label": "pending"},
        {"image_path": "two/image.jpg", "label": "pending"},
    ]
    frozen = freeze_manifest([
        {"image_path": "one/image.jpg", "label": "photographic"},
        {"image_path": "two/image.jpg", "label": "artificial_graphic"},
    ], {"one/image.jpg", "two/image.jpg"})
    assert frozen[0]["label"] == "photographic"


def test_freeze_rejects_pending_or_incomplete_labels() -> None:
    with pytest.raises(ValueError, match="Unsupported label"):
        freeze_manifest([{"image_path": "one.jpg", "label": "pending"}], {"one.jpg"})
    with pytest.raises(ValueError, match="does not match"):
        freeze_manifest([{"image_path": "one.jpg", "label": "uncertain"}], {"one.jpg", "two.jpg"})
