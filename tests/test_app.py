from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import app


def make_image_file(size=(8, 8), image_format="PNG"):
    image_file = BytesIO()
    Image.new("RGB", size, color=(40, 120, 80)).save(image_file, image_format)
    image_file.seek(0)
    return image_file


def test_validate_uploaded_image_accepts_valid_images():
    image = app.validate_uploaded_image(make_image_file())

    assert image.mode == "RGB"
    assert image.size == (8, 8)
    assert image.format == "PNG"


def test_validate_uploaded_image_rejects_invalid_images():
    with pytest.raises(ValueError, match="valid supported image"):
        app.validate_uploaded_image(BytesIO(b"not an image"))


def test_validate_uploaded_image_rejects_large_images(monkeypatch):
    monkeypatch.setattr(app, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(ValueError, match="too large"):
        app.validate_uploaded_image(make_image_file(size=(4, 4)))


def test_image_metadata_html_escapes_dynamic_values():
    html = app.image_metadata_html(
        SimpleNamespace(format="<script>alert(1)</script>", mode='RGB"><x', size=(3, 5))
    )

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "RGB&quot;&gt;&lt;x" in html


def test_summarize_detections_filters_by_class_and_threshold():
    boxes = [
        SimpleNamespace(cls=[0], conf=[0.9]),
        SimpleNamespace(cls=[0], conf=[0.2]),
        SimpleNamespace(cls=[1], conf=[0.95]),
        SimpleNamespace(cls=[0], conf=[0.5]),
    ]

    count, confidences = app.summarize_detections(boxes, confidence_threshold=0.5)

    assert count == 2
    assert confidences == [0.9, 0.5]


def test_confidence_summary_handles_empty_and_non_empty_values():
    empty = app.confidence_summary([])
    filled = app.confidence_summary([0.5, 0.75, 1.0])

    assert empty["avg_conf"] == 0
    assert filled["avg_conf"] == pytest.approx(0.75)
    assert filled["min_conf_pct"] == pytest.approx(50.0)
    assert filled["max_conf_pct"] == pytest.approx(100.0)


def test_run_model_inference_cleans_up_temp_file(monkeypatch):
    writes = []
    removed = []

    class FakeCV2:
        def imwrite(self, path, image):
            writes.append((path, image))
            return True

    class FakeModel:
        def __call__(self, path, conf, save):
            assert path == writes[0][0]
            assert conf == 0.7
            assert save is False
            return ["result"]

    monkeypatch.setattr(app, "_cv2", lambda: FakeCV2())
    monkeypatch.setattr(app, "load_model", lambda: FakeModel())
    monkeypatch.setattr(app.os, "remove", lambda path: removed.append(path))

    assert app.run_model_inference(object(), 0.7) == "result"
    assert removed == [writes[0][0]]

