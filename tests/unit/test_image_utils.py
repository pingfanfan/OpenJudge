import base64
from pathlib import Path

from PIL import Image

from prism.utils.image import image_to_data_url


def test_from_path():
    fixture = Path(__file__).parent.parent / "fixtures" / "images" / "pixel_red.png"
    url = image_to_data_url(str(fixture))
    assert url.startswith("data:image/png;base64,")
    b64 = url.split(",", 1)[1]
    data = base64.b64decode(b64)
    assert data.startswith(b"\x89PNG\r\n")


def test_from_pil_image():
    img = Image.new("RGB", (2, 2), color=(0, 255, 0))
    url = image_to_data_url(img)
    assert url.startswith("data:image/png;base64,")
    b64 = url.split(",", 1)[1]
    data = base64.b64decode(b64)
    assert data.startswith(b"\x89PNG\r\n")


def test_from_pathlib_path():
    fixture = Path(__file__).parent.parent / "fixtures" / "images" / "pixel_blue.png"
    url = image_to_data_url(fixture)
    assert url.startswith("data:image/png;base64,")


def test_rejects_unsupported_type():
    import pytest
    with pytest.raises(TypeError, match="unsupported"):
        image_to_data_url(12345)
