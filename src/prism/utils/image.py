"""Image encoding utilities for multimodal benchmarks.

Converts either a PIL.Image.Image (as returned by the HuggingFace datasets
library for image fields) or a local filesystem path into a base64 data URL
suitable for OpenAI-style multimodal `image_url` content parts.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any


def image_to_data_url(img: Any) -> str:
    """Return `data:image/png;base64,...` for a PIL image or a local path.

    Always re-encodes to PNG for consistency. Callers that need format
    preservation can encode their own data URL upstream.
    """
    if isinstance(img, (str, Path)):
        with open(img, "rb") as f:
            data = f.read()
        mime = "image/png"
    else:
        if not hasattr(img, "save"):
            raise TypeError(
                f"unsupported image type: {type(img).__name__} — "
                f"expected str, Path, or PIL.Image.Image"
            )
        buf = io.BytesIO()
        if getattr(img, "mode", "RGB") not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(buf, format="PNG")
        data = buf.getvalue()
        mime = "image/png"

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
