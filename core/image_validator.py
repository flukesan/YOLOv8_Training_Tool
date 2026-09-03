"""
Image Validator - detect empty or damaged image files.

Used in two places:
- before copying images into a project, so broken files never enter the
  dataset in the first place;
- by the clean-up action, for files that are already there.

The default check is cheap (file size plus the format's start/end
markers) so it can run across tens of thousands of images. It catches the
common real-world case of a truncated download or interrupted copy: a
JPEG with no EOI marker, or a PNG with no IEND chunk. `deep=True` also
decodes the pixel data with Pillow, which catches damage in the middle of
an otherwise well-terminated file, at a much higher cost per image.
"""
from pathlib import Path
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

_JPEG_SOI = b'\xff\xd8'          # start of image
_JPEG_EOI = b'\xff\xd9'          # end of image
_PNG_SIG = b'\x89PNG\r\n\x1a\n'
_PNG_IEND = b'IEND'
_BMP_SIG = b'BM'
_TAIL_BYTES = 64                 # enough to find an end marker past padding
_MIN_PLAUSIBLE_SIZE = 100        # bytes; smaller cannot be a real image


def check_image_integrity(path: Path, deep: bool = False) -> Optional[str]:
    """
    Return a human-readable reason if `path` is not a usable image, or
    None if it looks fine.

    Args:
        path: image file to check
        deep: also fully decode the pixels with Pillow (slower, but
            catches interior corruption the markers miss)
    """
    path = Path(path)

    try:
        size = path.stat().st_size
    except OSError as e:
        return f"cannot read file ({e})"

    if size == 0:
        return "empty file (0 byte)"
    if size < _MIN_PLAUSIBLE_SIZE:
        return f"file too small to be an image ({size} bytes)"

    try:
        with open(path, 'rb') as f:
            head = f.read(8)
            f.seek(max(0, size - _TAIL_BYTES))
            tail = f.read()
    except OSError as e:
        return f"cannot read file ({e})"

    suffix = path.suffix.lower()
    if suffix in ('.jpg', '.jpeg'):
        if not head.startswith(_JPEG_SOI):
            return "not a valid JPEG (missing SOI marker)"
        if _JPEG_EOI not in tail:
            return "truncated JPEG (no EOI marker)"
    elif suffix == '.png':
        if not head.startswith(_PNG_SIG):
            return "not a valid PNG (bad signature)"
        if _PNG_IEND not in tail:
            return "truncated PNG (no IEND chunk)"
    elif suffix == '.bmp':
        if not head.startswith(_BMP_SIG):
            return "not a valid BMP (bad signature)"

    if deep:
        return _deep_check(path)
    return None


def _deep_check(path: Path) -> Optional[str]:
    """Fully decode the image with Pillow to catch interior damage."""
    try:
        from PIL import Image, ImageFile
    except ImportError:
        return None  # Pillow unavailable - skip the deep check

    # Ensure a truncated file raises instead of being silently padded out
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as img:
            img.load()
    except Exception as e:
        return f"cannot decode image ({type(e).__name__}: {e})"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    return None


def is_valid_image(path: Path, deep: bool = False) -> bool:
    """Convenience wrapper: True when the file is a usable image."""
    return check_image_integrity(path, deep=deep) is None
