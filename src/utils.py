import cv2
import numpy as np


def fun_remove_bg(image):
    # Create a mask by thresholding (background is dark)
    _, mask = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    # Clean up mask with morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    # Apply mask to keep only foreground
    bg_removed = cv2.bitwise_and(image, image, mask=mask)
    return bg_removed, mask


# (width, height) every notebook resizes to, so N1/N2/N3 see identical pixels
PREPROCESS_SIZE = (150, 200)


def fun_preprocess(image, size=PREPROCESS_SIZE):
    """
    Shared preprocessing used by N1, N2 and N3:
    resize -> grayscale -> background removal (fun_remove_bg).

    Returns (gray, bg_removed, mask). Pass size=None to keep the original size.
    """
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        gray = image
    bg_removed, mask = fun_remove_bg(gray)
    return gray, bg_removed, mask


def fun_load_preprocessed(path, size=PREPROCESS_SIZE):
    """Read an image from disk and apply fun_preprocess. Returns (gray, bg_removed, mask)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return fun_preprocess(image, size)