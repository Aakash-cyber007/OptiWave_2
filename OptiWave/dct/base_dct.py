import numpy as np
import cv2
from utils.image_loader import load_image
from utils.metrics import time_function  # Adjust import if needed


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to range [0, 1].
    """
    return np.float32(image) / 255.0


def apply_dct(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D Discrete Cosine Transform.
    """
    return cv2.dct(image)


def apply_idct(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply Inverse Discrete Cosine Transform.
    """
    return cv2.idct(dct_coeffs)


def reconstruct_image(normalized_image: np.ndarray) -> np.ndarray:
    """
    Convert back to uint8 image format.
    """
    return np.clip(normalized_image * 255.0, 0, 255).astype(np.uint8)


@time_function("DCT Pipeline")
def dct_pipeline() -> np.ndarray:
    """
    Full DCT pipeline using external image loader and metrics.
    Returns the reconstructed image.
    """
    image = load_image()  # Assumes grayscale image is returned

    normalized = normalize_image(image)
    dct_result = apply_dct(normalized)
    idct_result = apply_idct(dct_result)
    reconstructed = reconstruct_image(idct_result)

    return reconstructed
