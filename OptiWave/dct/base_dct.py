import numpy as np
import cv2

def read_image(path: str) -> np.ndarray:
    """
    Load a grayscale image.
    :param path: Path to the image file
    :return: Grayscale image as 2D numpy array
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to range [0, 1].
    :param image: Grayscale image
    :return: Normalized image
    """
    return np.float32(image) / 255.0

def apply_dct(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D Discrete Cosine Transform.
    :param image: Normalized grayscale image
    :return: DCT coefficients
    """
    return cv2.dct(image)

def apply_idct(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply inverse DCT to recover image.
    :param dct_coeffs: DCT coefficients
    :return: Reconstructed normalized image
    """
    return cv2.idct(dct_coeffs)

def reconstruct_image(reconstructed: np.ndarray) -> np.ndarray:
    """
    Convert reconstructed image back to 8-bit format.
    :param reconstructed: Image in [0,1] range
    :return: 8-bit grayscale image
    """
    return np.clip(reconstructed * 255.0, 0, 255).astype(np.uint8)

def dct_pipeline(image_path: str) -> np.ndarray:
    """
    Full DCT-based image transform and reconstruction pipeline.
    :param image_path: Path to grayscale image
    :return: Reconstructed 8-bit image
    """
    original = read_image(image_path)
    normalized = normalize_image(original)
    dct_result = apply_dct(normalized)
    idct_result = apply_idct(dct_result)
    
    return reconstruct_image(idct_result)
