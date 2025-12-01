"""
AIU Image Toolkit Backend - Image Processing and Cryptography Functions
Based on CSE281 Lecture Notes

This module contains all necessary functions for image processing and cryptography
techniques. All functions operate on NumPy arrays as inputs and outputs.
"""

import numpy as np
from PIL import Image
from skimage import io, color, exposure
import cv2


# ==============================================================================
# 2. Core Utilities & Metrics
# ==============================================================================

def to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    
    Academic Reference: rgb2gray (Lecture 5, 4)
    
    Args:
        rgb_image: RGB NumPy array of shape (H, W, 3)
        
    Returns:
        Grayscale NumPy array of shape (H, W) with dtype uint8
    """
    if len(rgb_image.shape) == 2:
        # Already grayscale
        return rgb_image.astype(np.uint8)
    
    if rgb_image.shape[2] == 4:
        # RGBA image, convert to RGB first
        rgb_image = rgb_image[:, :, :3]
    
    # Use standard luminosity method: 0.2989*R + 0.5870*G + 0.1140*B
    grayscale = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale.astype(np.uint8)


def calc_psnr(original: np.ndarray, decrypted: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Academic Reference: PSNR (Lecture 8)
    
    PSNR = 10 * log10(MAX^2 / MSE)
    where MAX is the maximum pixel value (255 for 8-bit images)
    
    Args:
        original: Original image as NumPy array
        decrypted: Decrypted/reconstructed image as NumPy array
        
    Returns:
        PSNR value in dB (decibels)
    """
    original = original.astype(np.float64)
    decrypted = decrypted.astype(np.float64)
    
    mse = np.mean((original - decrypted) ** 2)
    
    if mse == 0:
        return float('inf')  # Images are identical
    
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def calc_correlation(original: np.ndarray, decrypted: np.ndarray) -> float:
    """
    Calculate correlation coefficient between two images.
    
    Academic Reference: Correlation (Lecture 8)
    
    Args:
        original: Original image as NumPy array
        decrypted: Decrypted/reconstructed image as NumPy array
        
    Returns:
        Correlation coefficient value between -1 and 1
    """
    original = original.astype(np.float64).flatten()
    decrypted = decrypted.astype(np.float64).flatten()
    
    # Calculate Pearson correlation coefficient
    correlation = np.corrcoef(original, decrypted)[0, 1]
    
    return correlation


# ==============================================================================
# 3. Spatial Domain Enhancement Techniques
# ==============================================================================

def negative_transform(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply the image negative operation: S = L - 1 - r (i.e., 255 - r)
    
    Academic Reference: Image Negatives (Lecture 4)
    
    Args:
        gray_image: Grayscale image as NumPy array
        
    Returns:
        Negative image as NumPy array with dtype uint8
    """
    return (255 - gray_image).astype(np.uint8)


def log_transform(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply the logarithmic transformation: s = c * ln(1 + r)
    
    Uses the recommended scaling factor c = 255 / ln(1 + 255)
    
    Academic Reference: Logarithmic transformation (Lecture 5, 4)
    
    Args:
        gray_image: Grayscale image as NumPy array
        
    Returns:
        Log-transformed image as NumPy array with dtype uint8
    """
    # Scaling factor as per lecture notes
    c = 255.0 / np.log(1 + 255)
    
    # Apply log transformation
    gray_float = gray_image.astype(np.float64)
    result = c * np.log(1 + gray_float)
    
    # Clip and convert to uint8
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def gamma_transform(gray_image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply the power-law (gamma) transformation: s = c * r^gamma
    
    The image is normalized to [0, 1] before exponentiation and 
    scaled back to [0, 255] after.
    
    Academic Reference: Power-law (gamma) transformation (Lecture 5, 4)
    
    Args:
        gray_image: Grayscale image as NumPy array
        gamma: Gamma value for the transformation
        
    Returns:
        Gamma-transformed image as NumPy array with dtype uint8
    """
    # Normalize to [0, 1]
    normalized = gray_image.astype(np.float64) / 255.0
    
    # Apply gamma transformation (c = 1 for normalized images)
    result = np.power(normalized, gamma)
    
    # Scale back to [0, 255]
    result = result * 255.0
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def hist_equalization(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.
    
    Academic Reference: Histogram Equalization (Lecture 5)
    
    Args:
        gray_image: Grayscale image as NumPy array
        
    Returns:
        Histogram-equalized image as NumPy array with dtype uint8
    """
    # Use skimage's equalize_hist function
    # It expects input in range [0, 1] or handles uint8 automatically
    equalized = exposure.equalize_hist(gray_image)
    
    # Scale back to [0, 255] and convert to uint8
    result = (equalized * 255).astype(np.uint8)
    
    return result


def binary_segmentation(gray_image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert the image to a binary (black and white) image based on threshold.
    
    If pixel >= threshold * 255 then 255, else 0
    
    Academic Reference: Binary Segmentation (Lecture 4)
    
    Args:
        gray_image: Grayscale image as NumPy array
        threshold: Threshold value in range [0.0, 1.0]
        
    Returns:
        Binary image as NumPy array with dtype uint8 (values 0 or 255)
    """
    threshold_value = threshold * 255
    binary = np.where(gray_image >= threshold_value, 255, 0)
    return binary.astype(np.uint8)


# ==============================================================================
# 4. Spatial Noise Filters
# ==============================================================================

def add_salt_pepper(image_array: np.ndarray, prob: float) -> np.ndarray:
    """
    Add Salt & Pepper noise to an image.
    
    Sets random pixels to 0 (pepper) or 255 (salt) based on probability.
    
    Academic Reference: Salt & Pepper Noise (Lecture 6)
    
    Args:
        image_array: Input image as NumPy array
        prob: Noise probability (0.0 to 1.0)
        
    Returns:
        Noisy image as NumPy array with dtype uint8
    """
    output = image_array.copy().astype(np.uint8)
    
    # Generate random values for each pixel
    random_matrix = np.random.random(image_array.shape[:2])
    
    # Salt noise (white pixels)
    salt_mask = random_matrix < (prob / 2)
    # Pepper noise (black pixels)
    pepper_mask = (random_matrix >= (prob / 2)) & (random_matrix < prob)
    
    if len(output.shape) == 3:
        # Color image
        output[salt_mask] = [255, 255, 255] if output.shape[2] == 3 else 255
        output[pepper_mask] = [0, 0, 0] if output.shape[2] == 3 else 0
    else:
        # Grayscale image
        output[salt_mask] = 255
        output[pepper_mask] = 0
    
    return output


def apply_gaussian_filter(image_array: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing filter (Low-pass filter).
    
    Academic Reference: Gaussian Filter (Lecture 6)
    
    Args:
        image_array: Input image as NumPy array
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        Filtered image as NumPy array with dtype uint8
    """
    # Ensure kernel size is odd
    if size % 2 == 0:
        size += 1
    
    # Apply Gaussian blur using OpenCV
    result = cv2.GaussianBlur(image_array, (size, size), sigma)
    
    return result.astype(np.uint8)


def apply_median_filter(image_array: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Apply Median filter (Non-linear filter).
    
    Highly effective against Salt & Pepper noise.
    
    Academic Reference: Median Filter (Lecture 6)
    
    Args:
        image_array: Input image as NumPy array
        size: Kernel size (must be odd)
        
    Returns:
        Filtered image as NumPy array with dtype uint8
    """
    # Ensure kernel size is odd
    if size % 2 == 0:
        size += 1
    
    # Apply median filter using OpenCV
    result = cv2.medianBlur(image_array, size)
    
    return result.astype(np.uint8)


# ==============================================================================
# 5. Frequency Domain Filtering
# ==============================================================================

def get_frequency_spectrum(image_array: np.ndarray) -> np.ndarray:
    """
    Compute and return the magnitude spectrum after DFT and shifting.
    
    Uses np.fft.fft2 and np.fft.fftshift for visualization.
    
    Academic Reference: Discrete Fourier Transform (Lecture 8)
    
    Args:
        image_array: Grayscale image as NumPy array
        
    Returns:
        Magnitude spectrum as NumPy array with dtype uint8 (log-scaled for visualization)
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = to_grayscale(image_array)
    
    # Apply 2D DFT
    dft = np.fft.fft2(image_array.astype(np.float64))
    
    # Shift zero frequency to center
    dft_shifted = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum (log scale for visualization)
    magnitude = np.abs(dft_shifted)
    magnitude_log = np.log(1 + magnitude)
    
    # Normalize to 0-255 for display
    magnitude_normalized = (magnitude_log / magnitude_log.max() * 255).astype(np.uint8)
    
    return magnitude_normalized


def apply_ideal_lpf(image_array: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """
    Apply Ideal Low-Pass Filter (LPF) in the frequency domain.
    
    Removes high frequencies (smoothing/blurring effect).
    
    Academic Reference: Low-Pass Filter (Lecture 8)
    
    Args:
        image_array: Grayscale image as NumPy array
        cutoff_ratio: Cutoff frequency ratio (0.0 to 1.0, relative to image dimensions)
        
    Returns:
        Filtered image as NumPy array with dtype uint8
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = to_grayscale(image_array)
    
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    
    # Apply 2D DFT
    dft = np.fft.fft2(image_array.astype(np.float64))
    dft_shifted = np.fft.fftshift(dft)
    
    # Create ideal low-pass filter mask
    cutoff = int(min(rows, cols) * cutoff_ratio / 2)
    mask = np.zeros((rows, cols), np.float64)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1
    
    # Apply filter
    filtered_dft = dft_shifted * mask
    
    # Inverse DFT
    dft_ishift = np.fft.ifftshift(filtered_dft)
    result = np.fft.ifft2(dft_ishift)
    result = np.abs(result)
    
    # Normalize and convert to uint8
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def apply_ideal_hpf(image_array: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """
    Apply Ideal High-Pass Filter (HPF) in the frequency domain.
    
    Removes low frequencies (edge enhancement effect).
    
    Academic Reference: High-Pass Filter (Lecture 8)
    
    Args:
        image_array: Grayscale image as NumPy array
        cutoff_ratio: Cutoff frequency ratio (0.0 to 1.0, relative to image dimensions)
        
    Returns:
        Filtered image as NumPy array with dtype uint8
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = to_grayscale(image_array)
    
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    
    # Apply 2D DFT
    dft = np.fft.fft2(image_array.astype(np.float64))
    dft_shifted = np.fft.fftshift(dft)
    
    # Create ideal high-pass filter mask
    cutoff = int(min(rows, cols) * cutoff_ratio / 2)
    mask = np.ones((rows, cols), np.float64)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff:
                mask[i, j] = 0
    
    # Apply filter
    filtered_dft = dft_shifted * mask
    
    # Inverse DFT
    dft_ishift = np.fft.ifftshift(filtered_dft)
    result = np.fft.ifft2(dft_ishift)
    result = np.abs(result)
    
    # Normalize and convert to uint8
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def apply_notch_filter(image_array: np.ndarray, centers: list, radius: int) -> np.ndarray:
    """
    Apply Notch filter in the frequency domain.
    
    Eliminates specific periodic noise by zeroing out targeted frequency points.
    
    Academic Reference: Notch Filter (Lecture 8)
    
    Args:
        image_array: Grayscale image as NumPy array
        centers: List of tuples (u, v) specifying notch center positions
        radius: Radius of each notch
        
    Returns:
        Filtered image as NumPy array with dtype uint8
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = to_grayscale(image_array)
    
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    
    # Apply 2D DFT
    dft = np.fft.fft2(image_array.astype(np.float64))
    dft_shifted = np.fft.fftshift(dft)
    
    # Create notch filter mask
    mask = np.ones((rows, cols), np.float64)
    
    for (u, v) in centers:
        # Notch at (crow + u, ccol + v) and its conjugate at (crow - u, ccol - v)
        for i in range(rows):
            for j in range(cols):
                # Distance to notch center
                d1 = np.sqrt((i - (crow + u)) ** 2 + (j - (ccol + v)) ** 2)
                d2 = np.sqrt((i - (crow - u)) ** 2 + (j - (ccol - v)) ** 2)
                
                if d1 <= radius or d2 <= radius:
                    mask[i, j] = 0
    
    # Apply filter
    filtered_dft = dft_shifted * mask
    
    # Inverse DFT
    dft_ishift = np.fft.ifftshift(filtered_dft)
    result = np.fft.ifft2(dft_ishift)
    result = np.abs(result)
    
    # Normalize and convert to uint8
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


# ==============================================================================
# 6. Cryptography
# ==============================================================================

def xor_encrypt_decrypt(image_array: np.ndarray, key: int) -> np.ndarray:
    """
    Encrypt/decrypt an image using bitwise XOR operation.
    
    The same function is used for both encryption and decryption
    since XOR is its own inverse: (A XOR K) XOR K = A
    
    Academic Reference: XOR Encryption (Lecture 8)
    
    Args:
        image_array: Input image as NumPy array
        key: Secret key (integer value 0-255)
        
    Returns:
        Encrypted/decrypted image as NumPy array with dtype uint8
    """
    # Ensure key is within valid range
    key = key % 256
    
    # Apply XOR operation
    result = np.bitwise_xor(image_array.astype(np.uint8), key)
    
    return result.astype(np.uint8)


# ==============================================================================
# 7. Example Usage and Integration
# ==============================================================================

def example_attack_decryption_scenario():
    """
    Demonstrates the Attack-Decryption Scenario:
    1. Loading the original image
    2. Encrypting it
    3. Simulating an attack (adding Salt & Pepper noise) to the encrypted image
    4. Decrypting the "attacked" image
    5. Calculating and displaying PSNR and Correlation metrics
    """
    print("=" * 60)
    print("AIU Image Toolkit - Attack-Decryption Scenario Demo")
    print("=" * 60)
    
    # Create a sample grayscale image for demonstration
    # In real usage, load an actual image using: io.imread('image.jpg')
    print("\n1. Creating sample image (256x256 gradient)...")
    original = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    print(f"   Original image shape: {original.shape}")
    print(f"   Original image dtype: {original.dtype}")
    
    # Define encryption key
    secret_key = 123
    print(f"\n2. Encrypting image with key: {secret_key}")
    encrypted = xor_encrypt_decrypt(original, secret_key)
    print(f"   Encrypted image created.")
    
    # Simulate attack: Add Salt & Pepper noise
    noise_probability = 0.02
    print(f"\n3. Simulating attack: Adding Salt & Pepper noise (prob={noise_probability})")
    attacked_encrypted = add_salt_pepper(encrypted, noise_probability)
    print(f"   Attack simulation complete.")
    
    # Decrypt the attacked image
    print(f"\n4. Decrypting attacked image with key: {secret_key}")
    decrypted_attacked = xor_encrypt_decrypt(attacked_encrypted, secret_key)
    print(f"   Decryption complete.")
    
    # Calculate metrics
    print("\n5. Calculating quality metrics...")
    psnr_value = calc_psnr(original, decrypted_attacked)
    correlation_value = calc_correlation(original, decrypted_attacked)
    
    print(f"\n{'=' * 60}")
    print("RESULTS:")
    print(f"{'=' * 60}")
    print(f"   PSNR between original and attacked-then-decrypted: {psnr_value:.2f} dB")
    print(f"   Correlation between original and attacked-then-decrypted: {correlation_value:.6f}")
    print(f"{'=' * 60}")
    
    # Also demonstrate some image processing functions
    print("\n\nAdditional Demonstrations:")
    print("-" * 40)
    
    # Negative transform
    negative = negative_transform(original)
    print(f"   Negative transform applied. Sample pixel (0,0): {original[0,0]} -> {negative[0,0]}")
    
    # Log transform
    log_img = log_transform(original)
    print(f"   Log transform applied. Sample pixel (128,0): {original[0,128]} -> {log_img[0,128]}")
    
    # Gamma transform
    gamma_img = gamma_transform(original, gamma=2.2)
    print(f"   Gamma transform (gamma=2.2) applied.")
    
    # Histogram equalization
    hist_eq = hist_equalization(original)
    print(f"   Histogram equalization applied.")
    
    # Binary segmentation
    binary = binary_segmentation(original, threshold=0.5)
    print(f"   Binary segmentation (threshold=0.5) applied.")
    
    # Gaussian filter
    gaussian_filtered = apply_gaussian_filter(original, size=5, sigma=1.0)
    print(f"   Gaussian filter (5x5, sigma=1.0) applied.")
    
    # Median filter
    median_filtered = apply_median_filter(original, size=5)
    print(f"   Median filter (5x5) applied.")
    
    # Frequency spectrum
    spectrum = get_frequency_spectrum(original)
    print(f"   Frequency spectrum computed.")
    
    # Ideal LPF
    lpf_result = apply_ideal_lpf(original, cutoff_ratio=0.3)
    print(f"   Ideal LPF (cutoff_ratio=0.3) applied.")
    
    # Ideal HPF
    hpf_result = apply_ideal_hpf(original, cutoff_ratio=0.1)
    print(f"   Ideal HPF (cutoff_ratio=0.1) applied.")
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)
    
    return {
        'original': original,
        'encrypted': encrypted,
        'attacked_encrypted': attacked_encrypted,
        'decrypted_attacked': decrypted_attacked,
        'psnr': psnr_value,
        'correlation': correlation_value
    }


if __name__ == "__main__":
    example_attack_decryption_scenario()
