import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def perona_malik_diffusion(image: np.ndarray, num_iter: int = 15, kappa: float = 30.0, 
                           gamma: float = 0.15, option: int = 1) -> np.ndarray:
    """
    Perona-Malik Anisotropic Diffusion for image denoising.
    
    Args:
        image: Input image as numpy array (grayscale or color, usually float in [0, 255]).
        num_iter: Number of iterations.
        kappa: Conduction coefficient threshold.
        gamma: Integration constant (<= 0.25 for stability).
        option: 1 for exponential conduction, 2 for quadratic conduction.
        
    Returns:
        Denoised image.
    """
    # If image is color, process each channel independently or convert to float
    is_color = len(image.shape) == 3
    if is_color:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            result[:, :, c] = _perona_malik_channel(image[:, :, c], num_iter, kappa, gamma, option)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = _perona_malik_channel(image, num_iter, kappa, gamma, option)
        return np.clip(result, 0, 255).astype(np.uint8)

def _perona_malik_channel(image: np.ndarray, num_iter: int, kappa: float, gamma: float, option: int) -> np.ndarray:
    """
    Apply Perona-Malik on a single 2D channel.
    """
    img = image.astype(np.float32)
    
    for _ in range(num_iter):
        # Calculate gradients (differences) in 4 directions
        # N, S, E, W
        delta_N = np.zeros_like(img)
        delta_S = np.zeros_like(img)
        delta_E = np.zeros_like(img)
        delta_W = np.zeros_like(img)
        
        delta_N[:-1, :] = np.diff(img, axis=0) # img[1:] - img[:-1] => S instead of N if index 0 is top
        delta_S[1:, :] = -np.diff(img, axis=0) # img[:-1] - img[1:]
        delta_E[:, :-1] = np.diff(img, axis=1) # img[:, 1:] - img[:, :-1]
        delta_W[:, 1:] = -np.diff(img, axis=1) # img[:, :-1] - img[:, 1:]

        # Conduction coefficients
        if option == 1:
            cN = np.exp(-(delta_N / kappa)**2)
            cS = np.exp(-(delta_S / kappa)**2)
            cE = np.exp(-(delta_E / kappa)**2)
            cW = np.exp(-(delta_W / kappa)**2)
        elif option == 2:
            cN = 1.0 / (1.0 + (delta_N / kappa)**2)
            cS = 1.0 / (1.0 + (delta_S / kappa)**2)
            cE = 1.0 / (1.0 + (delta_E / kappa)**2)
            cW = 1.0 / (1.0 + (delta_W / kappa)**2)
        else:
            raise ValueError("option must be 1 or 2")

        img += gamma * (cN * delta_N + cS * delta_S + cE * delta_E + cW * delta_W)
        
    return img

def gaussian_denoise(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Baseline Gaussian blur denoising using OpenCV.
    
    Args:
        image: Input image (uint8 or float).
        ksize: Gaussian kernel size (must be odd).
        sigma: Gaussian standard deviation.
        
    Returns:
        Gaussian blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1 # Ensure odd kernel size
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def process_folder(input_dir: str, output_dir: str, method: str = 'pde', **kwargs):
    """
    Batch-process a folder of images and save denoised outputs to disk.
    Preserves the subfolder structure (e.g. classes).
    
    Args:
        input_dir: Root directory of raw images.
        output_dir: Root directory to save denoised images.
        method: 'pde' or 'gaussian'.
        **kwargs: Arguments for the specific denoising function.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.rglob(ext))
        
    print(f"Found {len(image_paths)} images to process using {method} denoising.")
    
    for img_p in tqdm(image_paths):
        rel_path = img_p.relative_to(input_path)
        out_p = output_path / rel_path
        
        # Ensure target subfolder exists
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        if method == 'pde':
            denoised = perona_malik_diffusion(img, **kwargs)
        elif method == 'gaussian':
            denoised = gaussian_denoise(img, **kwargs)
        else:
            raise ValueError(f"Unknown method {method}")
            
        cv2.imwrite(str(out_p), denoised)

if __name__ == "__main__":
    # Example usage placeholder
    pass
