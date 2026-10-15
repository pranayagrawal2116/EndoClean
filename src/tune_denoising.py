import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
from itertools import product
from pathlib import Path
import argparse
from tqdm import tqdm

from denoising import perona_malik_diffusion, gaussian_denoise

def add_synthetic_noise(image: np.ndarray) -> np.ndarray:
    """Adds Gaussian and Salt-and-Pepper noise to an image."""
    # skimage random_noise converts to [0, 1] float
    noisy = random_noise(image, mode='gaussian', var=0.01)
    noisy = random_noise(noisy, mode='s&p', amount=0.02)
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)

def evaluate_pde(clean_img: np.ndarray, noisy_img: np.ndarray, kappas: list, iters: list):
    """Grid search for PDE parameters."""
    results = []
    best_psnr = -1
    best_params = {}
    best_img = None
    
    print("Tuning PDE parameters...")
    for k, i in tqdm(list(product(kappas, iters))):
        denoised = perona_malik_diffusion(noisy_img, num_iter=i, kappa=k, option=1)
        
        # Calculate metrics
        p = psnr(clean_img, denoised)
        s = ssim(clean_img, denoised, channel_axis=2 if len(clean_img.shape) == 3 else None)
        
        results.append({'Method': 'PDE', 'Kappa': k, 'Iterations': i, 'PSNR': p, 'SSIM': s})
        
        if p > best_psnr:
            best_psnr = p
            best_params = {'Kappa': k, 'Iterations': i}
            best_img = denoised
            
    return pd.DataFrame(results), best_params, best_img

def evaluate_gaussian(clean_img: np.ndarray, noisy_img: np.ndarray, sigmas: list, ksizes: list):
    """Grid search for Gaussian blur parameters."""
    results = []
    best_psnr = -1
    best_params = {}
    best_img = None
    
    print("Tuning Gaussian parameters...")
    for s, ks in tqdm(list(product(sigmas, ksizes))):
        denoised = gaussian_denoise(noisy_img, ksize=ks, sigma=s)
        
        # Calculate metrics
        p = psnr(clean_img, denoised)
        ss = ssim(clean_img, denoised, channel_axis=2 if len(clean_img.shape) == 3 else None)
        
        results.append({'Method': 'Gaussian', 'Sigma': s, 'KernelSize': ks, 'PSNR': p, 'SSIM': ss})
        
        if p > best_psnr:
            best_psnr = p
            best_params = {'Sigma': s, 'KernelSize': ks}
            best_img = denoised
            
    return pd.DataFrame(results), best_params, best_img

def main(image_path: str, output_dir: str):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    if not image_path or not os.path.exists(image_path):
        print("No valid image provided. Using synthetic test image (e.g. from skimage).")
        from skimage import data
        clean_img = data.astronaut()
    else:
        clean_img = cv2.imread(image_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    # Add noise
    noisy_img = add_synthetic_noise(clean_img)
    
    # Grid search spaces
    kappas = [10, 20, 30, 40, 50]
    iters = [5, 10, 15, 20, 30]
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
    ksizes = [3, 5, 7]
    
    # Run evaluation
    pde_df, best_pde_params, best_pde_img = evaluate_pde(clean_img, noisy_img, kappas, iters)
    gauss_df, best_gauss_params, best_gauss_img = evaluate_gaussian(clean_img, noisy_img, sigmas, ksizes)
    
    # Save results table
    combined_df = pd.concat([pde_df, gauss_df], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, 'tuning_results.csv'), index=False)
    
    print("\nBest PDE params:", best_pde_params)
    print("Best Gaussian params:", best_gauss_params)
    
    # Plotting comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean_img)
    axes[0].set_title('Original Clean')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_img)
    axes[1].set_title(f'Noisy\nPSNR: {psnr(clean_img, noisy_img):.2f} / SSIM: {ssim(clean_img, noisy_img, channel_axis=2 if len(clean_img.shape) == 3 else None):.4f}')
    axes[1].axis('off')
    
    pde_psnr = psnr(clean_img, best_pde_img)
    pde_ssim = ssim(clean_img, best_pde_img, channel_axis=2 if len(clean_img.shape) == 3 else None)
    axes[2].imshow(best_pde_img)
    axes[2].set_title(f"PDE (k={best_pde_params['Kappa']}, i={best_pde_params['Iterations']})\nPSNR: {pde_psnr:.2f} / SSIM: {pde_ssim:.4f}")
    axes[2].axis('off')
    
    gauss_psnr = psnr(clean_img, best_gauss_img)
    gauss_ssim = ssim(clean_img, best_gauss_img, channel_axis=2 if len(clean_img.shape) == 3 else None)
    axes[3].imshow(best_gauss_img)
    axes[3].set_title(f"Gaussian (s={best_gauss_params['Sigma']}, ks={best_gauss_params['KernelSize']})\nPSNR: {gauss_psnr:.2f} / SSIM: {gauss_ssim:.4f}")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'denoising_comparison.png'), dpi=300)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='', help='Path to a clean sample image.')
    parser.add_argument('--output', type=str, default='../results', help='Output directory for tuning results.')
    args = parser.parse_args()
    
    # Assume script is run from src directory, resolving output directory properly
    out_dir = Path(__file__).parent.parent / 'results' if args.output == '../results' else Path(args.output)
    
    main(args.image, str(out_dir))
