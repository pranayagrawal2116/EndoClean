import os
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_dataloaders, TARGET_CLASSES, get_transforms
from train import build_model

def get_gradcam(model, image_tensor, target_class_idx):
    # ResNet50 last conv layer is layer4[-1]
    target_layers = [model.layer4[-1]]
    
    # Construct the CAM object once, and then re-use it on many images
    cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    # In this example grayscale_cam has only one image in the batch
    return grayscale_cam[0, :]

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    out_dir = Path('results/gradcam')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset to get some images
    # We'll just take the raw test set
    _, _, test_loader, _ = get_dataloaders(
        data_dir='data/raw',
        metadata_path=args.metadata_csv,
        batch_size=1, # One by one
        seed=42
    )
    
    # Load Models
    raw_model_path = 'results/checkpoints/raw_seed42/best_model.pth'
    pde_model_path = 'results/checkpoints/pde_denoised_seed42/best_model.pth'
    
    if not os.path.exists(raw_model_path) or not os.path.exists(pde_model_path):
        print("Models not found. Train them first.")
        return
        
    raw_model = build_model(len(TARGET_CLASSES)).to(device)
    raw_model.load_state_dict(torch.load(raw_model_path, map_location=device))
    raw_model.eval()
    
    pde_model = build_model(len(TARGET_CLASSES)).to(device)
    pde_model.load_state_dict(torch.load(pde_model_path, map_location=device))
    pde_model.eval()
    
    # Get one image per class
    images_found = {cls: False for cls in TARGET_CLASSES}
    
    print("Generating Grad-CAM visualizations...")
    for images, labels in test_loader:
        class_idx = labels[0].item()
        class_name = TARGET_CLASSES[class_idx]
        
        if images_found[class_name]:
            continue
            
        images_found[class_name] = True
        
        input_tensor = images.to(device)
        
        # Get raw image for visualization
        # Invert normalization to display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        unnorm_img = input_tensor[0] * std + mean
        rgb_img = unnorm_img.permute(1, 2, 0).cpu().numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # GradCAM for Raw Model
        raw_cam_map = get_gradcam(raw_model, input_tensor, class_idx)
        raw_cam_img = show_cam_on_image(rgb_img, raw_cam_map, use_rgb=True)
        
        # GradCAM for PDE Model
        # (Technically we should feed the PDE-denoised version of the image to the PDE model,
        # but for direct comparison of focus on the SAME structure, feeding the same image 
        # or the denoised counterpart is interesting. Let's do denoised for PDE model if we can,
        # but since we loaded from 'raw' dir, we only have raw here. 
        # For simplicity and direct head-to-head on the same input pixels, we pass the raw image,
        # or we should denoise it first. Let's denoise it inline.)
        from denoising import perona_malik_diffusion
        raw_np = (rgb_img * 255).astype(np.uint8)
        pde_np = perona_malik_diffusion(raw_np, num_iter=15, kappa=30, option=1)
        
        # Need to transform PDE image for model
        transform = get_transforms(is_train=False)
        pde_tensor = transform(image=pde_np)['image'].unsqueeze(0).to(device)
        
        pde_cam_map = get_gradcam(pde_model, pde_tensor, class_idx)
        
        pde_rgb_img = pde_np.astype(np.float32) / 255.0
        pde_cam_img = show_cam_on_image(pde_rgb_img, pde_cam_map, use_rgb=True)
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original Raw ({class_name})")
        axes[0].axis('off')
        
        axes[1].imshow(raw_cam_img)
        axes[1].set_title("Grad-CAM (Raw Model)")
        axes[1].axis('off')
        
        axes[2].imshow(pde_rgb_img)
        axes[2].set_title(f"PDE Denoised")
        axes[2].axis('off')
        
        axes[3].imshow(pde_cam_img)
        axes[3].set_title("Grad-CAM (PDE Model)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(out_dir / f'{class_name}_gradcam.png', dpi=300)
        plt.close()
        
        if all(images_found.values()):
            break
            
    print(f"Saved Grad-CAM comparisons to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, default='../data/metadata.csv')
    args = parser.parse_args()
    main(args)
