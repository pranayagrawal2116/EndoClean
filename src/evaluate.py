import os
import argparse
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from dataset import get_dataloaders, TARGET_CLASSES
from train import build_model

def evaluate_model(model_path, data_dir, metadata_csv, seed, device):
    """Evaluates a single model checkpoint on the test set."""
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        metadata_path=metadata_csv,
        batch_size=32,
        seed=seed
    )
    
    model = build_model(len(TARGET_CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {Path(model_path).parent.name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    mac_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(TARGET_CLASSES)))
    
    metrics = {
        'accuracy': acc,
        'macro_precision': mac_p,
        'macro_recall': mac_r,
        'macro_f1': mac_f1
    }
    
    return metrics, cm, all_labels, all_preds

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    conditions = ['raw', 'pde_denoised', 'gaussian_denoised']
    seeds = [42, 43, 44] # Must match what was used in training
    
    all_results = []
    confusion_matrices = {cond: np.zeros((len(TARGET_CLASSES), len(TARGET_CLASSES))) for cond in conditions}
    
    out_dir = Path('results')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for condition in conditions:
        for seed in seeds:
            model_dir = Path(f'results/checkpoints/{condition}_seed{seed}')
            model_path = model_dir / 'best_model.pth'
            
            if not model_path.exists():
                print(f"Warning: {model_path} not found. Skipping.")
                continue
                
            data_dir = f'data/{condition}'
            
            metrics, cm, _, _ = evaluate_model(model_path, data_dir, args.metadata_csv, seed, device)
            
            metrics['condition'] = condition
            metrics['seed'] = seed
            all_results.append(metrics)
            
            confusion_matrices[condition] += cm
            
    if not all_results:
        print("No models found to evaluate.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Save individual results
    df.to_csv(out_dir / 'test_metrics_all_runs.csv', index=False)
    
    # Aggregate results (mean ± std)
    agg_df = df.groupby('condition').agg(['mean', 'std']).reset_index()
    agg_df.to_csv(out_dir / 'test_metrics_aggregated.csv', index=False)
    print("\nAggregated Results:")
    print(agg_df)
    
    # Plot Comparison Bar Chart
    conditions_present = df['condition'].unique()
    means_acc = df.groupby('condition')['accuracy'].mean()
    stds_acc = df.groupby('condition')['accuracy'].std()
    means_f1 = df.groupby('condition')['macro_f1'].mean()
    stds_f1 = df.groupby('condition')['macro_f1'].std()
    
    x = np.arange(len(conditions_present))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, means_acc, width, yerr=stds_acc, label='Accuracy', capsize=5)
    rects2 = ax.bar(x + width/2, means_f1, width, yerr=stds_f1, label='Macro F1', capsize=5)
    
    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison by Preprocessing Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions_present)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'performance_comparison.png', dpi=300)
    
    # Plot Confusion Matrices
    for condition in conditions_present:
        cm = confusion_matrices[condition]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize by row
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
        plt.title(f'Normalized Confusion Matrix: {condition} (Aggregated)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(out_dir / f'confusion_matrix_{condition}.png', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, default='../data/metadata.csv')
    args = parser.parse_args()
    main(args)
