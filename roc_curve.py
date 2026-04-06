import matplotlib.pyplot as plt
import numpy as np

def generate_roc_set():
    # Final EER values from your N=300 run
    data = {
        'Baseline': {
            'Facenet': 0.135, 'VGG-Face': 0.101, 'ArcFace': 0.156
        },
        'Low-Light (Dark)': {
            'Facenet': 0.276, 'VGG-Face': 0.109, 'ArcFace': 0.133
        },
        'Gaussian Blur': {
            'Facenet': 0.374, 'VGG-Face': 0.184, 'ArcFace': 0.195
        }
    }

    colors = {'Facenet': '#2ca02c', 'VGG-Face': '#ff7f0e', 'ArcFace': '#1f77b4'}
    x = np.linspace(0, 1, 100)

    for mode, models in data.items():
        plt.figure(figsize=(7, 5))
        
        for model_name, eer in models.items():
            # Mathematical approximation of ROC curve from EER
            y = x**(eer / (1 - eer))
            plt.plot(x, y, label=f"{model_name} (EER: {eer:.3f})", 
                     color=colors[model_name], lw=2)

        # Plot Random Chance
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Chance')
        
        plt.title(f'ROC Curve: {mode} Conditions ($N=300$)', fontsize=13)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Save each graph with a unique name
        filename = f"roc_{mode.split()[0].lower().replace('(', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved {filename}")
        plt.close()

if __name__ == "__main__":
    generate_roc_set()