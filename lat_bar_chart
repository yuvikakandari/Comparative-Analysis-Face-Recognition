import matplotlib.pyplot as plt
import numpy as np

def plot_latency_comparison():
    models = ['Facenet', 'VGG-Face', 'ArcFace']
    inference_times = [212.1, 172.0, 140.1]
    encryption_times = [0.64, 1.13, 0.60]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, inference_times, width, label='Inference (AI)', color='#3498db')
    rects2 = ax.bar(x + width/2, encryption_times, width, label='Encryption (AES-256)', color='#e74c3c')

    ax.set_ylabel('Latency (ms)')
    ax.set_title('Computational Throughput: Inference vs. Cryptographic Overhead')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('latency_comparison.png', dpi=300)
    plt.show()

plot_latency_comparison()