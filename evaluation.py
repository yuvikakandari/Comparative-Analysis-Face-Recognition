import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from recognition_deepface import get_embedding, calculate_distance
from secure_storage import save_encrypted

# Configuration
models = ["Facenet", "VGG-Face", "ArcFace"]
lfw_path = "lfw-deepfunneled"
stress_modes = [None, "low_light", "blur"]

def apply_stress(img, mode=None):
    if mode == "low_light":
        # Luminance Attenuation: Reduce brightness by 60% [cite: 115]
        return cv2.convertScaleAbs(img, alpha=0.4, beta=0)
    elif mode == "blur":
        # Gaussian Perturbation: 15x15 Kernel to simulate motion blur [cite: 116]
        return cv2.GaussianBlur(img, (15, 15), 0)
    return img

import random

def load_benchmark_data(model_name, stress=None):
    stress_label = stress if stress else "Baseline"
    print(f"\n[INFO] Auditing {model_name} manifold stability ({stress_label})...")
    data = {}
    
    if not os.path.exists(lfw_path):
        print(f"❌ Error: {lfw_path} folder not found!")
        return None

    # Get all identities from the folder
    all_persons = [p for p in os.listdir(lfw_path) if os.path.isdir(os.path.join(lfw_path, p))]
    
    # REPRESENTATIVE SAMPLING: Select 300 random identities for statistical power
    # We use a seed so results remain consistent for paper
    random.seed(42) 
    if len(all_persons) > 300:
        person_list = random.sample(all_persons, 300)
    else:
        person_list = all_persons

    print(f"[PROCESS] Processing {len(person_list)} identities...")

    for person in person_list:
        person_path = os.path.join(lfw_path, person)
        embeddings = []
        
        # Take up to 5 images per person to balance the dataset
        img_files = os.listdir(person_path)[:5]
        for img_name in img_files:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Apply environmental stressors (Luminance/Gaussian)
            img = apply_stress(img, mode=stress)
            
            # Extract high-dimensional embedding
            emb = get_embedding(img, model_name)
            if emb is not None: 
                embeddings.append(emb)

        # Only include identities with enough samples for genuine pair testing
        if len(embeddings) >= 2:
            data[person] = embeddings
            
    return data

def compute_research_metrics(genuine_distances, impostor_distances, model_name, stress=None):
    thresholds = np.linspace(0, 1.2, 100)
    far_list, frr_list = [], []
    best_acc = 0
    
    gen_dist = np.array(genuine_distances)
    imp_dist = np.array(impostor_distances)
    
    for t in thresholds:
        far = np.mean(imp_dist < t)
        frr = np.mean(gen_dist > t)
        far_list.append(far)
        frr_list.append(frr)

        acc = (np.sum(gen_dist < t) + np.sum(imp_dist > t)) / (len(gen_dist) + len(imp_dist))
        if acc > best_acc:
            best_acc = acc

    far_list, frr_list = np.array(far_list), np.array(frr_list)
    eer = far_list[np.nanargmin(np.absolute(far_list - frr_list))]

    # Only save ROC plots for baseline to avoid clutter
    if stress is None:
        plt.figure()
        plt.plot(far_list, 1 - frr_list, label=f'{model_name} (EER: {eer:.3f})')
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(); plt.grid(True)
        plt.savefig(f"roc_{model_name}.png")
        plt.close()
    
    return eer, best_acc

def run_comparison():
    robustness_results = {m: {} for m in models}
    performance_summary = []

    for model in models:
        for mode in stress_modes:
            start_time = time.time()
            data = load_benchmark_data(model, stress=mode)
            end_time = time.time()
            
            if not data: continue

            persons = list(data.keys())
            genuine_distances = []
            impostor_distances = []

            # Measure Encryption Speed (only for baseline to save time)
            enc_speed = 0
            if mode is None:
                start_enc = time.time()
                for person, embs in data.items():
                    raw_data = np.array(embs).tobytes()
                    save_encrypted(f"{person}_{model}.enc", raw_data)
                enc_speed = (time.time() - start_enc) / len(data)

            # Pairwise distance calculation
            for person in persons:
                embs = data[person]
                for i in range(len(embs)):
                    for j in range(i + 1, len(embs)):
                        genuine_distances.append(calculate_distance(embs[i], embs[j]))

            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    impostor_distances.append(calculate_distance(data[persons[i]][0], data[persons[j]][0]))

            eer, acc = compute_research_metrics(genuine_distances, impostor_distances, model, stress=mode)
            
            # Store for Table I
            mode_key = mode if mode else "baseline"
            robustness_results[model][mode_key] = eer

            # Store for Table II (Baseline performance)
            if mode is None:
                performance_summary.append({
                    "Model": model,
                    "Accuracy": acc,
                    "EER": eer,
                    "Inf_Time": (end_time - start_time) / (len(persons) * 5),
                    "Enc_Time": enc_speed
                })

    # Output Table I: Environmental Robustness [cite: 121, 122]
    print("\n" + "="*50)
    print(f"{'TABLE I: ROBUSTNESS (EER)':<15} | {'Base':<6} | {'Dark':<6} | {'Blur'}")
    print("-" * 50)
    for m in models:
        res = robustness_results[m]
        print(f"{m:<15} | {res['baseline']:.3f} | {res['low_light']:.3f} | {res['blur']:.3f}")
    
    # Output Table II: Technical Performance [cite: 124, 126]
    print("\n" + "="*65)
    print(f"{'TABLE II: BASELINE':<12} | {'Acc':<6} | {'EER':<6} | {'Inf. Speed':<12} | {'Enc. Speed'}")
    print("-" * 65)
    for r in performance_summary:
        print(f"{r['Model']:<12} | {r['Accuracy']:.3f} | {r['EER']:.3f} | {r['Inf_Time']*1000:>7.1f} ms | {r['Enc_Time']*1000:>6.2f} ms")
    print("="*65)

if __name__ == "__main__":
    run_comparison()