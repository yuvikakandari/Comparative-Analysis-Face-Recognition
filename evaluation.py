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

def load_benchmark_data(model_name):
    print(f"\n[INFO] Extracting embeddings for {model_name}...")
    data = {}
    if not os.path.exists(lfw_path):
        print(f"❌ Error: {lfw_path} folder not found!")
        return None

    person_list = os.listdir(lfw_path)[:25]
    for person in person_list:
        person_path = os.path.join(lfw_path, person)
        if not os.path.isdir(person_path): continue
        
        embeddings = []
        for img_name in os.listdir(person_path)[:5]:
            img = cv2.imread(os.path.join(person_path, img_name))
            if img is None: continue
            emb = get_embedding(img, model_name)
            if emb is not None: embeddings.append(emb)

        if len(embeddings) >= 2:
            data[person] = embeddings
    return data

def compute_research_metrics(genuine_distances, impostor_distances, model_name):
    thresholds = np.linspace(0, 1.2, 100)
    far_list, frr_list = [], []
    best_acc, optimal_threshold = 0, 0
    
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
            optimal_threshold = t

    far_list, frr_list = np.array(far_list), np.array(frr_list)
    eer = far_list[np.nanargmin(np.absolute(far_list - frr_list))]

    plt.figure()
    plt.plot(far_list, 1 - frr_list, label=f'{model_name} (EER: {eer:.3f})')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(); plt.grid(True)
    plt.savefig(f"roc_{model_name}.png")
    plt.close()
    
    return eer, best_acc, optimal_threshold

def run_comparison():
    results_summary = []

    for model in models:
        # 1. Measure Extraction Speed
        start_time = time.time()
        data = load_benchmark_data(model)
        end_time = time.time()
        
        if not data:
            print(f"Skipping {model}: No data found.")
            continue

        # Correct placement of variable initialization
        persons = list(data.keys())
        genuine_distances = []
        impostor_distances = []
        
        avg_time_per_face = (end_time - start_time) / (len(persons) * 5)

        # 2. Measure Encryption Speed
        start_enc = time.time()
        for person, embs in data.items():
            raw_data = np.array(embs).tobytes()
            save_encrypted(f"{person}_{model}.enc", raw_data)
        end_enc = time.time()

        # 3. Create Pairs
        for person in persons:
            embs = data[person]
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    genuine_distances.append(calculate_distance(embs[i], embs[j]))

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                impostor_distances.append(calculate_distance(data[persons[i]][0], data[persons[j]][0]))

        # 4. Compute Metrics
        eer, acc, opt_t = compute_research_metrics(genuine_distances, impostor_distances, model)
        
        results_summary.append({
            "Model": model,
            "Accuracy": acc,
            "EER": eer,
            "Inference_Time": avg_time_per_face,
            "Enc_Time": (end_enc - start_enc) / len(data)
        })

    # Final Research Table
    print("\n" + "="*60)
    print(f"{'Model':<12} | {'Acc':<6} | {'EER':<6} | {'Inf. Speed':<12} | {'Enc. Speed'}")
    print("-" * 60)
    for r in results_summary:
        print(f"{r['Model']:<12} | {r['Accuracy']:.3f} | {r['EER']:.3f} | {r['Inference_Time']*1000:>7.1f} ms | {r['Enc_Time']*1000:>6.2f} ms")
    print("="*60)

if __name__ == "__main__":
    run_comparison()