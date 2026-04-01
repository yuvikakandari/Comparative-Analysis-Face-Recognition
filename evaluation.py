import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from recognition_deepface import get_embedding, calculate_distance

# Configuration
models = ["Facenet", "VGG-Face", "ArcFace"]
lfw_path = "lfw-deepfunneled"

def load_benchmark_data(model_name):
    print(f"\n[INFO] Extracting embeddings for {model_name}...")
    data = {}
    
    if not os.path.exists(lfw_path):
        print(f"❌ Error: {lfw_path} folder not found!")
        return None

    # Process first 25 people for efficiency
    person_list = os.listdir(lfw_path)[:25]
    
    for person in person_list:
        person_path = os.path.join(lfw_path, person)
        if not os.path.isdir(person_path):
            continue
        
        embeddings = []
        # Limit to 5 images per person
        for img_name in os.listdir(person_path)[:5]:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            emb = get_embedding(img, model_name)
            if emb is not None:
                embeddings.append(emb)

        # We need at least 2 images to create a 'Genuine' pair
        if len(embeddings) >= 2:
            data[person] = embeddings
            
    return data

def compute_research_metrics(genuine_distances, impostor_distances, model_name):
    thresholds = np.linspace(0, 1.2, 100)
    far_list = []
    frr_list = []
    best_acc = 0
    optimal_threshold = 0
    
    print(f"--- {model_name} Threshold Sweep ---")
    print("Threshold | FAR     | FRR     | Accuracy")
    print("-" * 40)
    
    # Pre-convert to numpy arrays for speed
    gen_dist = np.array(genuine_distances)
    imp_dist = np.array(impostor_distances)
    
    for t in thresholds:
        far = np.mean(imp_dist < t)
        frr = np.mean(gen_dist > t)
        
        far_list.append(far)
        frr_list.append(frr)

        tp = np.sum(gen_dist < t)
        tn = np.sum(imp_dist > t)
        acc = (tp + tn) / (len(gen_dist) + len(imp_dist))

        if acc > best_acc:
            best_acc = acc
            optimal_threshold = t
            
        if round(t, 1) in [0.4, 0.6, 0.8]:
            print(f"{t:.2f}      | {far:.3f}   | {frr:.3f}   | {acc:.3f}")

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    
    # Calculate EER (where FAR ≈ FRR)
    eer_idx = np.nanargmin(np.absolute(far_list - frr_list))
    eer = far_list[eer_idx]

    # Plotting
    plt.figure()
    # TPR is 1 - FRR
    plt.plot(far_list, 1 - frr_list, label=f'{model_name} (EER: {eer:.3f})')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"roc_{model_name}.png") 
    plt.close() # Close to free up memory
    
    return eer, best_acc, optimal_threshold

def run_comparison():
    for model in models:
        data = load_benchmark_data(model)
        if not data or len(data) < 2: 
            print(f"Skipping {model}: Not enough data/identities found.")
            continue
        
        genuine_distances = []
        impostor_distances = []
        persons = list(data.keys())

        # Genuine Pairs (Within the same person)
        for person in persons:
            embs = data[person]
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    dist = calculate_distance(embs[i], embs[j])
                    genuine_distances.append(dist)

        # Impostor Pairs (Between different people)
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                # Compare first embedding of each person
                dist = calculate_distance(data[persons[i]][0], data[persons[j]][0])
                impostor_distances.append(dist)

        eer, acc, opt_t = compute_research_metrics(genuine_distances, impostor_distances, model)
        print(f"\n✅ {model} Final Results:")
        print(f"   EER: {eer:.3f}")
        print(f"   Best Accuracy: {acc:.3f} at Threshold: {opt_t:.2f}")

if __name__ == "__main__":
    run_comparison()