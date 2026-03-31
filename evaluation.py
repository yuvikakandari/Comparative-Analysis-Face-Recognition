def compute_research_metrics(genuine_distances, impostor_distances):
    thresholds = np.linspace(0, 2, 100) # Cosine distance range is 0 to 2
    best_acc = 0
    optimal_threshold = 0
    
    print("Threshold | FAR     | FRR     | Accuracy")
    print("-" * 40)
    
    for t in thresholds:
        # FAR: Impostors who were accepted (distance < t)
        far = np.mean(np.array(impostor_distances) < t)
        
        # FRR: Genuine users who were rejected (distance > t)
        frr = np.mean(np.array(genuine_distances) > t)
        
        # Accuracy at this specific threshold
        tp = np.sum(np.array(genuine_distances) < t)
        tn = np.sum(np.array(impostor_distances) > t)
        acc = (tp + tn) / (len(genuine_distances) + len(impostor_distances))
        
        if acc > best_acc:
            best_acc = acc
            optimal_threshold = t
            
        # Optional: Print a few samples to see the trend
        if round(t, 1) in [0.4, 0.6, 0.8]:
            print(f"{t:.2f}      | {far:.3f}   | {frr:.3f}   | {acc:.3f}")

    return best_acc, optimal_threshold