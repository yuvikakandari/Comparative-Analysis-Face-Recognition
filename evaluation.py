import matplotlib.pyplot as plt
def compute_research_metrics(genuine_distances, impostor_distances, model_name):
    thresholds = np.linspace(0, 1.2, 100)
    far_list = []
    frr_list = []
    best_acc = 0
    optimal_threshold = 0
    
    print(f"\n--- {model_name} Threshold Sweep ---")
    print("Threshold | FAR     | FRR     | Accuracy")
    print("-" * 40)
    
    for t in thresholds:
        # Calculate FAR and FRR
        far = np.mean(np.array(impostor_distances) < t)
        frr = np.mean(np.array(genuine_distances) > t)
        
        far_list.append(far)
        frr_list.append(frr)

        # Calculate Accuracy
        tp = np.sum(np.array(genuine_distances) < t)
        tn = np.sum(np.array(impostor_distances) > t)
        acc = (tp + tn) / (len(genuine_distances) + len(impostor_distances))

        if acc > best_acc:
            best_acc = acc
            optimal_threshold = t
            
        # Print progress for key thresholds
        if round(t, 1) in [0.4, 0.6, 0.8]:
            print(f"{t:.2f}      | {far:.3f}   | {frr:.3f}   | {acc:.3f}")

    # Convert to arrays AFTER the loop
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    
    # Calculate EER
    eer_idx = np.nanargmin(np.absolute(far_list - frr_list))
    eer = far_list[eer_idx]

    # Plotting (Outside the loop!)
    plt.figure() # Create a new figure for each model
    plt.plot(far_list, 1 - frr_list, label=f'{model_name} (EER: {eer:.3f})')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"roc_{model_name}.png") 
    
    return eer, best_acc, optimal_threshold
def run_comparison():
    for model in models:
        data = load_benchmark_data(model)
        if not data: 
            print(f"Directory {lfw_path} not found or empty!")
            continue
        
        genuine_distances = []
        impostor_distances = []
        persons = list(data.keys())

        # Create Genuine Pairs
        for person in persons:
            embs = data[person]
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    dist = calculate_distance(embs[i], embs[j])
                    genuine_distances.append(dist)

        # Create Impostor Pairs
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                dist = calculate_distance(data[persons[i]][0], data[persons[j]][0])
                impostor_distances.append(dist)

        # NOW call the metrics function
        eer, acc, opt_t = compute_research_metrics(genuine_distances, impostor_distances, model)
        print(f"Results for {model}: EER={eer:.3f}, Best Acc={acc:.3f} at {opt_t:.2f}")

if __name__ == "__main__":
    run_comparison()