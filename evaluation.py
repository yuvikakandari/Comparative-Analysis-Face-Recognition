def apply_stress(img, mode=None):
    if mode == "low_light":
        # Reduce brightness by 60% (Luminance Attenuation) [cite: 115]
        return cv2.convertScaleAbs(img, alpha=0.4, beta=0)
    elif mode == "blur":
        # Apply 15x15 Gaussian Blur (Gaussian Perturbation) [cite: 116]
        return cv2.GaussianBlur(img, (15, 15), 0)
    return img

# Update your load_benchmark_data function:
def load_benchmark_data(model_name, stress=None):
    print(f"\n[INFO] Extracting embeddings for {model_name} (Stress: {stress})...")
    data = {}
    person_list = os.listdir(lfw_path)[:25]
    for person in person_list:
        person_path = os.path.join(lfw_path, person)
        embeddings = []
        for img_name in os.listdir(person_path)[:5]:
            img = cv2.imread(os.path.join(person_path, img_name))
            if img is None: continue
            
            # Apply the selected stressor
            img = apply_stress(img, mode=stress)
            
            emb = get_embedding(img, model_name)
            if emb is not None: embeddings.append(emb)
        if len(embeddings) >= 2:
            data[person] = embeddings
    return data