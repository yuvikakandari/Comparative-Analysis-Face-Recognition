import numpy as np
import cv2
from deepface import DeepFace

def get_embedding(image, model_name="Facenet"):
    """
    Extracts face embeddings using the specified model.
    Supported: "Facenet", "VGG-Face", "ArcFace"
    """
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extraction with detection disabled for LFW stability
        result = DeepFace.represent(
            img_path=image_rgb, 
            model_name=model_name, 
            enforce_detection=False
        )

        emb = np.array(result[0]["embedding"])
        
        # Unit Vector Normalization (Crucial for Cosine Similarity)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
            
        return emb

    except Exception as e:
        print(f"Error extracting {model_name}: {e}")
        return None

def calculate_distance(emb1, emb2):
    """
    Calculates Cosine Distance: 1 - Cosine Similarity
    Range: 0 (Same) to 2 (Different)
    """
    dot_product = np.clip(np.dot(emb1, emb2), -1.0, 1.0)
    return 1.0 - dot_product