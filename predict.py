import sys
import joblib
import numpy as np
from preprocess import extract_features

def predict_identity(audio_path):
    model = joblib.load("model.pkl")
    features = extract_features(audio_path)
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]

    # Récupère les 3 classes les plus probables
    top_indices = np.argsort(probabilities)[::-1][:3]
    print("Top prédictions :")
    for i in top_indices:
        label = model.classes_[i]
        confidence = probabilities[i]
        print(f"  {label} - {confidence*100:.2f} %")

    return prediction

if __name__ == "__main__":
    audio_file = sys.argv[1]
    print("\n\nRésultat final :")
    print("Prédiction:", predict_identity(audio_file))
