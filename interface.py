import os
import shutil
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from train import train_model
import joblib
import librosa

phrases = [
    "On n'apprend pas au vieux singe à faire des grimaces.",
    "Si les chats gardent les chèvres, qui attrapera les souris ?",
    "On ne fait pas d'omelette sans casser des oeufs.",
    "C'est dans les vieux pots qu'on fait les meilleures soupes!",
    "Le monde se divise en deux catégories : ceux qui tire et ceux qui creuses.",
    "C'est dans le besoin que l'on reconnaît ses vrais amis.",
]

def record_and_save_audio(filename, duration=5, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)

def normalize_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000)
    y = librosa.util.normalize(y)
    sf.write(output_path, y, sr)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def load_dataset(data_dir):
    X, y = [], []
    for user in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user)
        if os.path.isdir(user_path):
            for file in os.listdir(user_path):
                if file.endswith(".wav"):
                    features = extract_features(os.path.join(user_path, file))
                    X.append(features)
                    y.append(user)
    return np.array(X), np.array(y)

def predict_identity_with_proba(file_path):
    clf = joblib.load("model.pkl")
    features = extract_features(file_path).reshape(1, -1)
    prediction = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    classes = clf.classes_
    proba_dict = dict(zip(classes, proba))
    return prediction, proba_dict

def enregistrer_utilisateur(nom):
    dossier_utilisateur = os.path.join("data", "audio", nom)
    os.makedirs(dossier_utilisateur, exist_ok=True)

    try:
        for i, phrase in enumerate(phrases):
            validé = False
            while not validé:
                reponse = messagebox.askyesnocancel("Enregistrement", f"Phrase à dire ({i+1}/6) :\n{phrase}\n\nOK pour commencer à enregistrer ?\nNon ou Annuler pour quitter l'enregistrement. \n\n Conseil : Essayez de parler avec des intonnations différentes entres vos phrase pour un meilleur résultat.")
                
                if reponse is None or reponse is False:
                    raise KeyboardInterrupt("Enregistrement annulé")

                chemin_temp = "temp.wav"
                record_and_save_audio(chemin_temp)

                # Lecture
                data, samplerate = sf.read(chemin_temp)
                sd.play(data, samplerate)
                sd.wait()

                rep = messagebox.askyesno("Validation", "Souhaitez-vous valider cette phrase ?")
                if rep:
                    chemin_final = os.path.join(dossier_utilisateur, f"{nom}_{i+1}.wav")
                    normalize_audio(chemin_temp, chemin_final)
                    validé = True
                else:
                    messagebox.showinfo("Réessayer", "Réessayons cette phrase.")
        
        messagebox.showinfo("Succès", f"Utilisateur {nom} enregistré avec succès !")
        train_model()
    except KeyboardInterrupt:
        shutil.rmtree(dossier_utilisateur, ignore_errors=True)
        messagebox.showinfo("Annulé", f"Création de l'utilisateur '{nom}' annulée.")


def tester_utilisateur():
    chemin_temp = "test.wav"
    messagebox.showinfo("Enregistrement", "Cliquez sur OK et commencez à parler pendant 5 secondes.")
    record_and_save_audio(chemin_temp)
    normalize_audio(chemin_temp, chemin_temp)
    prediction, proba = predict_identity_with_proba(chemin_temp)
    proba_triee = sorted(proba.items(), key=lambda x: x[1], reverse=True)
    details = "\n".join([f"{nom} : {int(p * 100)}%" for nom, p in proba_triee])
    messagebox.showinfo("Résultat", f"Identité prédite : {prediction}\n\nDétails :\n{details}")

def supprimer_utilisateur(nom):
    dossier = os.path.join("data", "audio", nom)
    if os.path.exists(dossier):
        shutil.rmtree(dossier)
        messagebox.showinfo("Succès", f"Utilisateur '{nom}' supprimé.")
    else:
        messagebox.showwarning("Erreur", f"Utilisateur '{nom}' introuvable.")

def lister_utilisateurs():
    dossier = "data/audio"
    if not os.path.exists(dossier):
        messagebox.showinfo("Utilisateurs", "Aucun utilisateur enregistré.")
        return
    utilisateurs = [d for d in os.listdir(dossier) if os.path.isdir(os.path.join(dossier, d))]
    if utilisateurs:
        messagebox.showinfo("Utilisateurs", "Utilisateurs enregistrés :\n" + "\n".join(utilisateurs))
    else:
        messagebox.showinfo("Utilisateurs", "Aucun utilisateur enregistré.")

def lancer_interface():
    window = tk.Tk()
    window.title("Reconnaissance Vocale IA")

    # Nom utilisateur + bouton s'enregistrer
    frame_enreg = tk.Frame(window)
    frame_enreg.pack(pady=10, fill="x")

    label_nom = tk.Label(frame_enreg, text="Nom de l'utilisateur :")
    label_nom.pack(side=tk.LEFT, padx=(0,5))

    entry_nom = tk.Entry(frame_enreg)
    entry_nom.pack(side=tk.LEFT, fill="x", expand=True, padx=(0,5))

    def bouton_enregistrer():
        nom = entry_nom.get().strip()
        if nom:
            if os.path.exists(os.path.join("data", "audio", nom)):
                messagebox.showwarning("Erreur", f"Le nom '{nom}' existe déjà.")
            else:
                enregistrer_utilisateur(nom)
        else:
            messagebox.showwarning("Erreur", "Veuillez entrer un nom.")

    bouton_enregistrer_btn = tk.Button(frame_enreg, text="S'enregistrer", command=bouton_enregistrer)
    bouton_enregistrer_btn.pack(side=tk.LEFT)

    # Bouton Tester
    bouton_test = tk.Button(window, text="Tester", command=tester_utilisateur)
    bouton_test.pack(pady=5)

    # Nom utilisateur à supprimer + bouton supprimer utilisateur
    frame_supp = tk.Frame(window)
    frame_supp.pack(pady=10, fill="x")

    label_supp = tk.Label(frame_supp, text="Nom à supprimer :")
    label_supp.pack(side=tk.LEFT, padx=(0,5))

    entry_supp = tk.Entry(frame_supp)
    entry_supp.pack(side=tk.LEFT, fill="x", expand=True, padx=(0,5))

    def bouton_supprimer():
        nom = entry_supp.get().strip()
        if nom:
            supprimer_utilisateur(nom)
        else:
            messagebox.showwarning("Erreur", "Veuillez entrer un nom.")

    bouton_delete = tk.Button(frame_supp, text="Supprimer l'utilisateur", command=bouton_supprimer)
    bouton_delete.pack(side=tk.LEFT)

    # bouton Réentraîner
    bouton_train = tk.Button(window, text="Réentraîner le modèle", command=train_model)
    bouton_train.pack(pady=(10,5))

    # Bouton Liste des utilisateurs
    bouton_liste = tk.Button(window, text="Liste des utilisateurs", command=lister_utilisateurs)
    bouton_liste.pack(pady=5)

    # bouton quitter
    bouton_quitter = tk.Button(window, text="Quitter", command=window.quit)
    bouton_quitter.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    lancer_interface()
