import os
import librosa
import numpy as np

def extract_features(file_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def load_dataset(audio_dir):
    X, y = [], []
    for person in os.listdir(audio_dir):
        person_dir = os.path.join(audio_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if fname.endswith(".wav"):
                fpath = os.path.join(person_dir, fname)
                features = extract_features(fpath)
                X.append(features)
                y.append(person)
    return np.array(X), np.array(y)

def normalize_audio(audio, target_amplitude=0.9):
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return audio
    return audio * (target_amplitude / max_amplitude)
