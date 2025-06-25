# ----------------------- Modele d'entrainement avec RandomForestClassifier ----------------------- #
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib
# from tkinter import messagebox
# from preprocess import load_dataset

# def train_model():
#     X, y = load_dataset("data/audio")
#     if len(set(y)) < 2:
#         messagebox.showwarning("Erreur", "Il faut au moins deux utilisateurs pour entraîner.")
#         return
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(classification_report(y_test, y_pred))
#     joblib.dump(clf, "model.pkl")
#     messagebox.showinfo("Succès", "Modèle entraîné avec succès.")

# if __name__ == "__main__":
#     train_model()
# ------------------------------------------------------------------------------------------------- #


# ----------------------- Modele d'entrainement plus complexe ----------------------- #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import messagebox
from preprocess import load_dataset

def train_model():
    X, y = load_dataset("data/audio")

    if len(set(y)) < 2:
        messagebox.showwarning("Erreur", "Il faut au moins deux utilisateurs pour entraîner.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Résultats
    best_model = grid_search.best_estimator_
    print("Meilleurs paramètres :", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.show()

    # Sauvegarde du modèle
    joblib.dump(best_model, "model.pkl")
    messagebox.showinfo("Succès", "Modèle entraîné avec succès.")

if __name__ == "__main__":
    train_model()
# ----------------------------------------------------------------------------------- #