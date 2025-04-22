import json
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def save_metrics(metrics):
    with open("metrics.json", "w") as fp:
        json.dump(metrics, fp)
        
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculer les moyennes et les écarts-types des scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Tracer la courbe d'apprentissage
    plt.figure()
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('MSE')
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Erreur d\'entraînement')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Erreur de validation')

    plt.legend(loc='best')

    # Sauvegarder la figure
    plt.savefig('learning_curve.png')
    plt.show()