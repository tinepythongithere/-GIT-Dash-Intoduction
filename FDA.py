import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

lda = LDA(priors=[1/3, 1/3, 1/3])
lda.fit(X, y)

y_pred = lda.predict(X)

# Calcul du taux de bon classement
accuracy = accuracy_score(y, y_pred)

# Création du tableau de confusion
conf_matrix = confusion_matrix(y, y_pred)
conf_matrix = pd.DataFrame(conf_matrix, columns=['setosa', 'versicolor', 'virginica'],
                                               index=['setosa', 'versicolor', 'virginica'])


def predict_species_with_proba(features):
    """
    Prédit l'espèce d'iris et retourne les probabilités associées à chaque classe.
    :param features: Liste ou tableau contenant les 4 caractéristiques (float).
    :return: Tuple contenant l'espèce prédite et les probabilités associées.
    """
    proba = lda.predict_proba([features])[0]  # Probabilités pour chaque classe
    prediction = lda.predict([features])[0]  # Classe prédite
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    return species_map[prediction], proba