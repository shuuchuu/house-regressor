from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(
    training_data_path: str,
    training_labels_path: str,
    kept_variance: float,
) -> None:
    """
    Entraînement d'un modèle de forêt d'arbres aléatoires de régression.

    Args:
        training_data_path: Chemin du CSV des données d'entraînement.
        training_labels_path: Chemin du CSV des labels d'entraînement.
        kept_variance: Variance à conserver pendant la PCA.
    """
    # Code ML d'entraînement
    # Chargement des données
    X = read_csv(training_data_path)
    y = read_csv(training_labels_path)
    # Split train/eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    # Normalisation
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X_train.values)
    # Réduction de dimensionnalité
    pca = PCA(n_components=kept_variance)
    X_pca = pca.fit_transform(X_scaled)
    # Entraînement du modèle
    rfr = RandomForestRegressor()
    rfr.fit(X_pca, y_train)
    # Calcul du r2 sur la validation
    val_r2 = rfr.score(pca.transform(standard_scaler.transform(X_test.values)), y_test)
