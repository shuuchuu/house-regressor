from io import StringIO
from pathlib import Path
from pickle import load
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import FastAPI
from mlflow.artifacts import download_artifacts
from mlflow.client import MlflowClient
from mlflow.sklearn import load_model
from pandas import read_json

app = FastAPI()

client = MlflowClient()


def load_artifact(run_id: str, path: str) -> Any:
    """
    Charge un artefact sauvé avec pickle qui est le seul fichier de son dossier.

    Args:
        run_id: run_id de l'artefact à charger.
        path (str): Chemin de l'artefact à charger relatif à la racine du run.

    Returns:
        Objet chargé depuis pickle.
    """
    # Création d'un dossier temporaire pour le téléchargement de l'artefact
    with TemporaryDirectory() as temp_dir:
        # Téléchargement depuis le dépôt MLFlow
        download_artifacts(run_id=run_id, artifact_path=path, dst_path=temp_dir)
        # Récupération du chemin du premier élément du dossier (normalement le seul)
        pickle_path = next((Path(temp_dir) / path).iterdir())
        # Chargement de l'objet
        with pickle_path.open("rb") as fh:
            return load(fh)


def get_latest_version_and_runid(name: str) -> tuple[str, str]:
    """
    Récupère la dernière version et le run_id d'un modèle par son nom.

    Args:
        name: Nom du modèle.

    Returns:
        La dernière version et le run_id du modèle.
    """
    result = client.search_registered_models(f"name = '{name}'")
    last_version = next(iter(result)).latest_versions[0]
    return last_version.version, last_version.run_id


# Récupération des informations du modèle à charger
model_name = "ames-regressor"
version, run_id = get_latest_version_and_runid(model_name)

# Chargement des artefacts qui ont été créé avec pickle et respectent la convention d'un
# artefact par dossier
pca = load_artifact(run_id, "sk_pca")
standard_scaler = load_artifact(run_id, "sk_standard_scaler")

# Chargement du modèle au lancement de l'API
model = load_model(model_uri="models:/{model_name}/{version}")


@app.post("/")
async def classify(data: str) -> float:
    """
    Point d'entrée de prédiction.

    Args:
        data: Données d'entrée sous forme de DataFrame convertie en chaîne de caractère
              JSON (souvent avec to_json())

    Returns:
        La prédiction du modèle
    """
    array = read_json(StringIO(data)).values
    return model.predict(pca.transform(standard_scaler.transform(array)))[0, 0]
