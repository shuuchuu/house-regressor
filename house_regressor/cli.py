from argparse import ArgumentParser

from pandas import read_csv

from .training import train_model
from .utils import row_to_json


def main() -> None:
    # Création d'un parseur
    parser = ArgumentParser("tp-ghcli")
    # Création d'un gestionnaire de commandes
    subparsers = parser.add_subparsers(help="Commands", dest="command", required=True)
    # Création d'un parseur pour la commande train
    train_parser = subparsers.add_parser("train")
    # Ajout d'un argument pour les données
    train_parser.add_argument("data_path", help="Chemin des données")
    # Ajout d'un argument pour les labels
    train_parser.add_argument("labels_path", help="Chemin des labels")
    # Ajout d'un argument pour la variance, de type flottant et avec une valeur par
    # défaut
    train_parser.add_argument(
        "--kept-variance",
        type=float,
        default=0.99,
        help="Variance conservée pendant la PCA",
    )
    test_json_parser = subparsers.add_parser("test-json")
    # Ajout d'un argument pour les données
    test_json_parser.add_argument("data_path", help="Chemin des données")
    # Ajout d'un argument pour l'index de la ligne à renvoyer en JSON
    test_json_parser.add_argument(
        "test_row", type=int, help="Ligne des données de test à transformer en JSON"
    )

    # Récupération des arguments
    args = parser.parse_args()

    # Appel de la fonction train
    if args.command == "train":
        # Appel de train_model avec les arguments récupérés
        train_model(
            training_data_path=args.data_path,
            training_labels_path=args.labels_path,
            kept_variance=args.kept_variance,
        )

    # Appel de la fonction test-json
    if args.command == "test-json":
        # Chargement des données
        data = read_csv(args.data_path)
        # Appel de row_to_json
        print(row_to_json(data, args.test_row))
