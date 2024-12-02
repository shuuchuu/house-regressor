"""Fonctions utilitaires."""

from pandas import DataFrame


def row_to_json(df: DataFrame, row: int) -> str:
    """Convertit la ligne `row` de la table `df` en JSON.

    Args:
        df: DataFrame pandas.
        row (int): Index de la ligne à convertir.

    Returns:
        Export JSON de la ligne.
    """
    return df.iloc[[row]].to_json()
