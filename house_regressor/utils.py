from pandas import DataFrame


def row_to_json(df: DataFrame, row: int) -> str:
    return df.iloc[[row]].to_json()
