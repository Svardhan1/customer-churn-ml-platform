import pandas as pd


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the TotalCharges column.

    Business rule:
    - Blank TotalCharges is valid when tenure == 0.
    - Blank TotalCharges for tenure > 0 is treated as a data-quality issue.
    """

    df = df.copy()

    # Convert whitespace-only values to missing values
    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    # Identify invalid missing TotalCharges records
    invalid_missing = (
        df["TotalCharges"].isna()
        & (df["tenure"] > 0)
    )

    if invalid_missing.any():
        invalid_count = invalid_missing.sum()

        raise ValueError(
            f"Found {invalid_count} customers with missing "
            "TotalCharges but tenure > 0."
        )

    # Valid case: new customers with zero tenure
    zero_tenure_missing = (
        df["TotalCharges"].isna()
        & (df["tenure"] == 0)
    )

    df.loc[zero_tenure_missing, "TotalCharges"] = "0"

    # Convert to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="raise"
    )

    return df