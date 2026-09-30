import pandas as pd

from src.data.clean import clean_total_charges


def test_zero_tenure_blank_total_charges_becomes_zero():
    df = pd.DataFrame(
        {
            "tenure": [0],
            "TotalCharges": [" "],
        }
    )

    result = clean_total_charges(df)

    assert result.loc[0, "TotalCharges"] == 0.0


def test_missing_total_charges_with_positive_tenure_fails():
    df = pd.DataFrame(
        {
            "tenure": [12],
            "TotalCharges": [" "],
        }
    )

    try:
        clean_total_charges(df)
        assert False, "Expected ValueError"
    except ValueError:
        assert True