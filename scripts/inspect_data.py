import pandas as pd

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n--- DATASET SHAPE ---")
    print(df.shape)

    print("\n--- COLUMNS ---")
    print(df.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- EMPTY / WHITESPACE VALUES ---")
    for column in df.columns:
        empty_count = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
        if empty_count > 0:
            print(f"{column}: {empty_count}")
            
    print("\n--- TOTAL CHARGES UNIQUE SAMPLE ---")
    print(df["TotalCharges"].unique()[:20])
    
    print("\n--- TOTAL CHARGES EMPTY ROWS ---")
    print(
        df[df["TotalCharges"].astype(str).str.strip() == ""]
        [["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    )

    print("\n--- DUPLICATES ---")
    print(df.duplicated().sum())

    print("\n--- TARGET DISTRIBUTION ---")
    print(df["Churn"].value_counts())


if __name__ == "__main__":
    main()