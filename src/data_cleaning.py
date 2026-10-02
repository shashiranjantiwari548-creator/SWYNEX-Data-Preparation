from pathlib import Path
import pandas as pd


# ============================================================
# SWYNEX Technologies - Data Science Internship
# Task 1: Data Preparation
# Dataset: Netflix Movies and TV Shows
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
RAW_FILE = BASE_DIR / "data" / "raw" / "netflix_titles.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

PROCESSED_FILE = PROCESSED_DIR / "netflix_titles_cleaned.csv"
SUMMARY_FILE = REPORT_DIR / "cleaning_summary.csv"


def main():
    print("=" * 60)
    print("SWYNEX DATA PREPARATION - TASK 1")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Create required output directories
    # --------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2. Check raw dataset
    # --------------------------------------------------------
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}"
        )

    print("\n[1] Loading raw dataset...")

    df = pd.read_csv(
        RAW_FILE,
        low_memory=False
    )

    original_rows, original_columns = df.shape

    print(f"Rows: {original_rows}")
    print(f"Columns: {original_columns}")

    # --------------------------------------------------------
    # 3. Standardize column names
    # --------------------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # --------------------------------------------------------
    # 4. Clean text columns
    # --------------------------------------------------------
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        # Convert empty strings to missing values
        df[column] = df[column].mask(
            df[column].str.fullmatch(r"\s*", na=False),
            pd.NA
        )

    # --------------------------------------------------------
    # 5. Check duplicate rows
    # --------------------------------------------------------
    duplicate_rows = int(df.duplicated().sum())

    print("\n[2] Duplicate row check")
    print(f"Duplicate rows found: {duplicate_rows}")

    # Remove exact duplicate rows if any exist
    if duplicate_rows > 0:
        df = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # 6. Handle missing categorical/text values
    # --------------------------------------------------------
    missing_fill_columns = [
        "director",
        "cast",
        "country",
        "rating",
        "duration"
    ]

    print("\n[3] Handling missing values")

    for column in missing_fill_columns:
        if column in df.columns:
            missing_before = int(df[column].isna().sum())

            df[column] = df[column].fillna(
                "Not Specified"
            )

            print(
                f"{column}: "
                f"{missing_before} missing values handled"
            )

    # --------------------------------------------------------
    # 7. Convert date_added to datetime
    # --------------------------------------------------------
    print("\n[4] Converting date_added to datetime")

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(
            df["date_added"],
            errors="coerce"
        )

        invalid_dates = int(df["date_added"].isna().sum())

        print(
            f"Missing/invalid dates after conversion: "
            f"{invalid_dates}"
        )

    # --------------------------------------------------------
    # 8. Convert release_year to numeric integer type
    # --------------------------------------------------------
    print("\n[5] Validating release_year")

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(
            df["release_year"],
            errors="coerce"
        ).astype("Int64")

        current_year = pd.Timestamp.now().year

        invalid_years = int(
            (
                ~df["release_year"].between(
                    1900,
                    current_year
                )
            ).sum()
        )

        print(f"Invalid release years: {invalid_years}")

    # --------------------------------------------------------
    # 9. Validate show_id uniqueness
    # --------------------------------------------------------
    print("\n[6] Validating show_id")

    if "show_id" in df.columns:
        duplicate_show_ids = int(
            df["show_id"].duplicated().sum()
        )

        print(
            f"Duplicate show IDs: "
            f"{duplicate_show_ids}"
        )

    # --------------------------------------------------------
    # 10. Missing values after cleaning
    # --------------------------------------------------------
    missing_after = df.isna().sum()

    # --------------------------------------------------------
    # 11. Generate cleaning summary report
    # --------------------------------------------------------
    summary_data = [
        {
            "metric": "Original rows",
            "value": original_rows,
            "action": "Initial dataset size"
        },
        {
            "metric": "Original columns",
            "value": original_columns,
            "action": "Initial dataset column count"
        },
        {
            "metric": "Duplicate rows",
            "value": duplicate_rows,
            "action": "Removed exact duplicates if present"
        },
        {
            "metric": "Missing director handled",
            "value": "Not Specified",
            "action": "Filled missing categorical/text values"
        },
        {
            "metric": "Missing cast handled",
            "value": "Not Specified",
            "action": "Filled missing categorical/text values"
        },
        {
            "metric": "Missing country handled",
            "value": "Not Specified",
            "action": "Filled missing categorical/text values"
        },
        {
            "metric": "Missing rating handled",
            "value": "Not Specified",
            "action": "Filled missing categorical values"
        },
        {
            "metric": "Missing duration handled",
            "value": "Not Specified",
            "action": "Filled missing categorical values"
        },
        {
            "metric": "date_added type",
            "value": str(df["date_added"].dtype),
            "action": "Converted to datetime"
        },
        {
            "metric": "release_year type",
            "value": str(df["release_year"].dtype),
            "action": "Converted to numeric nullable integer"
        },
        {
            "metric": "Final rows",
            "value": len(df),
            "action": "Cleaned dataset"
        },
        {
            "metric": "Final columns",
            "value": len(df.columns),
            "action": "Cleaned dataset"
        }
    ]

    summary_df = pd.DataFrame(summary_data)

    # --------------------------------------------------------
    # 12. Save cleaned dataset
    # --------------------------------------------------------
    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 13. Save cleaning report
    # --------------------------------------------------------
    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 14. Final validation output
    # --------------------------------------------------------
    print("\n[7] Final dataset validation")

    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    print("\nRemaining missing values:")
    print(missing_after)

    print("\nFinal data types:")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)

    print(f"\nCleaned dataset:")
    print(PROCESSED_FILE)

    print(f"\nCleaning report:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()