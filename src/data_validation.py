from pathlib import Path
import pandas as pd


# ============================================================
# SWYNEX Technologies - Data Science Internship
# Task 1: Data Preparation
# Data Validation Script
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CLEANED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "netflix_titles_cleaned.csv"
)

REPORT_DIR = BASE_DIR / "outputs" / "reports"
VALIDATION_REPORT = REPORT_DIR / "validation_report.csv"


def main():

    print("=" * 60)
    print("SWYNEX DATA VALIDATION - TASK 1")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------------
    print("\n[1] Loading cleaned dataset...")

    if not CLEANED_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {CLEANED_FILE}"
        )

    df = pd.read_csv(
        CLEANED_FILE,
        low_memory=False
    )

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Required columns validation
    # --------------------------------------------------------
    print("\n[2] Checking required columns...")

    required_columns = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        columns_check = "FAIL"
    else:
        print("All required columns are present.")
        columns_check = "PASS"

    # --------------------------------------------------------
    # 3. Duplicate validation
    # --------------------------------------------------------
    print("\n[3] Checking duplicate rows...")

    duplicate_rows = int(df.duplicated().sum())

    print(f"Duplicate rows: {duplicate_rows}")

    duplicate_check = (
        "PASS"
        if duplicate_rows == 0
        else "FAIL"
    )

    # --------------------------------------------------------
    # 4. show_id uniqueness
    # --------------------------------------------------------
    print("\n[4] Checking show_id uniqueness...")

    duplicate_show_ids = int(
        df["show_id"].duplicated().sum()
    )

    print(
        f"Duplicate show IDs: "
        f"{duplicate_show_ids}"
    )

    show_id_check = (
        "PASS"
        if duplicate_show_ids == 0
        else "FAIL"
    )

    # --------------------------------------------------------
    # 5. Required fields validation
    # --------------------------------------------------------
    print("\n[5] Checking required fields...")

    required_fields = [
        "show_id",
        "type",
        "title",
        "release_year",
        "listed_in",
        "description"
    ]

    required_missing = int(
        df[required_fields].isna().sum().sum()
    )

    print(
        f"Missing values in required fields: "
        f"{required_missing}"
    )

    required_fields_check = (
        "PASS"
        if required_missing == 0
        else "FAIL"
    )

    # --------------------------------------------------------
    # 6. Release year validation
    # --------------------------------------------------------
    print("\n[6] Checking release years...")

    df["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    )

    current_year = pd.Timestamp.now().year

    invalid_release_years = int(
        (
            ~df["release_year"].between(
                1900,
                current_year
            )
        ).sum()
    )

    print(
        f"Invalid release years: "
        f"{invalid_release_years}"
    )

    release_year_check = (
        "PASS"
        if invalid_release_years == 0
        else "FAIL"
    )

    # --------------------------------------------------------
    # 7. Date validation
    # --------------------------------------------------------
    print("\n[7] Checking date_added...")

    df["date_added"] = pd.to_datetime(
        df["date_added"],
        errors="coerce"
    )

    missing_dates = int(
        df["date_added"].isna().sum()
    )

    print(
        f"Missing date_added values: "
        f"{missing_dates}"
    )

    # Missing dates are retained because
    # the actual date is unknown.
    date_check = "PASS"

    # --------------------------------------------------------
    # 8. Text quality validation
    # --------------------------------------------------------
    print("\n[8] Checking text fields...")

    text_columns = [
        "show_id",
        "type",
        "title",
        "listed_in",
        "description"
    ]

    empty_text_values = 0

    for column in text_columns:

        empty_values = int(
            df[column]
            .astype("string")
            .str.strip()
            .eq("")
            .sum()
        )

        empty_text_values += empty_values

    print(
        f"Empty text values in important fields: "
        f"{empty_text_values}"
    )

    text_check = (
        "PASS"
        if empty_text_values == 0
        else "FAIL"
    )

    # --------------------------------------------------------
    # 9. Missing values summary
    # --------------------------------------------------------
    print("\n[9] Final missing value check...")

    missing_values = df.isna().sum()

    print(missing_values)

    # --------------------------------------------------------
    # 10. Create validation report
    # --------------------------------------------------------
    validation_results = [
        {
            "validation": "Required columns",
            "result": columns_check,
            "details": "All expected columns are present"
        },
        {
            "validation": "Duplicate rows",
            "result": duplicate_check,
            "details": f"{duplicate_rows} duplicate rows found"
        },
        {
            "validation": "Unique show IDs",
            "result": show_id_check,
            "details": f"{duplicate_show_ids} duplicate show IDs found"
        },
        {
            "validation": "Required fields",
            "result": required_fields_check,
            "details": (
                f"{required_missing} missing values "
                "in required fields"
            )
        },
        {
            "validation": "Release year",
            "result": release_year_check,
            "details": (
                f"{invalid_release_years} invalid values found"
            )
        },
        {
            "validation": "date_added",
            "result": date_check,
            "details": (
                f"{missing_dates} dates remain unknown "
                "and are represented as missing"
            )
        },
        {
            "validation": "Important text fields",
            "result": text_check,
            "details": (
                f"{empty_text_values} empty text values found"
            )
        }
    ]

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df.to_csv(
        VALIDATION_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # 11. Overall validation status
    # --------------------------------------------------------
    failed_checks = int(
        (validation_df["result"] == "FAIL").sum()
    )

    print("\n" + "=" * 60)

    if failed_checks == 0:
        print("OVERALL VALIDATION STATUS: PASS")
    else:
        print(
            f"OVERALL VALIDATION STATUS: "
            f"{failed_checks} CHECK(S) FAILED"
        )

    print("=" * 60)

    print("\nValidation report:")
    print(VALIDATION_REPORT)


if __name__ == "__main__":
    main()