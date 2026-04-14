"""Data cleaning functions.

Addresses all cleaning issues identified in the audit:
1. Duplicate removal (exact + Case ID dedup)
2. CompanySize Excel date corruption fix
3. Customer Tier inconsistency (Platinum+ vs Platinum +)
4. Negative value handling
5. Zero VAN flagging
6. BoB non-active record filtering
7. Resolution status filtering
8. Proper null handling (categorical vs numeric)
"""
import pandas as pd
import numpy as np
from ..utils.config import (
    COMPANY_SIZE_MAP, COMPANY_SIZE_NUMERIC_MAP,
    CHURN_STATUS, CHURN_REASON_KEYWORDS
)


def remove_duplicates(df: pd.DataFrame, id_col: str = None) -> pd.DataFrame:
    """Remove exact duplicate rows and optionally deduplicate by ID.

    Args:
        df: Input DataFrame.
        id_col: If provided, keep first row per unique ID after removing exact duplicates.

    Returns:
        Deduplicated DataFrame.
    """
    n_before = len(df)

    # Step 1: Remove exact duplicates
    df = df.drop_duplicates()
    n_exact = n_before - len(df)

    # Step 2: Deduplicate by ID column (keep first occurrence)
    n_id_dupes = 0
    if id_col and id_col in df.columns:
        n_before_id = len(df)
        df = df.drop_duplicates(subset=id_col, keep='first')
        n_id_dupes = n_before_id - len(df)

    print(f"  Deduplication: {n_before:,} -> {len(df):,} rows")
    print(f"    Exact duplicates removed: {n_exact:,}")
    if id_col:
        print(f"    ID duplicates removed ({id_col}): {n_id_dupes:,}")

    return df


def fix_company_size(df: pd.DataFrame, col: str = 'CompanySize') -> pd.DataFrame:
    """Fix CompanySize column: map Excel-corrupted values and create numeric version.

    Args:
        df: DataFrame with CompanySize column.
        col: Name of the column to fix.

    Returns:
        DataFrame with fixed CompanySize and CompanySize_numeric columns.
    """
    if col not in df.columns:
        return df

    before_vals = df[col].value_counts().to_dict()
    df[col] = df[col].map(COMPANY_SIZE_MAP).fillna(df[col])
    df['CompanySize_numeric'] = df[col].map(COMPANY_SIZE_NUMERIC_MAP)
    after_vals = df[col].value_counts().to_dict()

    print(f"  CompanySize fix:")
    print(f"    Before: {before_vals}")
    print(f"    After:  {after_vals}")

    return df


def fix_customer_tier(df: pd.DataFrame, col: str = 'Customer Tier') -> pd.DataFrame:
    """Standardize Customer Tier values.

    Merges 'Platinum +' (with space) into 'Platinum+' (no space).

    Args:
        df: DataFrame with Customer Tier column.
        col: Name of the column to fix.

    Returns:
        DataFrame with standardized tier values.
    """
    if col not in df.columns:
        return df

    before = df[col].value_counts().to_dict()
    df[col] = df[col].str.strip().replace({'Platinum +': 'Platinum+'})
    after = df[col].value_counts().to_dict()

    print(f"  Customer Tier fix:")
    print(f"    Before: {before}")
    print(f"    After:  {after}")

    return df


def handle_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle negative values in numeric columns.

    Clips New VAN negatives to 0 (negative VAN = revenue clawback, treat as zero).

    Args:
        df: DataFrame.

    Returns:
        DataFrame with negatives handled.
    """
    for col in ['New VAN']:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                df[col] = df[col].clip(lower=0)
                print(f"  Clipped {n_neg} negative values in '{col}' to 0")

    return df


def flag_zero_van(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with zero VAN.

    Args:
        df: DataFrame with VAN column.

    Returns:
        DataFrame with 'is_zero_van' flag.
    """
    if 'VAN' in df.columns:
        df['is_zero_van'] = (df['VAN'] == 0).astype(int)
        n_zero = df['is_zero_van'].sum()
        print(f"  Zero VAN rows flagged: {n_zero:,} ({n_zero / len(df):.1%})")

    return df


def filter_resolved_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only resolved cases and create target variable.

    Args:
        df: Raw retention DataFrame.

    Returns:
        Filtered DataFrame with 'is_churned' target column.
    """
    resolved_statuses = ['Customer Lost', 'Customer Saved']
    mask = df['Resolution Status'].isin(resolved_statuses)
    df_resolved = df[mask].copy()

    df_resolved['is_churned'] = (df_resolved['Resolution Status'] == CHURN_STATUS).astype(int)

    churn_rate = df_resolved['is_churned'].mean()
    print(f"  Filtered to resolved cases: {len(df_resolved):,} rows")
    print(f"    Customer Lost: {df_resolved['is_churned'].sum():,} ({churn_rate:.1%})")
    print(f"    Customer Saved: {(~df_resolved['is_churned'].astype(bool)).sum():,} ({1 - churn_rate:.1%})")

    return df_resolved


def categorize_churn_reason(title: str) -> str:
    """Categorize churn reason from Case Title text.

    Args:
        title: Case Title string.

    Returns:
        Churn reason category.
    """
    if pd.isna(title):
        return 'Unknown'
    title_lower = str(title).lower()

    for reason, keywords in CHURN_REASON_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return reason

    return 'Other'


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Handle null values with appropriate strategies per column type.

    - Categorical columns: fill with 'Unknown' or mode
    - Numeric columns: fill with median
    - Creates missing indicator flags for high-null columns

    Args:
        df: DataFrame.

    Returns:
        DataFrame with nulls handled.
    """
    print("  Null handling:")

    # Categorical nulls → 'Unknown'
    cat_fill = {
        'Risk': 'Unknown',
        'Pull Type': 'None',
        'Customer Tier': 'Unknown',
        'CompanySize': 'Unknown',
        'Case Origin': 'Unknown',
        'Branch': 'Unknown',
    }
    for col, fill_val in cat_fill.items():
        if col in df.columns and df[col].isnull().sum() > 0:
            # Create missing indicator for high-null columns
            null_pct = df[col].isnull().mean()
            if null_pct > 0.1:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(fill_val)
            print(f"    {col}: filled {df[col].isnull().sum()} nulls with '{fill_val}'")

    # Numeric nulls → median or 0
    numeric_zero_fill = ['Number Of Repair Cases', 'Number of OverdueServices']
    for col in numeric_zero_fill:
        if col in df.columns and df[col].isnull().sum() > 0:
            n = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            print(f"    {col}: filled {n} nulls with 0")

    # Remaining numeric → median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            n = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"    {col}: filled {n} nulls with median ({median_val:.2f})")

    remaining = df.isnull().sum().sum()
    print(f"  Remaining nulls: {remaining}")

    return df


def clean_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for Retention data.

    Args:
        df: Raw retention DataFrame.

    Returns:
        Cleaned and filtered DataFrame.
    """
    print("\n  Cleaning Retention data...")

    # 0. Drop columns that are 100% null
    drop_cols = [c for c in df.columns if df[c].isnull().all()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Dropped 100% null columns: {drop_cols}")

    # 1. Remove duplicates
    df = remove_duplicates(df, id_col='Case ID')

    # 2. Filter to resolved cases + create target
    df = filter_resolved_cases(df)

    # 3. Fix CompanySize
    df = fix_company_size(df)

    # 4. Fix Customer Tier
    df = fix_customer_tier(df)

    # 5. Handle negatives
    df = handle_negative_values(df)

    # 6. Flag zero VAN
    df = flag_zero_van(df)

    # 7. Categorize churn reasons
    df['churn_reason_group'] = df['Case Title'].apply(categorize_churn_reason)
    print(f"  Churn reasons categorized: {df['churn_reason_group'].value_counts().to_dict()}")

    print(f"\n  Cleaning complete: {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_bob(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for BoB data.

    Args:
        df: Raw BoB DataFrame.

    Returns:
        Cleaned DataFrame (active records only, deduplicated).
    """
    print("\n  Cleaning BoB data...")

    # 1. Remove duplicates
    df = remove_duplicates(df)

    # 2. Filter to active records only
    n_before = len(df)
    df = df[df['system_status'] == 'Active'].copy()
    print(f"  Filtered to Active records: {n_before:,} -> {len(df):,}")

    # 3. Fix empty company_sizing
    if 'company_sizing' in df.columns:
        df['company_sizing'] = df['company_sizing'].replace('', np.nan)

    print(f"\n  BoB cleaning complete: {len(df):,} rows, {len(df.columns)} columns")
    return df
