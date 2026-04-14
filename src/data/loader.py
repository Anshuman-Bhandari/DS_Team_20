"""Data loading functions."""
import pandas as pd
from ..utils.config import RETENTION_FILE, BOB_FILE


def load_retention(path: str = None) -> pd.DataFrame:
    """Load the Retention dataset.

    Args:
        path: Path to the CSV file. Defaults to config path.

    Returns:
        Raw retention DataFrame.
    """
    path = path or RETENTION_FILE
    df = pd.read_csv(path, encoding='latin-1')
    print(f"  Loaded Retention: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Unique Case IDs: {df['Case ID'].nunique():,}")
    print(f"  Unique Accounts: {df['Customer Account Number'].dropna().nunique():,}")
    return df


def load_bob(path: str = None) -> pd.DataFrame:
    """Load the Book of Business (BoB) dataset.

    Args:
        path: Path to the Excel file. Defaults to config path.

    Returns:
        Raw BoB DataFrame.
    """
    path = path or BOB_FILE
    df = pd.read_excel(path)
    print(f"  Loaded BoB: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Unique Accounts: {df['account_number'].nunique():,}")
    print(f"  Unique Agreements: {df['agreement_number'].nunique():,}")
    return df


def analyze_overlap(retention: pd.DataFrame, bob: pd.DataFrame) -> dict:
    """Analyze the overlap between Retention and BoB datasets.

    Args:
        retention: Retention DataFrame.
        bob: BoB DataFrame.

    Returns:
        Dictionary with overlap statistics.
    """
    ret_accts = set(retention['Customer Account Number'].dropna().unique())
    bob_accts = set(bob['account_number'].dropna().unique())

    common = ret_accts & bob_accts
    only_ret = ret_accts - bob_accts
    only_bob = bob_accts - ret_accts

    stats = {
        'retention_accounts': len(ret_accts),
        'bob_accounts': len(bob_accts),
        'common': len(common),
        'only_retention': len(only_ret),
        'only_bob': len(only_bob),
        'overlap_pct': len(common) / len(ret_accts) * 100 if ret_accts else 0,
    }

    print(f"\n  Dataset Overlap Analysis:")
    print(f"    Retention unique accounts: {stats['retention_accounts']:,}")
    print(f"    BoB unique accounts:       {stats['bob_accounts']:,}")
    print(f"    Common (overlapping):      {stats['common']:,}")
    print(f"    Only in Retention:         {stats['only_retention']:,}")
    print(f"    Only in BoB:               {stats['only_bob']:,}")
    print(f"    Overlap rate:              {stats['overlap_pct']:.1f}%")

    return stats


def profile_dataframe(df: pd.DataFrame, name: str = 'DataFrame'):
    """Print a detailed profile of a DataFrame.

    Args:
        df: DataFrame to profile.
        name: Display name for the DataFrame.
    """
    print(f"\n  {name} Profile:")
    print(f"    Shape: {df.shape}")
    print(f"    Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Data types
    dtypes = df.dtypes.value_counts()
    print(f"    Data types: {dict(dtypes)}")

    # Nulls
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0].sort_values(ascending=False)
    if len(null_cols) > 0:
        print(f"    Columns with nulls: {len(null_cols)}")
        for col in null_cols.head(5).index:
            pct = null_cols[col] / len(df) * 100
            print(f"      {col}: {null_cols[col]:,} ({pct:.1f}%)")
        if len(null_cols) > 5:
            print(f"      ... and {len(null_cols) - 5} more")
    else:
        print("    No null values")
