"""Simple utility functions for the churn prediction project."""
import os
import pandas as pd
import joblib


def save_dataframe(df, path, description=''):
    """Save a DataFrame to CSV with a confirmation message."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    desc = f" ({description})" if description else ''
    print(f"Saved: {os.path.basename(path)} - {len(df):,} rows{desc}")


def load_dataframe(path):
    """Load a DataFrame from CSV."""
    df = pd.read_csv(path)
    print(f"Loaded: {os.path.basename(path)} - {len(df):,} rows, {len(df.columns)} cols")
    return df


def save_model(model, path, model_name=''):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    name = model_name or os.path.basename(path)
    print(f"Saved model: {name}")


def load_model(path):
    """Load a trained model from disk using joblib."""
    model = joblib.load(path)
    print(f"Loaded model: {os.path.basename(path)}")
    return model


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_subheader(title):
    """Print a formatted sub-header."""
    print(f"\n--- {title} ---")


def describe_nulls(df, threshold=0.0):
    """Print columns with null percentages above a given threshold."""
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > threshold]
    if len(null_pct) > 0:
        print(f"\n  Columns with >{threshold:.0f}% nulls:")
        for col, pct in null_pct.items():
            print(f"    {col:40s} {pct:6.1f}%  ({df[col].isnull().sum():,} rows)")
    else:
        print("  No columns with significant nulls.")
    return null_pct
