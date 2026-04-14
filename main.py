"""
Customer Churn Prediction Pipeline (Simplified)
=================================================
Runs the complete analysis end-to-end:
1. Load and clean data
2. Engineer features
3. Select features
4. Train and evaluate models
5. Generate business outputs (priority bands, risk table)

Usage:
    python main.py
"""
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# add project root to path so we can import config and src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

from config import (
    RETENTION_FILE, BOB_FILE, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR,
    COMPANY_SIZE_MAP, COMPANY_SIZE_NUMERIC_MAP, CHURN_STATUS, CHURN_REASON_KEYWORDS,
    DATE_COLS, TIER_MAP, EXCLUDE_FROM_MODELING, PRIORITY_BANDS, RETENTION_PLAYBOOK,
    RANDOM_STATE, TEST_SIZE
)
from src.util import print_header, print_subheader, save_dataframe, save_model

# try importing xgboost and lightgbm
try:
    from xgboost import XGBClassifier
except ImportError:
    print("Warning: xgboost not installed. XGBoost model will be skipped.")
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("Warning: lightgbm not installed. LightGBM model will be skipped.")
    LGBMClassifier = None


# ── PHASE 1: DATA LOADING ──────────────────────────────────
def load_data():
    """Load raw retention and BoB datasets."""
    print_header("PHASE 1: DATA LOADING")

    print_subheader("Loading Retention data")
    retention = pd.read_csv(RETENTION_FILE, encoding='latin-1')
    print(f"  Loaded Retention: {len(retention):,} rows, {len(retention.columns)} columns")

    print_subheader("Loading BoB data")
    bob = pd.read_excel(BOB_FILE)
    print(f"  Loaded BoB: {len(bob):,} rows, {len(bob.columns)} columns")

    return retention, bob


# ── PHASE 2: DATA CLEANING ─────────────────────────────────
def categorize_churn_reason(title):
    """Categorize churn reason from Case Title text."""
    if pd.isna(title):
        return 'Unknown'
    title_lower = str(title).lower()
    for reason, keywords in CHURN_REASON_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return reason
    return 'Other'


def clean_retention(df):
    """Clean the retention dataset."""
    print_subheader("Cleaning Retention")

    # drop columns that are 100% null
    drop_cols = [c for c in df.columns if df[c].isnull().all()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Dropped 100% null columns: {drop_cols}")

    # remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_exact = n_before - len(df)
    n_before_id = len(df)
    df = df.drop_duplicates(subset='Case ID', keep='first')
    n_id = n_before_id - len(df)
    print(f"  Deduplication: {n_before:,} -> {len(df):,} rows (exact: {n_exact:,}, ID: {n_id:,})")

    # filter to resolved cases and create target
    resolved_statuses = ['Customer Lost', 'Customer Saved']
    df = df[df['Resolution Status'].isin(resolved_statuses)].copy()
    df['is_churned'] = (df['Resolution Status'] == CHURN_STATUS).astype(int)
    churn_rate = df['is_churned'].mean()
    print(f"  Filtered to resolved: {len(df):,} rows, churn rate: {churn_rate:.1%}")

    # fix CompanySize
    if 'CompanySize' in df.columns:
        df['CompanySize'] = df['CompanySize'].map(COMPANY_SIZE_MAP).fillna(df['CompanySize'])
        df['CompanySize_numeric'] = df['CompanySize'].map(COMPANY_SIZE_NUMERIC_MAP)

    # fix Customer Tier
    if 'Customer Tier' in df.columns:
        df['Customer Tier'] = df['Customer Tier'].str.strip().replace({'Platinum +': 'Platinum+'})

    # handle negative values in New VAN
    if 'New VAN' in df.columns:
        n_neg = (df['New VAN'] < 0).sum()
        if n_neg > 0:
            df['New VAN'] = df['New VAN'].clip(lower=0)
            print(f"  Clipped {n_neg} negative values in 'New VAN' to 0")

    # flag zero VAN
    if 'VAN' in df.columns:
        df['is_zero_van'] = (df['VAN'] == 0).astype(int)

    # categorize churn reasons
    df['churn_reason_group'] = df['Case Title'].apply(categorize_churn_reason)

    # handle nulls
    cat_fill = {'Risk': 'Unknown', 'Pull Type': 'None', 'Customer Tier': 'Unknown',
                'CompanySize': 'Unknown', 'Case Origin': 'Unknown', 'Branch': 'Unknown'}
    for col, fill_val in cat_fill.items():
        if col in df.columns and df[col].isnull().sum() > 0:
            null_pct = df[col].isnull().mean()
            if null_pct > 0.1:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(fill_val)

    # numeric nulls -> 0 for repair/overdue, median for others
    for col in ['Number Of Repair Cases', 'Number of OverdueServices']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"  Cleaning complete: {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_bob(df):
    """Clean the BoB dataset."""
    print_subheader("Cleaning BoB")

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {n_before - len(df):,} exact duplicates")

    n_before = len(df)
    df = df[df['system_status'] == 'Active'].copy()
    print(f"  Filtered to Active: {n_before:,} -> {len(df):,}")

    if 'company_sizing' in df.columns:
        df['company_sizing'] = df['company_sizing'].replace('', np.nan)

    print(f"  BoB cleaning complete: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── PHASE 3: FEATURE ENGINEERING ───────────────────────────
def engineer_features(df, bob):
    """Run the full feature engineering pipeline."""
    print_header("PHASE 3: FEATURE ENGINEERING")

    # time features
    print_subheader("Time features")
    for col in DATE_COLS:
        if col in df.columns:
            df[col + '_parsed'] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    if 'Case Creation Date_parsed' in df.columns and 'Expected Pull Date_parsed' in df.columns:
        df['days_to_pull'] = (df['Expected Pull Date_parsed'] - df['Case Creation Date_parsed']).dt.days

    if 'Case Creation Date_parsed' in df.columns and 'Registered Date_parsed' in df.columns:
        df['days_reg_to_case'] = (df['Case Creation Date_parsed'] - df['Registered Date_parsed']).dt.days

    if 'Case Creation Date_parsed' in df.columns and 'Agreement End Date_parsed' in df.columns:
        df['months_to_agreement_end'] = (
            (df['Agreement End Date_parsed'] - df['Case Creation Date_parsed']).dt.days / 30.44
        ).clip(lower=-120, upper=120)

    if 'Case Creation Date_parsed' in df.columns:
        df['case_month'] = df['Case Creation Date_parsed'].dt.month
        df['case_year'] = df['Case Creation Date_parsed'].dt.year
        df['case_dayofweek'] = df['Case Creation Date_parsed'].dt.dayofweek

    # categorical encoding
    print_subheader("Categorical encoding")
    if 'Case Type' in df.columns:
        df['is_cancellation'] = (df['Case Type'] == 'Cancellation').astype(int)
        df['is_risk_case'] = (df['Case Type'] == 'Risk').astype(int)

    if 'Pull Type' in df.columns:
        df['is_full_pull'] = (df['Pull Type'] == 'Full').astype(int)
        df['has_pull_type'] = df['Pull Type'].notna().astype(int)

    if 'Risk' in df.columns:
        risk_dummies = pd.get_dummies(df['Risk'], prefix='risk', dummy_na=False)
        df = pd.concat([df, risk_dummies], axis=1)

    if 'Case Origin' in df.columns:
        proactive = ['Account Manager', 'Proactive Prevention', 'Branch Manager',
                     'Service Manager', 'Customer Service Manager', 'Head of Customer Services']
        reactive = ['Notice in Writing', 'Customer Email', 'Customer Call',
                    'Customer Letter', 'Email', 'Phone', 'Fax', 'Web']
        df['origin_proactive'] = df['Case Origin'].isin(proactive).astype(int)
        df['origin_reactive'] = df['Case Origin'].isin(reactive).astype(int)

    if 'Customer Tier' in df.columns:
        df['tier_numeric'] = df['Customer Tier'].map(TIER_MAP)

    if 'churn_reason_group' in df.columns:
        reason_dummies = pd.get_dummies(df['churn_reason_group'], prefix='reason')
        df = pd.concat([df, reason_dummies], axis=1)

    # derived features
    print_subheader("Derived features")
    if 'VAN' in df.columns and 'Machines' in df.columns:
        df['van_per_machine'] = df['VAN'] / df['Machines'].replace(0, np.nan)

    if 'VAN' in df.columns and 'Number of Contracts' in df.columns:
        df['van_per_contract'] = df['VAN'] / df['Number of Contracts'].replace(0, np.nan)

    if 'Number Of Repair Cases' in df.columns:
        df['repair_rate'] = df['Number Of Repair Cases'].fillna(0)
        df['has_repairs'] = (df['repair_rate'] > 0).astype(int)

    if 'Number of OverdueServices' in df.columns:
        df['overdue_rate'] = df['Number of OverdueServices'].fillna(0)
        df['has_overdue'] = (df['overdue_rate'] > 0).astype(int)

    if 'Pull VAN' in df.columns and 'VAN' in df.columns:
        df['pull_van_ratio'] = df['Pull VAN'] / df['VAN'].replace(0, np.nan)

    # merge BoB features
    print_subheader("Merging BoB features")
    bob_agg = bob.groupby('account_number').agg(
        bob_total_revenue=('total_bob', 'sum'),
        bob_mean_revenue=('total_bob', 'mean'),
        bob_product_revenue=('product_bob', 'sum'),
        bob_fee_revenue=('fee_bob', 'sum'),
        bob_num_agreements=('agreement_number', 'nunique'),
        bob_num_products=('product_name', 'nunique'),
        bob_lines_of_business=('line_of_business', 'nunique'),
        bob_avg_service_interval=('service_interval', 'mean'),
        bob_avg_unit_amount=('unit_amount', 'mean'),
        bob_pct_auto_renewal=('renewal_type', lambda x: (x == 'Automatic Renewal').mean()),
        bob_has_machine_services=('line_of_business', lambda x: int('Machine Services' in x.values)),
        bob_has_auto_waste=('line_of_business', lambda x: int('Auto waste' in x.values)),
    ).reset_index()

    df = df.merge(bob_agg, left_on='Customer Account Number', right_on='account_number', how='left')
    df['has_bob_data'] = df['account_number'].notna().astype(int)
    print(f"  Rows with BoB data: {df['has_bob_data'].sum():,} ({df['has_bob_data'].mean():.1%})")

    # handle remaining nulls in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"\n  Feature engineering complete. Shape: {df.shape}")
    return df


# ── PHASE 4: FEATURE SELECTION ─────────────────────────────
def select_features(df):
    """Select features using RF importance and multicollinearity check."""
    print_header("PHASE 4: FEATURE SELECTION")

    # get valid feature columns
    date_parsed_cols = [c for c in df.columns if '_parsed' in c]
    raw_date_cols = DATE_COLS + ['Registered Time']
    exclude = set(EXCLUDE_FROM_MODELING + date_parsed_cols + raw_date_cols)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude and c != 'is_churned']
    print(f"  Total candidate features: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df['is_churned'].copy()

    # rank by RF importance
    print("  Computing RF importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("  Top 10 features:")
    for i, row in importance.head(10).iterrows():
        print(f"    {i + 1:2d}. {row['Feature']:40s} {row['Importance']:.4f}")

    # select above median
    threshold = importance['Importance'].median()
    selected = importance[importance['Importance'] >= threshold]['Feature'].tolist()
    print(f"\n  Selected {len(selected)} features (importance >= {threshold:.4f})")

    # remove multicollinear features
    print("  Checking multicollinearity (threshold=0.85)...")
    corr_matrix = X[selected].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if upper_tri.loc[idx, col] > 0.85:
                imp1 = importance[importance['Feature'] == idx]['Importance'].values
                imp2 = importance[importance['Feature'] == col]['Importance'].values
                imp1 = imp1[0] if len(imp1) > 0 else 0
                imp2 = imp2[0] if len(imp2) > 0 else 0
                drop = col if imp1 >= imp2 else idx
                to_drop.add(drop)

    if to_drop:
        selected = [f for f in selected if f not in to_drop]
        print(f"  Dropped {len(to_drop)} correlated features. Remaining: {len(selected)}")

    print(f"\n  Final feature set: {len(selected)} features")
    return selected, importance


# ── PHASE 5: MODEL TRAINING ────────────────────────────────
def train_models(df, selected_features):
    """Train all models and return results."""
    print_header("PHASE 5: MODEL TRAINING")

    X = df[selected_features].copy()
    y = df['is_churned'].copy()

    # fill any remaining NaN or inf
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # train/test split using sklearn directly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features, index=X_test.index)

    print(f"  Train: {X_train.shape[0]:,} samples (churn rate: {y_train.mean():.1%})")
    print(f"  Test:  {X_test.shape[0]:,} samples (churn rate: {y_test.mean():.1%})")

    # set up models
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                                                 random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'),
    }

    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss',
                                           scale_pos_weight=scale_pos)

    if LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                                              scale_pos_weight=scale_pos)

    # train each model
    results = {}
    for name, model in models.items():
        print(f"\n  Training: {name}...")

        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'AUC-PR': average_precision_score(y_test, y_proba),
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }

        print(f"    F1: {results[name]['F1-Score']:.4f}  |  AUC-ROC: {results[name]['AUC-ROC']:.4f}")

        # save model
        safe_name = name.lower().replace(' ', '_')
        model_path = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
        save_model(model, model_path, name)

    return results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ── PHASE 6: MODEL EVALUATION ──────────────────────────────
def evaluate_models(results):
    """Print model comparison table and find best model."""
    print_header("PHASE 6: MODEL EVALUATION")

    metrics_df = pd.DataFrame({
        name: {k: v for k, v in metrics.items() if k not in ['model', 'y_pred', 'y_proba']}
        for name, metrics in results.items()
    }).T.round(4)

    print("\n  Model Comparison:")
    print(metrics_df.to_string())

    best_name = max(results, key=lambda x: results[x]['F1-Score'])
    print(f"\n  Best model (by F1-Score): {best_name}")
    return best_name


# ── PHASE 7: BUSINESS OUTPUT ───────────────────────────────
def generate_business_output(df, results, best_name, selected_features, scaler):
    """Generate priority bands and risk table."""
    print_header("PHASE 7: BUSINESS OUTPUT")

    best_model = results[best_name]['model']

    # score all cases
    X_full = df[selected_features].copy().fillna(0).replace([np.inf, -np.inf], 0)
    if best_name == 'Logistic Regression':
        X_full = pd.DataFrame(scaler.transform(X_full), columns=selected_features, index=X_full.index)

    df['churn_probability'] = best_model.predict_proba(X_full)[:, 1]

    # assign priority bands
    def assign_band(prob):
        for band, (low, high) in PRIORITY_BANDS.items():
            if low <= prob < high:
                return band
        return 'P4 - Low'

    df['priority_band'] = df['churn_probability'].apply(assign_band)
    df['revenue_at_risk'] = df['VAN'] * df['churn_probability']

    # print summary
    print_subheader("Priority Band Summary")
    for band in PRIORITY_BANDS:
        mask = df['priority_band'] == band
        if mask.sum() > 0:
            count = mask.sum()
            rev = df.loc[mask, 'revenue_at_risk'].sum()
            avg_prob = df.loc[mask, 'churn_probability'].mean()
            actual = df.loc[mask, 'is_churned'].mean()
            print(f"  {band:<20} {count:>8,}    Rev@Risk: £{rev:>13,.0f}    Avg Prob: {avg_prob:.1%}    Actual: {actual:.1%}")

    # create risk table
    cols = ['Case ID', 'Customer Account Number', 'CompanySize', 'Customer Tier',
            'VAN', 'Machines', 'Number of Contracts', 'Case Type',
            'churn_reason_group', 'churn_probability', 'priority_band', 'revenue_at_risk', 'is_churned']
    available = [c for c in cols if c in df.columns]
    risk_table = df[available].copy()

    col_rename = {
        'Case ID': 'Case_ID', 'Customer Account Number': 'Account_Number',
        'CompanySize': 'Company_Size', 'Customer Tier': 'Customer_Tier',
        'Number of Contracts': 'Contracts', 'Case Type': 'Case_Type',
        'churn_reason_group': 'Churn_Reason', 'churn_probability': 'Churn_Probability',
        'priority_band': 'Priority_Band', 'revenue_at_risk': 'Revenue_at_Risk',
        'is_churned': 'Actual_Churned',
    }
    risk_table = risk_table.rename(columns=col_rename)

    # add recommended actions
    risk_table['Recommended_Action'] = risk_table['Churn_Reason'].map(
        {k: v['action'] for k, v in RETENTION_PLAYBOOK.items()}
    )

    # sort by priority
    priority_order = {'P1 - Critical': 0, 'P2 - High': 1, 'P3 - Medium': 2, 'P4 - Low': 3}
    risk_table['_sort'] = risk_table['Priority_Band'].map(priority_order)
    risk_table = risk_table.sort_values(['_sort', 'Revenue_at_Risk'], ascending=[True, False])
    risk_table = risk_table.drop('_sort', axis=1)

    # save risk table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'customer_risk_table.csv')
    risk_table.to_csv(out_path, index=False)
    print(f"\n  Risk table saved: {out_path} ({len(risk_table):,} rows)")

    return df, risk_table


# ── MAIN ────────────────────────────────────────────────────
def main():
    """Run the complete churn prediction pipeline."""

    # phase 1: load data
    retention, bob = load_data()

    # phase 2: clean data
    print_header("PHASE 2: DATA CLEANING")
    retention_clean = clean_retention(retention)
    bob_clean = clean_bob(bob)

    # save cleaned data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_dataframe(retention_clean, os.path.join(PROCESSED_DATA_DIR, 'retention_cleaned.csv'), 'cleaned retention')
    save_dataframe(bob_clean, os.path.join(PROCESSED_DATA_DIR, 'bob_cleaned.csv'), 'cleaned BoB')

    # phase 3: feature engineering
    df = engineer_features(retention_clean, bob_clean)
    save_dataframe(df, os.path.join(PROCESSED_DATA_DIR, 'df_engineered.csv'), 'engineered features')

    # phase 4: feature selection
    selected_features, importance = select_features(df)

    # phase 5: train models
    results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \
        train_models(df, selected_features)

    # phase 6: evaluate
    best_name = evaluate_models(results)

    # phase 7: business output
    df, risk_table = generate_business_output(df, results, best_name, selected_features, scaler)

    # summary
    print_header("PIPELINE COMPLETE")
    print(f"""
  DATA:
    Retention: {len(retention):,} rows -> {len(retention_clean):,} cleaned
    BoB: {len(bob):,} rows -> {len(bob_clean):,} cleaned
    Engineered: {df.shape}

  BEST MODEL: {best_name}
    F1-Score: {results[best_name]['F1-Score']:.4f}
    AUC-ROC:  {results[best_name]['AUC-ROC']:.4f}

  OUTPUTS:
    Models saved to:      {MODELS_DIR}
    Risk table saved to:  {OUTPUT_DIR}
    """)


if __name__ == '__main__':
    main()
