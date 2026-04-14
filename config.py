import os

# project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# data paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# raw data files
RETENTION_FILE = os.path.join(RAW_DATA_DIR, 'Retention.csv')
BOB_FILE = os.path.join(RAW_DATA_DIR, 'BoB.xlsx')

# target variable
TARGET_COL = 'is_churned'
CHURN_STATUS = 'Customer Lost'

# column groups
DATE_COLS = ['Case Creation Date', 'Registered Date', 'Expected Pull Date', 'Agreement End Date']

NUMERIC_FEATURES = ['VAN', 'Pull VAN', 'New VAN', 'Number of Contracts', 'Machines',
                    'Number Of Repair Cases', 'Number of OverdueServices']

ID_COLS = ['Case ID', 'Customer Account Number', 'Customer Name']

EXCLUDE_FROM_MODELING = [
    'Case ID', 'Case Title', 'Customer Account Number', 'Customer Name',
    'Case Type', 'Pull Type', 'Risk', 'Case Origin', 'Customer Tier',
    'CompanySize', 'Branch', 'is_churned', 'account_number',
    'churn_reason_group', 'Resolution Status', 'Current Status',
    'Resolved Date', 'Resolved Time', 'Registered Time'
]

# CompanySize mapping (Excel date corruption fix)
COMPANY_SIZE_MAP = {
    '01-Sep': '1-9', 'Oct-19': '10-19', '20-49': '20-49',
    '50-99': '50-99', '100-249': '100-249', '250-499': '250-499',
    '500-999': '500-999', '>=1000': '1000+', '>1000': '1000+',
    '1000+': '1000+'
}

COMPANY_SIZE_NUMERIC_MAP = {
    '1-9': 5, '10-19': 15, '20-49': 35, '50-99': 75,
    '100-249': 175, '250-499': 375, '500-999': 750, '1000+': 1500
}

COMPANY_SIZE_ORDER = ['1-9', '10-19', '20-49', '50-99', '100-249', '250-499', '500-999', '1000+']

# Customer Tier mapping
TIER_MAP = {'Key Account': 1, 'Diamond': 2, 'Platinum': 3, 'Platinum+': 4, 'Platinum +': 4}

# priority band thresholds
PRIORITY_BANDS = {
    'P1 - Critical': (0.75, 1.01),
    'P2 - High': (0.50, 0.75),
    'P3 - Medium': (0.25, 0.50),
    'P4 - Low': (0.00, 0.25),
}

# churn reason keywords
CHURN_REASON_KEYWORDS = {
    'Site Closure': ['site closure', 'closed', 'closure'],
    'Price/Competition': ['competitor', 'value for money', 'tender', 'price'],
    'Service Issue': ['service', 'complaint'],
    'Machine Not Used': ['machine not used', 'nil use', 'not used', 'wrong machine'],
    'Debt': ['debt', 'credit'],
    'Contract Issue': ['contract', 'agreement'],
    'Access Issue': ['access denied', 'site access'],
}

# retention playbook
RETENTION_PLAYBOOK = {
    'Price/Competition': {'action': 'Match pricing / demonstrate ROI / offer discount', 'urgency': 'HIGH'},
    'Service Issue': {'action': 'Escalate to Service Manager, expedite next service visit', 'urgency': 'HIGH'},
    'Machine Not Used': {'action': 'Offer machine swap, downsizing, or temporary suspension', 'urgency': 'MEDIUM'},
    'Site Closure': {'action': 'Identify other customer sites, offer agreement transfer', 'urgency': 'MEDIUM'},
    'Debt': {'action': 'Involve Credit Controller, offer payment plan', 'urgency': 'HIGH'},
    'Contract Issue': {'action': 'Review contract terms, offer flexible renewal options', 'urgency': 'MEDIUM'},
    'Access Issue': {'action': 'Coordinate with site management for access scheduling', 'urgency': 'LOW'},
    'Other': {'action': 'Account Manager review, schedule customer meeting', 'urgency': 'MEDIUM'},
    'Unknown': {'action': 'Investigate case details, contact customer', 'urgency': 'MEDIUM'},
}

# model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
