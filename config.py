"""
Central configuration for the forecasting pipeline.
All paths, thresholds, and model hyperparameters live here.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "artifacts" / "data"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
REPORT_DIR = BASE_DIR / "artifacts" / "reports"
FORECAST_DIR = BASE_DIR / "artifacts" / "forecasts"

WORKBOOK_NAME = "AI Forecasting Data.xlsx"
WORKBOOK_PATH_RAW = BASE_DIR / WORKBOOK_NAME          # raw xlsx at project root
WORKBOOK_PATH = BASE_DIR / "artifacts" / WORKBOOK_NAME  # uploaded/copied version

# ─── Sheet names ─────────────────────────────────────────────────────────────
SHEET_HEADER = "Header"
SHEET_SALE_LINES = "Sale Lines"
SHEET_ITEM_MASTER = "Item Master"

# ─── Required columns per sheet (raw names from workbook) ─────────────────────
REQUIRED_COLS_HEADER = [
    "Store No.", "Receipt No.", "Date", "Net Amount", "Gross Amount"
]
REQUIRED_COLS_SALE_LINES = [
    "Store No.", "Receipt No.", "Item No.", "Price", "Quantity",
    "Net Amount", "Trans. Date", "Unit of Measure"
]
REQUIRED_COLS_ITEM_MASTER = [
    "No.", "Description", "Item Category Code"
]

# ─── Column rename maps ───────────────────────────────────────────────────────
RENAME_SALE_LINES = {
    "Store No.": "store_id",
    "Receipt No.": "receipt_no",
    "Item No.": "item_id",
    "Price": "price",
    "Quantity": "quantity",
    "Net Price": "net_price",
    "Net Amount": "net_amount",
    "Trans. Date": "date",
    "Trans. Time": "trans_time",
    "Unit of Measure": "uom",
}
RENAME_HEADER = {
    "Store No.": "store_id",
    "Receipt No.": "receipt_no",
    "Date": "date",
    "Net Amount": "header_net_amount",
    "Gross Amount": "gross_amount",
    "Transaction Type": "transaction_type",
    "Nature of Supply": "nature_of_supply",
}
RENAME_ITEM_MASTER = {
    "No.": "item_id",
    "Description": "description",
    "Base Unit of Measure": "base_uom",
    "Item Category Code": "category",
    "Brand Name": "brand",
    "Division Code": "division",
    "Retail Product Code": "retail_product_code",
}

# ─── UOM normalisation map ────────────────────────────────────────────────────
UOM_NORMALISE = {
    "NOS": "NOS", "Nos": "NOS", "nos": "NOS",
    "KGS": "KGS", "Kgs": "KGS", "kgs": "KGS",
    "LTRS": "LTRS", "Ltrs": "LTRS", "ltrs": "LTRS",
}

# ─── Data quality / filtering ─────────────────────────────────────────────────
MIN_ITEM_DAYS = 10          # minimum distinct sales days to train item model
MIN_CATEGORY_DAYS = 10      # minimum distinct days for category model
SPARSE_FALLBACK_ORDER = ["item", "item_category_pooled", "category", "store_category", "store"]

# ─── Feature engineering ──────────────────────────────────────────────────────
LAG_DAYS = [1, 2, 3, 7, 14, 28]
ROLLING_WINDOWS = [3, 7, 14]

# ─── Train / validation splits ───────────────────────────────────────────────
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15          # test = remaining 15 %
WALK_FORWARD_STEP = 7        # expand window in 7-day increments
FORECAST_HORIZONS = [7, 14, 30]

# ─── Model hyperparameters ────────────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTICNET_L1_RATIOS = [0.2, 0.5, 0.8]
RANDOM_SEED = 42

# ─── Bootstrap uncertainty ───────────────────────────────────────────────────
N_BOOTSTRAP = 50
CI_LOWER = 10   # percentile
CI_UPPER = 90   # percentile

# ─── Column alias map for flat CSV/API/DB imports ───────────────────────────
COLUMN_ALIASES = {
    # Date variants
    "trans. date": "date", "transaction_date": "date", "trans_date": "date",
    "sale_date": "date", "order_date": "date",
    # Store variants
    "store no.": "store_id", "store": "store_id", "store_code": "store_id",
    "location_id": "store_id", "branch": "store_id",
    # Item variants
    "item no.": "item_id", "item": "item_id", "product_id": "item_id",
    "sku": "item_id", "article_no": "item_id",
    # Receipt
    "receipt no.": "receipt_no", "invoice_no": "receipt_no",
    "transaction_id": "receipt_no", "order_id": "receipt_no",
    # Quantity
    "qty": "quantity", "units": "quantity", "sold_qty": "quantity",
    # Amount
    "net amount": "net_amount", "amount": "net_amount", "revenue": "net_amount",
    "sales_amount": "net_amount", "total": "net_amount",
    # Category
    "item category code": "category", "item_category": "category",
    "product_category": "category", "dept": "category",
    # UOM
    "unit of measure": "uom", "unit": "uom",
    # Price
    "unit price": "price", "sell_price": "price",
}
REQUIRED_FLAT_COLS = ["date", "store_id", "item_id", "quantity", "net_amount"]
OPTIONAL_FLAT_COLS = ["receipt_no", "category", "uom", "price"]

# ─── Connector settings ─────────────────────────────────────────────────────
DB_QUERY_TIMEOUT = 30
DB_MAX_ROWS = 500_000
ALLOWED_DB_TYPES = ["postgresql", "mysql", "sqlite", "mssql"]
API_FETCH_TIMEOUT = 30
API_MAX_ROWS = 500_000
IMPORTED_FLAT_PATH = DATA_DIR / "imported_flat.csv"

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ensure artifact dirs exist
for _d in [DATA_DIR, MODEL_DIR, REPORT_DIR, FORECAST_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
