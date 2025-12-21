import os
import polars as pl
from dotenv import load_dotenv

load_dotenv()

base_path = os.getenv("BASE_PATH")
output_dir = os.path.join(base_path, 'datasets/raw/goes18acmc')
csv_in_path = os.path.join(output_dir, 'goes18_lst_samples_2024_ACMC.csv')

def inspect_dataset_polars(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # Polars scan/read for efficiency
    df = pl.read_csv(path)
    
    print(f"--- Inspecting (Polars): {os.path.abspath(path)} ---")

    # 1. Dimensions
    print(f"\n1. Dimensions:")
    print(f"   Rows: {df.height}")
    print(f"   Columns: {df.width}")

    # 2. Column Names and 3. Data Types
    # Polars schema provides both name and type in an ordered dict
    print(f"\n2 & 3. Schema (Columns and Dtypes):")
    for col, dtype in df.schema.items():
        print(f"   - {col}: {dtype}")

    # 4. Unique Values Analysis
    print(f"\n4. Unique Values per Column:")
    for col in df.columns:
        n_unique = df[col].n_unique()
        # Polars handles unique values very efficiently
        unique_vals = df[col].unique().head(10).to_list()
        print(f"   - {col}: {n_unique} unique values")
        print(f"     Sample: {unique_vals}")

inspect_dataset_polars(csv_in_path)
