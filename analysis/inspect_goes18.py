import os
import polars as pl
from dotenv import load_dotenv

load_dotenv()

base_path = os.getenv("BASE_PATH")
file_path = os.path.join(base_path, 'datasets/raw/goes18cmipc/goes18_lst_samples_2024.csv')

def inspect_raw_samples(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # Polars read_csv is multithreaded and significantly faster for large datasets
    df = pl.read_csv(path)
    
    print(f"--- Inspecting: {os.path.abspath(path)} ---")

    print(f"\n1. Dimensions:")
    print(f"   Rows: {df.height}")
    print(f"   Columns: {df.width}")

    print(f"\n2 & 3. Schema and Data Types:")
    for col, dtype in df.schema.items():
        print(f"   - {col}: {dtype}")

    print(f"\n4. Unique Values Analysis:")
    for col in df.columns:
        n_unique = df[col].n_unique()
        # head(10) ensures we don't overwhelm the console with large unique sets
        sample_vals = df[col].unique().head(10).to_list()
        print(f"   - {col}: {n_unique} unique values")
        print(f"     Sample: {sample_vals}")

inspect_raw_samples(file_path)
