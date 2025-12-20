
import pandas as pd
import os

# Define path
file_path = 'datasets/raw/goes18acmc/goes18_lst_samples_2024_ACMC.csv'
absolute_path = os.path.abspath(file_path)

print(f"--- Inspecting: {absolute_path} ---")

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load data
    df = pd.read_csv(file_path)

    # 1. Number of rows and columns (Shape)
    rows, cols = df.shape
    print(f"\n1. Dimensions:")
    print(f"   Rows: {rows}")
    print(f"   Columns: {cols}")

    # 2. Column Names
    print(f"\n2. Column Names:")
    print(df.columns.tolist())

    # 3. Data Types
    print(f"\n3. Data Types:")
    print(df.dtypes)

    # 4. Unique Values Analysis
    print(f"\n4. Unique Values per Column:")
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"   - {col}: {n_unique} unique values")
        
        # Show sample unique values (top 10)
        unique_vals = df[col].unique()[:10]
        print(f"     Sample: {unique_vals}")
