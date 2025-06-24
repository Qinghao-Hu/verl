#!/usr/bin/env python3
"""
Script to inspect the training data from the parquet file
"""

import sys

import pandas as pd

# File path from the shell script
data_path = "/nobackup/qinghao/dataset/reasoning/gsm8k/train.parquet"

def inspect_parquet_file(file_path):
    """Read and display information about the parquet file"""
    
    try:
        # Read the parquet file
        print(f"Reading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Basic information
        print("\n" + "="*60)
        print("BASIC INFORMATION")
        print("="*60)
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        
        # Memory usage
        print(f"\nMemory usage:\n{df.memory_usage(deep=True)}")
        
        # First few rows
        print("\n" + "="*60)
        print("FIRST 5 ROWS")
        print("="*60)
        print(df.head())
        
        # Sample data for each column
        print("\n" + "="*60)
        print("SAMPLE DATA FROM EACH COLUMN")
        print("="*60)
        for col in df.columns:
            print(f"\n--- Column: {col} ---")
            # Show first non-null value
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            print(f"First value: {first_val}")
            if isinstance(first_val, str) and len(first_val) > 200:
                print(f"(Truncated, full length: {len(first_val)} characters)")
        
        # Statistics for numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            print("\n" + "="*60)
            print("NUMERICAL STATISTICS")
            print("="*60)
            print(df[numerical_cols].describe())
        
        # Check for missing values
        print("\n" + "="*60)
        print("MISSING VALUES")
        print("="*60)
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("Columns with missing values:")
            print(missing[missing > 0])
        else:
            print("No missing values found")
        
        # Show a random sample
        print("\n" + "="*60)
        print("RANDOM SAMPLE (3 rows)")
        print("="*60)
        if len(df) >= 3:
            print(df.sample(n=3, random_state=42))
        else:
            print(df)
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    inspect_parquet_file(data_path) 