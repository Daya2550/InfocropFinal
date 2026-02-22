import pandas as pd
import os
import sys

def test_dataset_load():
    try:
        print("Starting dataset load test...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Base Directory: '{base_dir}'")
        
        csv_path = os.path.join(base_dir, "crop_price_dataset.csv")
        print(f"CSV Path: '{csv_path}'")
        
        if not os.path.exists(csv_path):
            print("ERROR: File does not exist!")
            return

        print("Attempting pd.read_csv...")
        df = pd.read_csv(csv_path)
        print("Success!")
        print(df.head())
        
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_load()
