import os
import pandas as pd

def load_all_iv_surfaces(folder_path, instruments=None):
    """
    Loads and combines IV surfaces from all CSVs in a folder (and subfolders),
    optionally filtered by ticker symbols extracted from filenames.

    Args:
        folder_path (str): Path to root folder with per-date subfolders of IV surfaces.
        instruments (list[str], optional): If provided, only load surfaces for these tickers.
                                           File names must follow {ticker}_{date}_iv_surface.csv.

    Returns:
        pd.DataFrame: Combined surface data with columns ['strike', 'maturity', 'iv', 'ticker', 'date']
    """
    all_records = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv') and "_iv_surface" in file:
                try:
                    parts = file.split("_")
                    if len(parts) < 3:
                        print(f"Invalid filename format: {file}")
                        continue
                    ticker, date_str = parts[0], parts[1]

                    # Check ticker filter only if instruments is provided
                    if instruments is not None and ticker not in instruments:
                        continue

                    filepath = os.path.join(root, file)
                    df = pd.read_csv(filepath)

                    df['ticker'] = ticker
                    df['date'] = date_str
                    all_records.append(df[['strike', 'maturity', 'iv', 'ticker', 'date']])
                except Exception as e:
                    print(f"Skipping {file}: {e}")
    print(f"Loaded {len(all_records)} files into dataframe")

    return pd.concat(all_records, ignore_index=True)