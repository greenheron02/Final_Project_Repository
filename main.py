import pandas as pd
import numpy as np

inputFile = 'used_cars_data.csv'
outputFile = 'used_cars_processed.csv'

colsToLoad = [
    'has_accidents',
    'height',
    'highway_fuel_economy',
    'horsepower',
    'interior_color',
    'isCab',
    'is_cpo',
    'is_oemcpo',
    'latitude',
    'longitude',
    'length',
    'listing_color',
    'make_name',
    'maximum_seating',
    'mileage',
    'model_name',
    'salvage',
    'savings_amount',
    'seller_rating',
    'theft_title',
    'torque',
    'transmission',
    'trim_name',
    'wheel_system',
    'wheelbase',
    'width',
    'year',
    'city_fuel_economy',
]


def loadData(filePath, columns):
    print(f"Attempting to load data from '{filePath}'...")
    try:
        csv_headers = pd.read_csv(filePath, nrows=0).columns.tolist()
        
        valid_cols = [col for col in columns if col in csv_headers]
        ignored_cols = [col for col in columns if col not in csv_headers]

        if ignored_cols:
            print(f"Warning: The following columns were not found in the CSV and will be ignored: {ignored_cols}")

        df = pd.read_csv(filePath, usecols=valid_cols)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filePath}'")
        print("Please make sure the file is in the same directory as the script.")
        return None
    except ValueError as e:
        print(f"ERROR: Column mismatch")
        print(f"A column in your list might not exist in the CSV file: {e}")
        return None

def processData(df):
    print("Processing data...")
    dfProcessed = df.copy()
    initialRows = len(dfProcessed)

    colsToDropna = ['height', 'length', 'trim_name']
    
    existingDropnaCols = [col for col in colsToDropna if col in dfProcessed.columns]
    
    dfProcessed.dropna(subset=existingDropnaCols, inplace=True)
    rowsDropped = initialRows - len(dfProcessed)
    print(f"Dropped {rowsDropped} rows with null values in {existingDropnaCols}.")

    boolColsToFill = [
        'has_accidents', 'isCab', 'is_cpo', 'is_oemcpo', 'salvage', 'theft_title'
    ]
    
    for col in boolColsToFill:
        if col in dfProcessed.columns:
            dfProcessed[col] = dfProcessed[col].astype(pd.BooleanDtype())
            dfProcessed[col] = dfProcessed[col].fillna(False)
            

    
    if 'city_fuel_economy' in dfProcessed.columns and 'highway_fuel_economy' in dfProcessed.columns:
        dfProcessed['avg_fuel_economy'] = dfProcessed[
            ['city_fuel_economy', 'highway_fuel_economy']
        ].mean(axis=1)
        print("Created 'avg_fuel_economy' from city and highway values.")

    colsToDrop = ['city_fuel_economy', 'highway_fuel_economy']
    
    dfProcessed.drop(columns=colsToDrop, inplace=True, errors='ignore')

    print(f"Final dataset has {len(dfProcessed)} rows and {len(dfProcessed.columns)} columns.")
    
    return dfProcessed

def main():
    dfRaw = loadData(inputFile, colsToLoad)

    if dfRaw is not None:
        dfProcessed = processData(dfRaw)

        try:
            dfProcessed.to_csv(outputFile, index=False)
            print(f"\nSuccessfully saved processed data to '{outputFile}'")
        except Exception as e:
            print(f"\nERROR: Could not save file")
            print(e)

if __name__ == "__main__":
    main()