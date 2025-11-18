import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.plotting
from sklearn.linear_model import LinearRegression 
# --- Configuration ---
inputFile = 'Final_Project/used_cars_data.csv'
useSample = True
sampleSize = 1000

colsToLoad = [
    'has_accidents', 'height', 'highway_fuel_economy', 'horsepower',
    'interior_color', 'isCab', 'is_cpo', 'is_oemcpo', 'latitude',
    'longitude', 'length', 'listing_color', 'make_name', 'maximum_seating',
    'mileage', 'model_name', 'salvage', 'savings_amount', 'seller_rating',
    'theft_title', 'torque', 'transmission', 'trim_name', 'wheel_system',
    'wheelbase', 'width', 'year', 'price', 'city_fuel_economy', 'body_type'
]


def loadData(filePath, columns):
    """
    Loads the raw CSV data, printing status.
    """
    print(f"Attempting to load data from '{filePath}'...")
    try:
        csvHeaders = pd.read_csv(filePath, nrows=0).columns.tolist()
        validCols = [col for col in columns if col in csvHeaders]
        ignoredCols = [col for col in columns if col not in csvHeaders]

        if ignoredCols:
            print(f"Warning: The following columns were not found in the CSV and will be ignored: {ignoredCols}")

        df = pd.read_csv(filePath, usecols=validCols)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        print(f"--- ERROR: File not found at '{filePath}' ---")
        return None
    except ValueError as e:
        print(f"--- ERROR: Column mismatch ---")
        return None

def cleanAndFeatureEngineer(df):
    # --- 1. PREPROCESSING AND FEATURE ENGINEERING ---
    print("Preprocessing data...")
    dfProcessed = df.copy()

    colsToCleanNumeric = [
        'height', 'length', 'wheelbase', 'width', 'torque', 'horsepower',
        'maximum_seating', 'mileage', 'savings_amount', 'seller_rating',
        'latitude', 'longitude', 'price'
    ]
    for col in colsToCleanNumeric:
        if col in dfProcessed.columns:
            dfProcessed[col] = dfProcessed[col].astype(str)
            dfProcessed[col] = dfProcessed[col].str.replace(r"[^0-9\.]", "", regex=True)
            dfProcessed[col] = pd.to_numeric(dfProcessed[col], errors='coerce')

    colsToDropna = ['height', 'length', 'trim_name', 'price']
    existingDropnaCols = [col for col in colsToDropna if col in dfProcessed.columns]
    dfProcessed.dropna(subset=existingDropnaCols, inplace=True)

    boolColsToFill = [
        'has_accidents', 'isCab', 'is_cpo', 'is_oemcpo', 'salvage', 'theft_title'
    ]
    for col in boolColsToFill:
        if col in dfProcessed.columns:
            dfProcessed[col] = dfProcessed[col].astype(pd.BooleanDtype())
            dfProcessed[col] = dfProcessed[col].fillna(False)
            
    currentYear = datetime.datetime.now().year
    if 'year' in dfProcessed.columns:
        dfProcessed['Car_Age'] = currentYear - dfProcessed['year']

    if 'city_fuel_economy' in dfProcessed.columns and 'highway_fuel_economy' in dfProcessed.columns:
        dfProcessed['avg_fuel_economy'] = dfProcessed[
            ['city_fuel_economy', 'highway_fuel_economy']
        ].mean(axis=1)

    colsToDrop = ['city_fuel_economy', 'highway_fuel_economy', 'year']
    dfProcessed.drop(columns=colsToDrop, inplace=True, errors='ignore')

    print(f"...Preprocessing and feature engineering complete. {len(dfProcessed)} rows remaining.")
    return dfProcessed

def trainRegressor(df,reg):
    # --- 2. RANDOM FOREST MODEL ---

    reg_name = type(reg).__name__
    
    targetCol = 'price'
    X = df.drop(targetCol, axis=1)
    y = df[targetCol]
    
    numericalCols = [
        'latitude', 'longitude', 'mileage', 'Car_Age', 'horsepower', 'torque',
        'height', 'length', 'wheelbase', 'width', 'maximum_seating',
        'seller_rating', 'savings_amount', 'avg_fuel_economy'
    ]
    categoricalCols = [
        'make_name', 'model_name', 'trim_name', 'body_type', 'transmission',
        'wheel_system', 'fuel_type', 'listing_color', 'interior_color'
    ]
    booleanCols = [
        'has_accidents', 'isCab', 'is_cpo', 'is_oemcpo', 'salvage', 'theft_title'
    ]
    
    numericalCols = [col for col in numericalCols if col in X.columns]
    categoricalCols = [col for col in categoricalCols if col in X.columns]
    booleanCols = [col for col in booleanCols if col in X.columns]

    numericTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categoricalTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,
            min_frequency=0.01  
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numericTransformer, numericalCols),
            ('cat', categoricalTransformer, categoricalCols),
            ('bool', 'passthrough', booleanCols)
        ],
        remainder='drop' 
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n--- Training Random Forest model on {len(xTrain)} rows... ---")
    model.fit(xTrain, yTrain)

    print("Evaluating Random Forest model...")
    yPred = model.predict(xTest)
    
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    r2 = r2_score(yTest, yPred)
    
    print(f"\n--- Model Performance ({reg_name}) ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    plotData = pd.DataFrame({'Actual Price': yTest, 'Predicted Price': yPred})
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='Actual Price', y='Predicted Price', data=plotData, alpha=0.5)
    maxPrice = max(yTest.max(), yPred.max())
    plt.plot([0, maxPrice], [0, maxPrice], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual Price vs. Predicted Price ({reg_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{reg_name}_actual_vs_predicted.png')

    print(f"\n--- Example Predictions ({reg_name}) ---")
    exampleData = plotData.head(10).copy()
    exampleData['Error ($)'] = exampleData['Predicted Price'] - exampleData['Actual Price']
    exampleData = exampleData.round(2)
    print(exampleData.to_string())
    
    fig, ax = plt.subplots(figsize=(10, 3)) 
    ax.axis('tight')
    ax.axis('off')
    table = pd.plotting.table(ax, exampleData, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) 
    plt.title(f'Example Predictions ({reg_name})', fontsize=16)
    plt.savefig(f'{reg_name}_example_predictions.png', bbox_inches='tight', dpi=150)
    print(f"...{reg_name} plots saved.")
    
    return rmse, r2 

def trainRandomForest(df):
    # --- 2. RANDOM FOREST MODEL ---
    
    targetCol = 'price'
    X = df.drop(targetCol, axis=1)
    y = df[targetCol]
    
    numericalCols = [
        'latitude', 'longitude', 'mileage', 'Car_Age', 'horsepower', 'torque',
        'height', 'length', 'wheelbase', 'width', 'maximum_seating',
        'seller_rating', 'savings_amount', 'avg_fuel_economy'
    ]
    categoricalCols = [
        'make_name', 'model_name', 'trim_name', 'body_type', 'transmission',
        'wheel_system', 'fuel_type', 'listing_color', 'interior_color'
    ]
    booleanCols = [
        'has_accidents', 'isCab', 'is_cpo', 'is_oemcpo', 'salvage', 'theft_title'
    ]
    
    numericalCols = [col for col in numericalCols if col in X.columns]
    categoricalCols = [col for col in categoricalCols if col in X.columns]
    booleanCols = [col for col in booleanCols if col in X.columns]

    numericTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categoricalTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,
            min_frequency=0.01  
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numericTransformer, numericalCols),
            ('cat', categoricalTransformer, categoricalCols),
            ('bool', 'passthrough', booleanCols)
        ],
        remainder='drop' 
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n--- Training Random Forest model on {len(xTrain)} rows... ---")
    model.fit(xTrain, yTrain)

    print("Evaluating Random Forest model...")
    yPred = model.predict(xTest)
    
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    r2 = r2_score(yTest, yPred)
    
    print("\n--- Model Performance (Random Forest) ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    regressor = model.named_steps['regressor']
    importances = regressor.feature_importances_
    featureNames = model.named_steps['preprocessor'].get_feature_names_out()
    
    cleanFeatureNames = []
    for name in featureNames:
        name = name.replace('num__', '').replace('cat__', '').replace('bool__', '')
        name = name.replace('_infrequent_sklearn', ' (Other)')
        name = name.replace('_', ' ')
        name = name.capitalize()
        cleanFeatureNames.append(name)
    
    importanceData = pd.DataFrame({
        'feature': cleanFeatureNames,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importanceData.head(20))
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')

    plotData = pd.DataFrame({'Actual Price': yTest, 'Predicted Price': yPred})
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='Actual Price', y='Predicted Price', data=plotData, alpha=0.5)
    maxPrice = max(yTest.max(), yPred.max())
    plt.plot([0, maxPrice], [0, maxPrice], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title('Actual Price vs. Predicted Price (Random Forest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rf_actual_vs_predicted.png')

    print("\n--- Example Predictions (Random Forest) ---")
    exampleData = plotData.head(10).copy()
    exampleData['Error ($)'] = exampleData['Predicted Price'] - exampleData['Actual Price']
    exampleData = exampleData.round(2)
    print(exampleData.to_string())
    
    fig, ax = plt.subplots(figsize=(10, 3)) 
    ax.axis('tight')
    ax.axis('off')
    table = pd.plotting.table(ax, exampleData, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) 
    plt.title('Example Predictions (Random Forest)', fontsize=16)
    plt.savefig('rf_example_predictions.png', bbox_inches='tight', dpi=150)
    print("...Random Forest plots saved.")
    
    return rmse, r2 

def trainNearestNeighbor(df):
    # --- 3. K-NEAREST NEIGHBORS MODEL ---
    
    targetCol = 'price'
    X = df.drop(targetCol, axis=1)
    y = df[targetCol]
    
    numericalCols = [
        'latitude', 'longitude', 'mileage', 'Car_Age', 'horsepower', 'torque',
        'height', 'length', 'wheelbase', 'width', 'maximum_seating',
        'seller_rating', 'savings_amount', 'avg_fuel_economy'
    ]
    categoricalCols = [
        'make_name', 'model_name', 'trim_name', 'body_type', 'transmission',
        'wheel_system', 'fuel_type', 'listing_color', 'interior_color'
    ]
    booleanCols = [
        'has_accidents', 'isCab', 'is_cpo', 'is_oemcpo', 'salvage', 'theft_title'
    ]
    
    numericalCols = [col for col in numericalCols if col in X.columns]
    categoricalCols = [col for col in categoricalCols if col in X.columns]
    booleanCols = [col for col in booleanCols if col in X.columns]

    numericTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categoricalTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,
            min_frequency=0.01 
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numericTransformer, numericalCols),
            ('cat', categoricalTransformer, categoricalCols),
            ('bool', 'passthrough', booleanCols)
        ],
        remainder='drop' 
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(
            n_neighbors=5,
            n_jobs=-1
        ))
    ])

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n--- Training K-Nearest Neighbors model on {len(xTrain)} rows... ---")
    model.fit(xTrain, yTrain)

    print("Evaluating model... (This may be slow for KNN)")
    yPred = model.predict(xTest)
    
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    r2 = r2_score(yTest, yPred)
    
    print("\n--- Model Performance (K-Nearest Neighbors) ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    plotData = pd.DataFrame({'Actual Price': yTest, 'Predicted Price': yPred})
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='Actual Price', y='Predicted Price', data=plotData, alpha=0.5)
    maxPrice = max(yTest.max(), yPred.max())
    plt.plot([0, maxPrice], [0, maxPrice], color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title('Actual Price vs. Predicted Price (K-Nearest Neighbors)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('knn_actual_vs_predicted.png')

    print("\n--- Example Predictions (K-Nearest Neighbors) ---")
    exampleData = plotData.head(10).copy()
    exampleData['Error ($)'] = exampleData['Predicted Price'] - exampleData['Actual Price']
    exampleData = exampleData.round(2)
    print(exampleData.to_string())

    fig, ax = plt.subplots(figsize=(10, 3)) 
    ax.axis('tight')
    ax.axis('off')
    table = pd.plotting.table(ax, exampleData, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) 
    plt.title('Example Predictions (K-Nearest Neighbors)', fontsize=16)
    plt.savefig('knn_example_predictions.png', bbox_inches='tight', dpi=150)
    print("...K-Nearest Neighbors plots saved.")
    
    return rmse, r2 

def createComparisonPlot(modelNames,RMSEs,R2s):
    """
    Creates a bar chart comparing the performance of two or more models.

    """
    print("\n--- Creating Model Comparison Plot ---")

    if len(RMSEs) <= 1 or len(R2s) <= 1 or len(modelNames) <= 1: 
        raise ValueError('createComparisonPlot() must have 2 arrays of len 2 or greater')


    data = {
        'Model': modelNames,
        'RMSE ($)': RMSEs,
        'R-Squared (R²)': R2s
    }
    
    # data = {
    #     'Model': ['Random Forest', 'K-Nearest Neighbors'],
    #     'RMSE ($)': [rfRMSE, knnRMSE],
    #     'R-Squared (R²)': [rfR2, knnR2]
    # }
    dfMetrics = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.barplot(x='Model', y='RMSE ($)', data=dfMetrics, ax=ax1, palette='Reds_r')
    ax1.set_title('Model Comparison: RMSE (Lower is Better)')
    ax1.set_ylabel('Root Mean Squared Error ($)')
    ax1.set_xlabel('Model')
    for p in ax1.patches:
        ax1.annotate(f"${p.get_height():,.2f}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     xytext=(0, 10), textcoords='offset points')

    sns.barplot(x='Model', y='R-Squared (R²)', data=dfMetrics, ax=ax2, palette='Greens_r')
    ax2.set_title('Model Comparison: R-Squared (Higher is Better)')
    ax2.set_ylabel('R-Squared (R²) Score')
    ax2.set_xlabel('Model')
    ax2.set_ylim(0, 1.0) 
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     xytext=(0, 10), textcoords='offset points')
    
    fig.suptitle('Regression Model Performance Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model_comparison.png')
    
    print("...Model comparison plot saved as 'model_comparison.png'")

def createCorrelationMatrices(df:pd.DataFrame):
    columns = [col for col in df.columns if (df[col].dtype == float or df[col].dtype == int)]
    columns.remove('price')

# Copy car_data into dataframe for processing
    corr_data = df.copy()

    col1 = columns[0:int(len(columns)/2)]
    col2 = columns[int(len(columns)/2):len(columns)]

    # Make sure price data is compared against
    col1.insert(0,'price')
    col2.insert(0,'price')

    # Get correlation values for each subarray
    matrix1 = corr_data[col1].corr()
    matrix2  = corr_data[col2].corr() 

    # Plot correlation matrices
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(matrix1,annot=True,cmap="coolwarm", fmt=".2f", linewidths=0.5) 
    plt.title("Correlation Heatmap 1")
    plt.savefig('corr_matrix1.png')
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(matrix2,annot=True,cmap="coolwarm", fmt=".2f", linewidths=0.5) 
    plt.title("Correlation Heatmap 2")
    plt.savefig('cor_matrix_2.png')

def createAnovaPlot(df,target,d_type):
    from scipy.stats import f_oneway 

    predictor_list = [col for col in df.columns if (isinstance(df[col].iloc[0],d_type))]
    output = [] 
    for predictor in predictor_list: 
        CategoryGroupLists=df.groupby(predictor)[target].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        output.append(AnovaResults[1])
    
    plt.figure(figsize=(12,5))
    plt.bar(predictor_list,output)
    if max(output) > 0.01: 
        plt.axhline(y=0.05,color='r',label='Maximum p value: 0.05')
    plt.xticks(rotation=30)
    plt.title(f'One-Way ANOVA analysis for fields of type: {d_type.__name__} ')
    plt.legend()
    plt.savefig(f'anova_{d_type.__name__}.png')


def main():
    dfRaw = loadData(inputFile, colsToLoad)
    
    if dfRaw is not None:
        
        dfClean = cleanAndFeatureEngineer(dfRaw)
        
        if useSample:
            print(f"--- Running on a sample of {sampleSize} rows ---")
            if len(dfClean) > sampleSize:
                dfRun = dfClean.sample(n=sampleSize, random_state=42)
            else:
                print(f"Warning: Dataset length ({len(dfClean)}) is less than sample size. Using full dataset.")
                dfRun = dfClean
        else:
            print("--- Running on the full dataset ---")
            dfRun = dfClean
        
        dfForSVR = dfRun.copy() 
        svrRMSE, svrR2 = trainRegressor(dfForSVR,LinearRegression())
        
        dfForRF = dfRun.copy()
        rfRMSE, rfR2 = trainRandomForest(dfForRF)
        
        dfForKNN = dfRun.copy()
        knnRMSE, knnR2 = trainNearestNeighbor(dfForKNN)
        
        createCorrelationMatrices(dfRun)
        
        createAnovaPlot(dfRun,'price',np.bool)
        createAnovaPlot(dfRun,'price',str)

        createComparisonPlot(['Random Forest', 'K-Nearest Neighbor','Linear Regressor'],
                             [rfRMSE,knnRMSE,svrRMSE],
                             [rfR2,knnR2,svrR2])
        
        print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
