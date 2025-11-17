# %%
## Used Car Price Predictor - CS488 Final Project
# 
# Enrico Addy, Jesus Rozas 
# Last Edit 11/17/2025 
# 
# 
# This program uses six statistical regression techniques to 
# predict the value of a used car based upon the prices and 
# details of a used car listing.  
# 
# This program was made using a used car sales databse listed on Kaggle, 
# which can be accessed here: 
# https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset?resource=download
#
# This program was made with a series of references which can be viewed below: 
# https://thinkingneuron.com/diamond-price-prediction-case-study-in-python
# https://amanxai.com/2022/09/26/diamond-price-analysis-using-python
# https://www.kaggle.com/code/girindradaafimada/sleman-group-spark-analytics-big-data-cs 
# 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.impute import SimpleImputer 

# Define an imputer for preprocessing purposes
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# helper function: get rid of additional string (gal) in numerical data 
def replace_generic(dataset:pd.Series): 
    data = dataset.copy()
    for i, val in enumerate(data): 
        if isinstance(val,str): 
                data.iloc[i] = float(val.split(' ')[0])
    return data

# Impute values into numerical columns, or replace null values with random 
# reasonable values
def apply_imputer(series,series_name,imputer:SimpleImputer): 
    data_df = pd.DataFrame(series,columns=[series_name])
    data_df = imputer.fit_transform(data_df).round(1)
    serie = pd.Series(data_df.ravel())
    return serie

# Early helper function, same as DataFrame.unique()    
def check_values(filename,row,data_name:str): 
    bar_data = pd.read_csv(filename,nrows=row,usecols=[data_name])
    diffs = [] 
    for i in bar_data[data_name]: 
        if not diffs.__contains__(i): 
            diffs.append(i)
    print(diffs)
    return diffs
    
# Replace classes ('V8 Engine') with numbers (1) in dataset
# This is for regression to act upon later
def mapcols(dataset:pd.DataFrame,column_labels):
    for col in column_labels: 
        # Create a dict with class names as keys and numbers as values ('V8 Engine':1)
        t_dict = dict(zip(dataset[col].unique(),range(1,len(dataset[col])+1)))
        # Use dict to replace class names in column with numbers via map()
        dataset.loc[:,col] = dataset[col].map(t_dict)
    return dataset

# %%
## Load used_cars_data 
import os 

try:
    filename = 'used_cars_data.csv'
    n_rows = 10000
    car_data = pd.read_csv(filename,nrows=n_rows)
except FileNotFoundError: 
    os.chdir('Final_Project')
    car_data = pd.read_csv(filename,nrows=n_rows)

# %%
## Preprocessing all necessary columns
colsToLoad = [
    'back_legroom',
    'bed', 
    'bed_length', 
    'body_type', 
    'cabin', 
    'city', 
    'description', 
    'engine_type',
    'engine_displacement',
    'exterior_color',
    'fleet',
    'frame_damaged',
    'front_legroom',
    'fuel_tank_volume',
    'fuel_type',
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
    'price',
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

# Apply all preprocessing steps to dataset and strip unnecessary columns 
# Delete null rows 
# Impute values to fill rows 
# Set null values to constant where needed 
# etc. 
def processData(df:pd.DataFrame) -> pd.DataFrame:
    # Create dataset to operate upon
    print("Processing data...")
    dfProcessed = df.copy()
    initialRows = len(dfProcessed)

    # Drop rows from columns with vital information
    colsToDropna = ['trim_name']

    existingDropnaCols = [col for col in colsToDropna if col in dfProcessed.columns]

    dfProcessed.dropna(subset=existingDropnaCols, inplace=True)
    rowsDropped = initialRows - len(dfProcessed)
    print(f"Dropped {rowsDropped} rows with null values in {existingDropnaCols}.")

    # Order columns of dataframe (df) into lists based upon datatype
    boolColsToFill = [
        'fleet', 'frame_damaged', 'has_accidents', 'isCab', 'is_cpo', 'theft_title',
        'is_oemcpo', 'salvage'
    ]

    numColsToFill = [
            'avg_fuel_economy','back_legroom','front_legroom','fuel_tank_volume','engine_displacement',
            'height','horsepower','latitude','longitude','length','maximum_seating','mileage',
            'savings_amount','seller_rating','torque','wheelbase','width','year',
    ]

    zeroColsToFill = [
        'bed_length'
    ]

    strColsToFill = [
        'bed','body_type','cabin','city','description','engine_type','exterior_color','fuel_type',
        'interior_color','listing_color','make_name','model_name','transmission','trim_name',
        'wheel_system',
    ]

    # Merge city_fuel_economy and highway_fuel_economy into avg_fuel_economy
    if 'city_fuel_economy' in dfProcessed.columns and 'highway_fuel_economy' in dfProcessed.columns:
        dfProcessed['avg_fuel_economy'] = dfProcessed[
            ['city_fuel_economy', 'highway_fuel_economy']
        ].mean(axis=1)
    print("Created 'avg_fuel_economy' from city and highway values.")

    # Fill null values in boolean columns with False
    for col in boolColsToFill:
        if col in dfProcessed.columns:
            dfProcessed.loc[:,col] = dfProcessed[col].astype(pd.BooleanDtype())
            dfProcessed.loc[:,col] = dfProcessed[col].fillna(False)

    # Fill null values in string columns with 'None'
    for col in strColsToFill: 
        if col in dfProcessed.columns: 
            dfProcessed.loc[:,col] = dfProcessed[col].fillna('None')

    # Fill null values in numerical columns with imputed values
    for col in numColsToFill: 
        if col in dfProcessed.columns: 
            dtype = dfProcessed[col].dtype

            # If column data does not contain additional string i.e. 4.5 in
            if dtype == float or dtype == int: 

                # only apply the imputer
                series = apply_imputer(dfProcessed[col],col,imp)
                dfProcessed.loc[:,col] = series.values
            else: 

                # otherwise strip the additional string and apply imputer
                data = dfProcessed.loc[:,col]
                data = data.str.replace('--','nan')
                data = replace_generic(data).astype(float)
                series = apply_imputer(data,col,imp)
                dfProcessed.loc[:,col] = series.values 
    
    # Fill null values in certain columns with zero
    for col in zeroColsToFill: 
        if col in dfProcessed.columns: 
            data = dfProcessed.loc[:,col].copy()
            data = data.str.replace('--','nan')
            data = replace_generic(data).astype(float)
            data = data.fillna(0.0)
            dfProcessed.loc[:,col] = data.values

    # Remove city_fuel_economy and highway_fuel_economy after merging into avg
    colsToDrop = ['city_fuel_economy', 'highway_fuel_economy']

    dfProcessed.drop(columns=colsToDrop, inplace=True, errors='ignore')

    print(f"Final dataset has {len(dfProcessed)} rows and {len(dfProcessed.columns)} columns.")

    # Return processed dataframe
    return dfProcessed

pd.set_option("future.no_silent_downcasting",True)
car_data = processData(car_data[colsToLoad])



# %%
## Create a correlation matrix to observe relation to price 
# Values below abs(0.5) should be dropped due to little relation 
# but as you'll see that applies to most numerical values in the dataset
 

columns = [
            'price','avg_fuel_economy','bed_length','back_legroom','front_legroom','fuel_tank_volume','engine_displacement',
            'height','horsepower','latitude','longitude','length','maximum_seating','mileage',
            'savings_amount','seller_rating','torque','wheelbase','width','year'
]

# Copy car_data into dataframe for processing
corr_data = car_data.copy()
# for col in columns: 
#     dat = corr_data[col]
#     data = pd.DataFrame(dat,columns=[col]) 
#     scaled_data = scaler_model.fit_transform(data)
#     corr_data[col] = scaled_data


# matrix = corr_data[columns].corr() 
# plt.figure(figsize=(8,6))
# sns.heatmap(matrix,annot=True,cmap="coolwarm", fmt=".2f", linewidths=0.5) 
# plt.title("Correlation Heatmap")

# Split columns array into two subarrays; the original array is too long
# for one graph
col1 = columns[0:int(len(columns)/2)]
print(col1)
col2 = columns[int(len(columns)/2):len(columns)]
print(col2)

# Make sure price data is compared against
col2.insert(0,'price')

# Get correlation values for each subarray
matrix1 = corr_data[col1].corr()
matrix2  = corr_data[col2].corr() 

# Plot correlation matrices
plt.figure(figsize=(8,6))
sns.heatmap(matrix1,annot=True,cmap="coolwarm", fmt=".2f", linewidths=0.5) 
plt.title("Correlation Heatmap")
plt.figure(figsize=(8,6))
sns.heatmap(matrix2,annot=True,cmap="coolwarm", fmt=".2f", linewidths=0.5) 
plt.title("Correlation Heatmap")





# %%
# collect column headers of car_data into subarrays for processing

strColsToFill = [
        'bed','body_type','cabin','city','engine_type','exterior_color','fuel_type',
        'interior_color','listing_color','make_name','model_name','transmission','trim_name',
        'wheel_system',
    ]

boolColsToFill = [
        'fleet', 'frame_damaged', 'has_accidents', 'isCab', 'is_cpo', 'theft_title',
        'is_oemcpo', 'salvage'
    ]

numColsToFill = [
            'avg_fuel_economy','back_legroom','bed_length','front_legroom','fuel_tank_volume','engine_displacement',
            'height','horsepower','latitude','longitude','length','maximum_seating','mileage',
            'savings_amount','seller_rating','torque','wheelbase','width','year',
    ]

# Combine subarrays to reflect all data
columns = strColsToFill + boolColsToFill + numColsToFill


## Calculate the one-way Anova values 
# This is a method for discovering correlation between categorical 
# data (predictor_list) and continuous data (target)
# A value below 0.05 reflect correlation
def AnovaFunc(inputdata,target,predictor_list):
    from scipy.stats import f_oneway 

    print('ANOVA RESULTS')
    output = [] 
    for predictor in predictor_list: 
        CategoryGroupLists=inputdata.groupby(predictor)[target].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        output.append(AnovaResults[1])
    return output

# Split column list into chunks for box plotting
def chunk_list(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


## Create box plots to evaluate success of correlation 
# The more consistent the lines above, the better

# clist = chunk_list(columns,3)
# long = []
# short = []
# for chunk in clist: 
#     long.extend([c for c in chunk if len(car_data[c].unique()) > 3])
#     short.extend([c for c in chunk if c not in long])

# cols = short
# fig, PlotCanvas = plt.subplots(nrows=1,ncols=len(cols),figsize=(24,5))
# for pcol, i in zip(cols, range(len(cols))):
#     car_data.boxplot(column='price',by=pcol,figsize=(5,5),vert=True,ax=PlotCanvas[i])

# cols = long
# figs = [0]*len(long)
# axes = [0]*len(long)
# for pcol, i in zip(cols, range(len(cols))):
#     figs[i], axes[i] = plt.subplots(figsize=(18,5))
#     car_data.boxplot(column='price',by=pcol,figsize=(18,5),vert=True,ax=axes[i])
#     for tick in axes[i].get_xticklabels(): 
#         tick.set_rotation(45)


## Create anova plots to evaluate validity for correlation 
# Strip rows that have an Anova value above 0.05
# Note: this only works for categorical data, not numerical
# take it from me you do not want to have that many classes

anova_results = AnovaFunc(car_data,'price',boolColsToFill)

plt.figure(figsize=(12,5))
plt.bar(boolColsToFill,anova_results)
plt.axhline(y=0.05)
plt.title('One-Way ANOVA analysis for boolean fields')


anova_results = AnovaFunc(car_data,'price',strColsToFill)

plt.figure(figsize=(12,5))
plt.bar(strColsToFill,anova_results)
plt.axhline(y=0.05)
plt.xticks(rotation=45)
plt.title('One-Way ANOVA analysis for string fields');


# %%
strColsToFill = [
        'body_type','cabin','city','engine_type','exterior_color','fuel_type',
        'interior_color','listing_color','make_name','model_name','transmission','trim_name',
        'wheel_system',
    ]

boolColsToFill = [
        'fleet', 'frame_damaged', 'has_accidents', 'isCab', 'is_cpo', 
        'is_oemcpo', 'salvage'
    ]

numColsToFill = [
            'avg_fuel_economy','back_legroom','bed_length','front_legroom','fuel_tank_volume','engine_displacement',
            'height','horsepower','latitude','longitude','length','maximum_seating','mileage',
            'savings_amount','seller_rating','torque','wheelbase','width','year',
    ]

# Removed theft_title and bed, since they have low correlation with price
# Added bed_length

# Create collective car_data column list
columns = numColsToFill + boolColsToFill + strColsToFill

# Assign a numerical value to each class in string columns
bed_data = mapcols(car_data[columns],strColsToFill)


## Dummy Variabling 

mldata_num = bed_data.copy()

# Add target variable

mldata_num['price'] = car_data['price']

# Name target and predictor variables 

target = 'price'
predictors = columns

# Get X and Y fields for ML

X = mldata_num[columns].values
y = mldata_num[target].values


# np.isnan(mldata_num.astype(float).to_numpy())


# %%
## Let's Start Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
mm_scaler = MinMaxScaler(feature_range=(0,1)) 
st_scaler = StandardScaler()

# fit standard and MinMax scalers

mm_fit = mm_scaler.fit(X)
st_fit = st_scaler.fit(X)

# transform to create source values for training

X1 = mm_scaler.transform(X)
X2 = st_scaler.transform(X)

from sklearn.model_selection import train_test_split

# Split data into training/testing and unwrap into lists

mm_train = [*train_test_split(X1,y, test_size=0.3, random_state=42)]
st_train = [*train_test_split(X2,y, test_size=0.3, random_state=42)]


# %%
## Import Regression Models

# Multiple Linear Regression - fast, but inaccurate
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()

# Random Forest Regression - slow, but very accurate
from sklearn.ensemble import RandomForestRegressor 
rfreg = RandomForestRegressor()

# Decision Tree Regression - medium of two above 
from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor() 

# Support Vector Regression - most versatile, depends on kernel
from sklearn.svm import SVR 
# polynomial fit, radial basis funciton, don't know how it works
svreg1 = SVR(kernel="rbf", C=100, gamma="auto", epsilon=0.1) 
# linear fit, expects linear data
svreg2 = SVR(kernel="linear", C=100, gamma="auto")
# polynomial fit, expects curvy line data
svreg3 = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

# %%
#  Import accuracy analysis tools

from sklearn import metrics

# Train and apply regression model to training data
def get_results(x_train, x_test, y_train, y_test, regressor):
    print(f'\n##### Model Validation and Accuracy Calculations for {type(regressor).__name__} ##########')
    
    # Fit regression model with training data (learning)
    model = regressor.fit(x_train,y_train)

    # Get preliminar R2 value to check correlation once again
    print('R2 Value mm:',metrics.r2_score(y_test, model.predict(x_test)))

    # predict y values using testing x data
    prediction = model.predict(x_test)
    
    # Analyze error results, method from Farukh Hashmi
    TestingDataResults = pd.DataFrame(data=x_test, columns=predictors)
    TestingDataResults[target] = y_test 
    TestingDataResults[('Predicted'+target)]=np.round(prediction)

    print(TestingDataResults[[target,'Predicted'+target]].head())

    # Calculate Average Percent Error using test data
    TestingDataResults['APE'] = 100 * ((abs(TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])

    # Get mean and medain of Average Percent Error
    MAPE = np.mean(TestingDataResults['APE'])
    MedianMAPE = np.median(TestingDataResults['APE'])

    # Calculate accuracy through inverse of Mean Average Percent Error
    Accuracy =100 - MAPE
    MedianAccuracy=100- MedianMAPE
    print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
    print('Median Accuracy on test data:', MedianAccuracy)

    # return results and trained model
    return TestingDataResults, model

# Check accuracy of model/model output using MAPE accuracy score
from sklearn.model_selection import cross_val_score 
def check_accuracy(reg,X,y):
    def AccScore(orig,pred): 
        MAPE = np.mean(100*(np.abs(orig-pred)/orig))
        return (100-MAPE)

    custom_Scorer = metrics.make_scorer(AccScore, greater_is_better=True)   


    Acc_Vals = cross_val_score(reg, X, y, cv=10, scoring=custom_Scorer)
    print('\nAccuracy values for 10-fold Cross Validation:\n',Acc_Vals)
    print('\nFinal Average Accuracy of the model:', round(Acc_Vals.mean(),2))
    return round(Acc_Vals.mean(),2)

final_accuracies = []
regressors = [lreg,rfreg,dtreg,svreg1,svreg2,svreg3]
results_arr = [] 

# get results from many regressors
# collect final accuracies and results dataframes
for regressor in regressors: 
    results, reg = get_results(*mm_train,regressor)
    final_accuracies.append(check_accuracy(reg,X1,y))
    results_arr.append(results)




# %%
## Plot best regressor output vs test values 
best_result_index = final_accuracies.index((max(final_accuracies))) 
best_result = results_arr[best_result_index]


plt.figure(figsize=(12,5))
plt.scatter(best_result['price'],best_result['Predictedprice'])
plt.title(f'Best regressor ({type(regressors[best_result_index]).__name__}) price vs predicted price')
plt.ylabel('Predicted price')
plt.xlabel('Price')
_ = sns.regplot(x=best_result.price,y=best_result.Predictedprice,line_kws={"color":"black"})

# %%
# Plot final accuracies of all used regression models
regressor_names = [type(c).__name__ for c in regressors]
print(car_data.shape[0])
plt.figure(figsize=(12,5))
b = plt.bar(regressor_names[0:3],final_accuracies[0:3])
plt.bar_label(b)
plt.title(f'Final Accuracy Comparison for Regression Models ({car_data.shape[0]} rows)')
plt.ylabel('Final Average Accuracy (10 iterations)')

plt.figure(figsize=(12,5))
b = plt.bar([c.kernel for c in regressors[3:6]],final_accuracies[3:6])
plt.bar_label(b)
plt.title(f'Final Accuracy Comparison for SVR Kernels ({car_data.shape[0]} rows)')
plt.ylabel('Final Average Accuracy (10 iterations)')

# %%
# Predict for a value in the actual dataset 
plt.show()

print(rfreg.predict(X1[3].reshape(1,-1)))


