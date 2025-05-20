# Machine Learning Project - README

## Team Members
1. Muhammad Wasif Shakeel (456092)
2. Muhammad Muntazar (470861)
3. Hafiz Abdul Basit (472617)
4. Abdullah Hassan (479819)
5. Adyaan Ahmed (479556)

## Overview
This repository contains code for a machine learning project that uses various regression models including Random Forest and XGBoost to predict a target variable. The code includes data preprocessing, feature engineering, model training, and evaluation.
1. `newsScrapper.py` contains the scraping script for getting news headline
2. `Data_Collection.ipynb` contains the scraping script for getting the data for cryptocurrencies.
3. `Dataset_Preprocessing.ipynb` contains the preprocessing of all the data from news and cryptocurrency prices.
4. `IDS_Final_Project.ipynb` contains the final pipeline that builds the model.

## Requirements
The code requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- pickle
- scipy

## Running in Google Colab

### Step 1: Open the Notebook in Colab
Upload your notebook to Google Colab by going to [colab.research.google.com](https://colab.research.google.com) and selecting "Upload" from the File menu.

### Step 2: Install Required Libraries
The code already includes a pip install command for seaborn. If you need to install any other libraries not available in Colab by default, you can add additional installation commands:

```python
!pip install seaborn
!pip install xgboost
```

### Step 3: Upload Data Files
Upload the merged_output.csv to Colab using the files panel on the left sidebar. Alternatively, you can load data directly using:

```python
from google.colab import files
uploaded = files.upload()
```

### Step 4: Execute the Notebook
Simply run all cells in sequence by selecting "Runtime" > "Run all" from the menu, or press Ctrl+F9.

## Code Structure

The code performs the following tasks:
1. **Data Loading and Exploration**: Loads data and performs initial exploration
2. **Data Preprocessing**: 
   - Handles missing values using SimpleImputer
   - Scales numerical features using StandardScaler or RobustScaler
   - Encodes categorical variables
   - Performs text feature extraction if applicable (TF-IDF)
3. **Feature Engineering**: Creates new features and prepares data for modeling
4. **Model Training**: 
   - Uses time series cross-validation
   - Trains multiple models (RandomForestRegressor, Ridge, XGBoost)
5. **Model Evaluation**: 
   - Calculates metrics (RMSE, MAE, R²)
   - Visualizes results

## Important Notes
- The code uses TimeSeriesSplit for cross-validation, which is appropriate for time series data
- Set `warnings.filterwarnings('ignore')` to suppress warnings
- Model artifacts are saved using pickle

## Troubleshooting
- For CUDA-related errors when using XGBoost, try setting `tree_method='hist'` instead of GPU acceleration
- If you face issues with library versions, check compatibility between the installed packages

## Customization
To adapt this code for your specific dataset:
1. Modify the data loading and preprocessing steps according to your data structure
2. Adjust feature engineering based on your domain knowledge
3. Tune model hyperparameters for better performance
4. Add or remove models as needed