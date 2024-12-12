import polars as pl
import os
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        
        Args:
            train_data_path (str): Path to the training dataset CSV file.
            test_data_path (str): Path to the testing dataset CSV file.
        
        Attributes to Initialize:
            - self.train_data: Polars DataFrame for the training dataset.
            - self.test_data: Polars DataFrame for the testing dataset.
        """
        # Load datasets
        self.train_data = pl.read_csv(train_data_path, separator=";", null_values=["NA"])
        self.test_data = pl.read_csv(test_data_path, separator=";", null_values=["NA"])

    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.
        """
        print("Starting data cleaning...")

        # Mode values for categorical columns (manually entered)
        mode_values = {
            'MSZoning': 'RL', 'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl',
            'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'NAmes',
            'Condition1': 'Norm', 'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '1Story',
            'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None', 'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc', 'BsmtQual': 'TA',
            'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'Unf', 'BsmtFinType2': 'Unf',
            'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y', 'Electrical': 'SBrkr', 'KitchenQual': 'TA',
            'Functional': 'Typ', 'FireplaceQu': 'None', 'GarageType': 'Attchd', 'GarageFinish': 'Unf',
            'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y', 'PoolQC': 'None', 'Fence': 'None',
            'MiscFeature': 'None', 'SaleType': 'WD', 'SaleCondition': 'Normal'
        }

        print("Filling missing values with mode values...")
        for col, mode_value in mode_values.items():
            if col in self.train_data.columns:
                self.train_data = self.train_data.with_columns(
                    pl.col(col).fill_null(mode_value).alias(col)
                )
                # Commenting out print statements to reduce unnecessary output
                # print(f"Filled missing values in {col} with mode value: {mode_value}")

        for col, mode_value in mode_values.items():
            if col in self.test_data.columns:
                self.test_data = self.test_data.with_columns(
                    pl.col(col).fill_null(mode_value).alias(col)
                )
                # Commenting out print statements to reduce unnecessary output
                # print(f"Filled missing values in {col} with mode value: {mode_value}")

        # Commenting out these print statements to avoid outputting rows count
        # print(f"Remaining rows in train data: {self.train_data.height}")
        # print(f"Remaining rows in test data: {self.test_data.height}")

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable, 
        and preprocessing them for training and testing.
        """
        print(f"Preparing features with target column: {target_column}...")

        if selected_predictors is None:
            selected_predictors = [col for col in self.train_data.columns if col != target_column]

        # Convert Polars DataFrame to Pandas DataFrame for ML
        X_train = self.train_data.select(selected_predictors).to_pandas()
        y_train = self.train_data.select(target_column).to_pandas().values.ravel()  # Ensure y_train is 1D array

        X_test = self.test_data.select(selected_predictors).to_pandas()
        y_test = None  # Test data does not contain SalePrice

        print("Features prepared successfully.")
        # Commenting out the print statement to avoid excessive output
        # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        # print(f"X_test shape: {X_test.shape}")

        # Preprocessing: Handle categorical and numerical columns separately
        # Categorical columns: apply OneHotEncoder
        # Numerical columns: apply StandardScaler for normalization
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), X_train.select_dtypes(include=['object']).columns)
            ])

        # Apply transformations
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Impute missing values using SimpleImputer for numerical features
        imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean for numerical columns
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        print(f"Transformed X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_baseline_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate baseline machine learning models for house price prediction.
        
        Models:
        1. Linear Regression
        2. RandomForestRegressor or GradientBoostingRegressor

        Returns:
            A dictionary of models with their performance metrics (MSE, R2, MAE, MAPE).
        """
        # Prepare features and target variables
        X_train, X_test, y_train, y_test = self.prepare_features()

        # Create the model
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
        }

        results = {}

        # Print available models and their keys for debugging
        print(f"Available models in results: {models.keys()}")

        for model_name, model in models.items():
            print(f"Training {model_name} model...")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics only if y_test is not None
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test) if y_test is not None else None
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test) if y_test is not None else None
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test) if y_test is not None else None
            mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) if y_test is not None else None
            
            results[model_name] = {
                "metrics": {
                    "MSE_train": mse_train,
                    "MSE_test": mse_test,
                    "MAE_train": mae_train,
                    "MAE_test": mae_test,
                    "R2_train": r2_train,
                    "R2_test": r2_test,
                    "MAPE_train": mape_train,
                    "MAPE_test": mape_test,
                },
                "model": model
            }

        return results

    def forecast_sales_price(self, model_type: str = 'LinearRegression'):
        """
        Use the trained model to forecast house prices on the test dataset.
        
        Args:
            model_type (str): Type of model to use for forecasting. Default is 'LinearRegression'.
        
        Returns:
            None. It saves the predictions in a CSV file.
        """
        print("Starting price forecasting...")
        
        # Prepare the features and target variables
        X_train, X_test, y_train, y_test = self.prepare_features()

        # Train models
        results = self.train_baseline_models()

        # Print available models and their keys
        print(f"Available models in results: {results.keys()}")
        
        # Ensure that the model exists in the results before proceeding
        if model_type in results:
            model = results[model_type]["model"]
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Prepare the results for submission
            submission = self.test_data.select("Id").to_pandas()  # Get IDs from the test set
            submission["SalePrice"] = y_pred
        