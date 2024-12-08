import polars as pl
import os
from typing import List, Dict, Any

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

        # Handle missing values
        total_rows_train = self.train_data.height  # Number of rows in the training DataFrame
        total_rows_test = self.test_data.height  # Number of rows in the testing DataFrame
        print(f"Total rows in train data: {total_rows_train}")
        print(f"Total rows in test data: {total_rows_test}")

        # Mode values for categorical columns (manually entered)
        mode_values = {
            'MSZoning': 'RL',
            'Street': 'Pave',
            'Alley': 'None',
            'LotShape': 'Reg',
            'LandContour': 'Lvl',
            'Utilities': 'AllPub',
            'LotConfig': 'Inside',
            'LandSlope': 'Gtl',
            'Neighborhood': 'NAmes',
            'Condition1': 'Norm',
            'Condition2': 'Norm',
            'BldgType': '1Fam',
            'HouseStyle': '1Story',
            'RoofStyle': 'Gable',
            'RoofMatl': 'CompShg',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None',
            'ExterQual': 'TA',
            'ExterCond': 'TA',
            'Foundation': 'PConc',
            'BsmtQual': 'TA',
            'BsmtCond': 'TA',
            'BsmtExposure': 'No',
            'BsmtFinType1': 'Unf',
            'BsmtFinType2': 'Unf',
            'Heating': 'GasA',
            'HeatingQC': 'Ex',
            'CentralAir': 'Y',
            'Electrical': 'SBrkr',
            'KitchenQual': 'TA',
            'Functional': 'Typ',
            'FireplaceQu': 'None',
            'GarageType': 'Attchd',
            'GarageFinish': 'Unf',
            'GarageQual': 'TA',
            'GarageCond': 'TA',
            'PavedDrive': 'Y',
            'PoolQC': 'None',
            'Fence': 'None',
            'MiscFeature': 'None',
            'SaleType': 'WD',
            'SaleCondition': 'Normal'
        }

        # Handle missing values in training data using the mode values
        for col, mode_value in mode_values.items():
            if col in self.train_data.columns:
                self.train_data = self.train_data.with_columns(
                    pl.col(col).fill_null(mode_value).alias(col)
                )
                print(f"Filling missing values in {col} with mode value: {mode_value}")

        # Handle missing values in test data using the mode values
        for col, mode_value in mode_values.items():
            if col in self.test_data.columns:
                self.test_data = self.test_data.with_columns(
                    pl.col(col).fill_null(mode_value).alias(col)
                )
                print(f"Filling missing values in {col} with mode value: {mode_value}")
        
        # After cleaning, display the size of the cleaned data
        print(f"Data cleaned successfully. Remaining rows in train data: {self.train_data.height}")
        print(f"Data cleaned successfully. Remaining rows in test data: {self.test_data.height}")
    
    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable, 
        and preprocessing them for training and testing.

        Args:
            target_column (str): Name of the target variable column. Default is 'SalePrice'.
            selected_predictors (List[str]): Specific columns to use as predictors. 
                                            If None, use all columns except the target.
        """
        # Separate predictors and target
        if selected_predictors is None:
            selected_predictors = [col for col in self.train_data.columns if col != target_column]
        
        X_train = self.train_data.select(selected_predictors).to_pandas()
        y_train = self.train_data.select(target_column).to_pandas()
        
        X_test = self.test_data.select(selected_predictors).to_pandas()
        y_test = self.test_data.select(target_column).to_pandas()

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
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
        }

        results = {}

        for model_name, model in models.items():
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
            
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
        # Prepare the features and target variables
        X_train, X_test, y_train, y_test = self.prepare_features()

        # Train models
        results = self.train_baseline_models()
        
        # Select the model
        model = results[model_type]["model"]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Prepare the results for submission
        submission = self.test_data.select("Id").to_pandas()  # Get IDs from the test set
        submission["SalePrice"] = y_pred
        
        # Save the submission to CSV
        output_folder = "src/real_estate_toolkit/ml_models/outputs"
        os.makedirs(output_folder, exist_ok=True)
        submission_file_path = os.path.join(output_folder, "submission.csv")
        submission.to_csv(submission_file_path, index=False)
        print(f"Predictions saved to {submission_file_path}")