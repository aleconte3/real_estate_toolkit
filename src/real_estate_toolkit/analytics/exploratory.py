import polars as pl
import os
import plotly.express as px

class MarketAnalyzer2:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        """
        try:
            # Load the dataset into a Polars DataFrame with specified options
            self.real_state_data = pl.read_csv(
                data_path,
                separator=";",  # Ensure the separator is correct
                null_values=["NA"],  # Specify NA as null
            )
            print(f"Data loaded successfully from {data_path}.")
            
            # Initialize the clean data attribute as None (to be processed later)
            self.real_state_clean_data = None

        except Exception as e:
            print(f"Error loading data from {data_path}: {str(e)}")
   
    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        print("Starting data cleaning...")

        # Handle missing values
        total_rows = self.real_state_data.height  # Number of rows in the DataFrame
        print(f"Total rows in data: {total_rows}")

        # Separate numeric and categorical columns
        numeric_columns = [col for col in self.real_state_data.columns if 
                           self.real_state_data[col].dtype in [pl.Float64, pl.Int64]]
        categorical_columns = [col for col in self.real_state_data.columns if
                               self.real_state_data[col].dtype == pl.Utf8]

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

        # Fill missing values in numeric columns with the mean
        for col in numeric_columns:
            mean_value = self.real_state_data[col].mean()  # Directly get mean value
            self.real_state_data = self.real_state_data.with_columns(
                pl.col(col).fill_null(mean_value).alias(col)
            )
            print(f"Filling nulls in {col} with mean value: {mean_value}")

        # Fill missing values in categorical columns with the mode
        for col in categorical_columns:
            if col in mode_values:
                mode_value = mode_values[col]
                self.real_state_data = self.real_state_data.with_columns(
                    pl.col(col).fill_null(mode_value).alias(col)
                )
                print(f"Filling nulls in {col} with mode value: {mode_value}")
        
        # Assign cleaned data to the attribute
        self.real_state_clean_data = self.real_state_data
        print(f"Data cleaned successfully. Remaining rows: {self.real_state_clean_data.height}")
    
    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        
        1. Compute basic price statistics and generate another data frame called price_statistics:
            - Mean
            - Median
            - Standard deviation
            - Minimum and maximum prices
        2. Create an interactive histogram of sale prices using Plotly.
        
        Returns:
            - Statistical insights dataframe
            - Save Plotly figures for price distribution in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Cleaned data is not available. Please run clean_data() first.")

        # Step 1: Compute basic price statistics
        price_statistics = self.real_state_clean_data.select(
            [
                pl.col("SalePrice").mean().alias("mean_price"),
                pl.col("SalePrice").median().alias("median_price"),
                pl.col("SalePrice").std().alias("std_dev_price"),
                pl.col("SalePrice").min().alias("min_price"),
                pl.col("SalePrice").max().alias("max_price"),
            ]
        )

        print("Price Statistics:", price_statistics)  # Debug output to check statistics
        
        # Step 2: Create an interactive histogram of sale prices using Plotly
        fig = px.histogram(
            self.real_state_clean_data.to_pandas(),  # Convert the Polars DataFrame to Pandas
            x="SalePrice",
            title="Histogram of Sale Prices",
            labels={"SalePrice": "Sale Price"},
            color_discrete_sequence=["blue"],  # Customize the color if needed
            marginal="rug"  # Optional: Add a rug plot to show individual observations
        )

        # Define the output folder and ensure it exists
        output_folder = "src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        # Save the figure as an HTML file
        histogram_file_path = os.path.join(output_folder, "price_distribution_histogram.html")
        fig.write_html(histogram_file_path)
        print(f"Histogram saved to {histogram_file_path}")

        return price_statistics  # Return the price statistics DataFrame

# Specify the path to your dataset
data_path = "/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv"

# Create the MarketAnalyzer2 object
analyzer = MarketAnalyzer2(data_path)

# Perform data cleaning
analyzer.clean_data()

# Generate price distribution analysis
price_statistics = analyzer.generate_price_distribution_analysis()