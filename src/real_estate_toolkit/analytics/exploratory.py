from typing import List, Dict, Any, Optional
import polars as pl
import os
import plotly.express as px
import plotly.graph_objects as go

class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            data_path (str): Path to the Ames Housing dataset
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

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        
        1. Group data by neighborhood
        2. Calculate price statistics for each neighborhood
        3. Create Plotly boxplot with:
            - Median prices
            - Price spread
            - Outliers
        
        Returns:
            Neighborhood statistics dataframe
            Save Plotly figures for neighborhood price comparison in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        # Step 1: Group data by neighborhood and compute price statistics
        neighborhood_stats = self.real_state_clean_data.group_by("Neighborhood").agg(
            pl.col("SalePrice").mean().alias("mean_price"),
            pl.col("SalePrice").std().alias("std_dev_price"),
            pl.col("SalePrice").min().alias("min_price"),
            pl.col("SalePrice").max().alias("max_price")
        )

        # Step 2: Create a boxplot comparing prices across neighborhoods
        fig = px.box(
            self.real_state_clean_data.to_pandas(),
            x="Neighborhood",
            y="SalePrice",
            title="Price Comparison by Neighborhood"
        )

        # Save the boxplot to the outputs folder
        output_folder = "src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        boxplot_file_path = os.path.join(output_folder, "neighborhood_price_comparison.png")
        fig.write_image(boxplot_file_path)
        print(f"Neighborhood price comparison boxplot saved to {boxplot_file_path}")

        return neighborhood_stats  # Return neighborhood statistics DataFrame

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for variables input.
        
        1. Pass a list of numerical variables
        2. Compute correlation matrix and plot it
        
        Args:
            variables (List[str]): List of variables to correlate
        
        Returns:
            Save Plotly figures for correlation heatmap in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        correlation_matrix = self.real_state_clean_data.select(variables).to_pandas().corr()
        fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Matrix")

        # Save the heatmap to the outputs folder
        output_folder = "src/real_estate_toolkit/analytics/outputs"
        os.makedirs(output_folder, exist_ok=True)

        heatmap_file_path = os.path.join(output_folder, "correlation_heatmap.png")
        fig.write_image(heatmap_file_path)
        print(f"Correlation heatmap saved to {heatmap_file_path}")

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        
        Scatter plots to create:
        1. House price vs. Total square footage
        2. Sale price vs. Year built
        3. Overall quality vs. Sale price
        
        Tasks to implement:
        - Use Plotly Express for creating scatter plots
        - Add trend lines
        - Include hover information
        - Color-code points based on a categorical variable
        - Save them in src/real_estate_toolkit/analytics/outputs/ folder.
        
        Returns:
            Dictionary of Plotly Figure objects for different scatter plots. 
        """
        scatter_plots = {}

        # Scatter plot for price vs area
        scatter_plots["price_vs_area"] = px.scatter(
            self.real_state_clean_data.to_pandas(), x="GrLivArea", y="SalePrice", 
            title="Price vs Area", trendline="ols", color="OverallQual"
        )
        scatter_plots["price_vs_area"].write_image("src/real_estate_toolkit/analytics/outputs/price_vs_area.png")

        # Scatter plot for price vs year built
        scatter_plots["price_vs_year_built"] = px.scatter(
            self.real_state_clean_data.to_pandas(), x="YearBuilt", y="SalePrice", 
            title="Price vs Year Built", trendline="ols", color="OverallQual"
        )
        scatter_plots["price_vs_year_built"].write_image("src/real_estate_toolkit/analytics/outputs/price_vs_year_built.png")

        # Scatter plot for overall quality vs price
        scatter_plots["quality_vs_price"] = px.scatter(
            self.real_state_clean_data.to_pandas(), x="OverallQual", y="SalePrice", 
            title="Quality vs Price", trendline="ols"
        )
        scatter_plots["quality_vs_price"].write_image("src/real_estate_toolkit/analytics/outputs/quality_vs_price.png")

        return scatter_plots

# Specify the path to your dataset
data_path = "/Users/aleconte/Documents/UNI/Upf/PROGRAMMING/real_estate_toolkit/data/train.csv"

# Create the MarketAnalyzer object
analyzer = MarketAnalyzer(data_path)

# Perform data cleaning
analyzer.clean_data()

# Generate price distribution analysis
price_statistics = analyzer.generate_price_distribution_analysis()

# Generate neighborhood price comparison
neighborhood_stats = analyzer.neighborhood_price_comparison()

# Generate correlation heatmap
variables_to_correlate = ['GrLivArea', 'OverallQual', 'SalePrice', 'LotArea']
analyzer.feature_correlation_heatmap(variables_to_correlate)

# Create scatter plots
scatter_plots = analyzer.create_scatter_plots()