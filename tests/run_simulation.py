# tests/run_simulation.py

from real_estate_toolkit.agent_based_model.simulation import Simulation, AnnualIncomeStatistics, ChildrenRange, CleaningMarketMechanism
from real_estate_toolkit.agent_based_model.houses import House, QualityScore

# Example data for the simulation
housing_market_data = [
    {"id": 1, "price": 250000, "area": 1500, "bedrooms": 3, "year_built": 2005, "quality_score": None},
    {"id": 2, "price": 300000, "area": 1800, "bedrooms": 4, "year_built": 2010, "quality_score": None},
    {"id": 3, "price": 350000, "area": 2000, "bedrooms": 5, "year_built": 2020, "quality_score": None}
]

annual_income_stats = AnnualIncomeStatistics(
    minimum=30000, 
    average=50000, 
    standard_deviation=15000, 
    maximum=80000
)

children_range = ChildrenRange(minimum=1, maximum=3)

# Create the Simulation object
simulation = Simulation(
    housing_market_data=housing_market_data, 
    consumers_number=5, 
    years=5, 
    annual_income=annual_income_stats, 
    children_range=children_range, 
    cleaning_market_mechanism=CleaningMarketMechanism.INCOME_ORDER_DESCENDANT
)

# Run the simulation
simulation.create_housing_market()  # Initialize the market
simulation.create_consumers()  # Create consumers
simulation.compute_consumers_savings()  # Calculate savings for each consumer
simulation.clean_the_market()  # Execute market transactions

# Print the results
simulation.print_simulation_results()