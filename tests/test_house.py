# tests/test_house.py

from real_estate_toolkit.agent_based_model.houses import House, QualityScore

def test_calculate_price_per_square_foot():
    # Create a house instance
    house = House(id=1, price=300000, area=2000, bedrooms=4, year_built=2010, quality_score=None)
    
    # Test if the price per square foot is correctly calculated
    assert house.calculate_price_per_square_foot() == 150.0

def test_is_new_construction():
    # Create a house instance that is less than 5 years old
    house = House(id=2, price=350000, area=2500, bedrooms=5, year_built=2020, quality_score=None)
    
    # Test if the house is considered new construction
    assert house.is_new_construction() == True
    
    # Create a house instance that is older than 5 years
    house_old = House(id=3, price=200000, area=1500, bedrooms=3, year_built=2000, quality_score=None)
    
    # Test if the house is not considered new construction
    assert house_old.is_new_construction() == False

def test_get_quality_score():
    # Create a house instance with missing quality_score
    house = House(id=4, price=400000, area=1800, bedrooms=4, year_built=2015, quality_score=None)
    
    # Test if the house gets a quality score based on its attributes
    assert house.get_quality_score() == QualityScore.EXCELLENT  # Updated to expect EXCELLENT
    
    # Create a house that should be classified as GOOD
    house = House(id=5, price=300000, area=1400, bedrooms=3, year_built=1990, quality_score=None)
    assert house.get_quality_score() == QualityScore.GOOD  # Area > 1500 and year > 1980
    
    # Create a house that should be classified as FAIR
    house = House(id=6, price=250000, area=1000, bedrooms=2, year_built=1975, quality_score=None)
    assert house.get_quality_score() == QualityScore.FAIR  # Doesn't meet criteria for GOOD or EXCELLENT

def test_sell_house():
    # Create a house instance
    house = House(id=6, price=600000, area=3500, bedrooms=6, year_built=2018, quality_score=QualityScore.EXCELLENT)
    
    # Test if the house is initially available for sale
    assert house.available == True
    
    # Sell the house and mark it as not available
    house.sell_house()
    
    # Test if the house is now marked as sold (not available)
    assert house.available == False