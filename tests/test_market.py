from src.real_estate_toolkit.agent_based_model.houses import House, QualityScore
from src.real_estate_toolkit.agent_based_model.market import HousingMarket

def test_get_house_by_id():
    # Create sample houses
    house1 = House(id=1, price=250000, area=1500, bedrooms=3, year_built=2005, quality_score=QualityScore.GOOD)
    house2 = House(id=2, price=300000, area=1800, bedrooms=4, year_built=2010, quality_score=QualityScore.EXCELLENT)
    
    # Create housing market
    market = HousingMarket([house1, house2])
    
    # Test retrieving houses by ID
    assert market.get_house_by_id(1) == house1
    assert market.get_house_by_id(2) == house2
    
    # Test non-existent house ID
    try:
        market.get_house_by_id(999)  # This should raise an exception
    except ValueError as e:
        assert str(e) == "House with ID 999 not found."

def test_calculate_average_price():
    # Create sample houses
    house1 = House(id=1, price=250000, area=1500, bedrooms=3, year_built=2005, quality_score=QualityScore.GOOD)
    house2 = House(id=2, price=300000, area=1800, bedrooms=4, year_built=2010, quality_score=QualityScore.EXCELLENT)
    
    # Create housing market
    market = HousingMarket([house1, house2])
    
    # Test average price without filtering by bedrooms
    assert market.calculate_average_price() == 275000  # (250000 + 300000) / 2
    
    # Test average price with filtering by bedrooms
    assert market.calculate_average_price(bedrooms=4) == 300000  # Only house2 has 4 bedrooms

def test_get_houses_that_meet_requirements():
    # Create sample houses
    house1 = House(id=1, price=250000, area=1500, bedrooms=3, year_built=2005, quality_score=QualityScore.GOOD, segment="luxury")
    house2 = House(id=2, price=300000, area=1800, bedrooms=4, year_built=2010, quality_score=QualityScore.EXCELLENT, segment="luxury")
    house3 = House(id=3, price=350000, area=2000, bedrooms=5, year_built=2020, quality_score=QualityScore.EXCELLENT, segment="affordable")
    
    # Create housing market
    market = HousingMarket([house1, house2, house3])
    
    # Test filtering houses based on price and segment
    filtered_houses = market.get_houses_that_meet_requirements(280000, "luxury")
    
    # Ensure only house1 is returned
    assert len(filtered_houses) == 1  # Ensure there is only 1 house
    assert filtered_houses[0].id == house1.id
    assert filtered_houses[0].price == house1.price
    assert filtered_houses[0].segment == house1.segment