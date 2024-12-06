from real_estate_toolkit.agent_based_model.consumers import Consumer, Segment
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.market import HousingMarket


def test_consumer_buy_house():
    # Create houses
    house1 = House(id=1, price=250000, area=1500, bedrooms=3, year_built=2015, quality_score=None, available=True)
    house2 = House(id=2, price=300000, area=1800, bedrooms=4, year_built=2018, quality_score=None, available=True)

    # Create a consumer with enough savings and segment OPTIMIZER
    consumer = Consumer(id=1, annual_income=60000, children_number=2, segment=Segment.OPTIMIZER)
    consumer.savings = 250000 * 0.2  # Enough for house1 down payment

    # Create a housing market
    market = HousingMarket(houses=[house1, house2])

    # Attempt to buy a house
    consumer.buy_a_house(market)

    # Verify the purchase
    assert consumer.house == house1, "Consumer should have bought house1"
    assert not house1.available, "House1 should not be available after purchase"
    assert consumer.savings == 0, "Savings should be reduced by the down payment"


def test_consumer_no_house_due_to_lack_of_savings():
    # Create a house
    house1 = House(id=1, price=250000, area=1500, bedrooms=3, year_built=2015, quality_score=None, available=True)

    # Create a consumer with insufficient savings
    consumer = Consumer(id=2, annual_income=60000, children_number=2, segment=Segment.OPTIMIZER)
    consumer.savings = 10000  # Not enough for the down payment

    # Create a housing market
    market = HousingMarket(houses=[house1])

    # Attempt to buy a house
    consumer.buy_a_house(market)

    # Verify no purchase
    assert consumer.house is None, "Consumer should not have bought a house"
    assert house1.available, "House1 should still be available"