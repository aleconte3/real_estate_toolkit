from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, shuffle
from typing import Optional, List, Dict, Any
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.market import HousingMarket
from real_estate_toolkit.agent_based_model.consumers import Segment, Consumer

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()

@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float

@dataclass
class ChildrenRange:
    minimum: int = 0
    maximum: int = 5

@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    
    def create_housing_market(self):
        houses = []
        for data in self.housing_market_data:
            house = House(**data)
            houses.append(house)
        self.housing_market = HousingMarket(houses)

    def create_consumers(self) -> None:
        self.consumers = []
        for _ in range(self.consumers_number):
            income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
            while not (self.annual_income.minimum <= income <= self.annual_income.maximum):
                income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
            
            children = randint(self.children_range.minimum, self.children_range.maximum)
            segment = Segment(randint(1, 3))
            consumer = Consumer(
                id=_,
                annual_income=income,
                children_number=children,
                segment=segment,
                house=None
            )
            self.consumers.append(consumer)

    def compute_consumers_savings(self) -> None:
        for consumer in self.consumers:
            consumer.compute_savings(self.years)

    def clean_the_market(self) -> None:
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda x: x.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda x: x.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            shuffle(self.consumers)
        
        for consumer in self.consumers:
            consumer.buy_a_house(self.housing_market)

    def compute_owners_population_rate(self) -> float:
        owners = sum(1 for consumer in self.consumers if consumer.house is not None)
        return owners / len(self.consumers)

    def compute_houses_availability_rate(self) -> float:
        available_houses = sum(1 for house in self.housing_market.houses if house.available)
        return available_houses / len(self.housing_market.houses)

    def print_simulation_results(self):
        owners_population_rate = self.compute_owners_population_rate()
        houses_availability_rate = self.compute_houses_availability_rate()
        
        print(f"Owners Population Rate: {owners_population_rate * 100:.2f}%")
        print(f"Houses Availability Rate: {houses_availability_rate * 100:.2f}%")