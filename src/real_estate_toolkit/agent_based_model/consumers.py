from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from src.real_estate_toolkit.agent_based_model.houses import House
from src.real_estate_toolkit.agent_based_model.market import HousingMarket


class Segment(Enum):
    FANCY = auto()  # Prefers new construction with high quality scores
    OPTIMIZER = auto()  # Focuses on price per square foot value
    AVERAGE = auto()  # Considers average market prices


@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time using compound interest formula.
        """
        self.savings = self.savings * ((1 + self.interest_rate) ** years) + (self.annual_income * self.saving_rate) * years

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house.
        """
        for house in housing_market.houses:
            down_payment = house.price * 0.2  # Assuming 20% down payment

            if self.savings >= down_payment:
                if self.segment == Segment.FANCY and house.is_new_construction():
                    self._complete_purchase(house, down_payment)
                    return
                elif self.segment == Segment.OPTIMIZER and house.calculate_price_per_square_foot() < self.annual_income / 12:
                    self._complete_purchase(house, down_payment)
                    return
                elif self.segment == Segment.AVERAGE and house.price <= housing_market.calculate_average_price():
                    self._complete_purchase(house, down_payment)
                    return

    def _complete_purchase(self, house: House, down_payment: float) -> None:
        """
        Helper method to handle purchasing logic.
        """
        self.house = house
        house.sell_house()
        self.savings -= down_payment