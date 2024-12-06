from typing import List, Dict, Optional
from real_estate_toolkit.agent_based_model.houses import House
import statistics

class HousingMarket:
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> House:
        """
        Retrieve specific house by ID.
        
        Implementation tips:
        - Use efficient search method
        - Handle non-existent IDs
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        raise ValueError(f"House with ID {house_id} not found.")

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate average house price, optionally filtered by bedrooms.
        
        Implementation tips:
        - Handle empty lists
        - Consider using statistics module
        - Implement bedroom filtering efficiently
        """
        if bedrooms is None:
            prices = [house.price for house in self.houses]
        else:
            prices = [house.price for house in self.houses if house.bedrooms == bedrooms]
        
        if not prices:
            raise ValueError("No houses found with the specified criteria.")
        
        return statistics.mean(prices)

    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> Optional[List[House]]:
        """
        Filter houses based on buyer requirements.
        
        Implementation tips:
        - Consider multiple filtering criteria
        - Implement efficient filtering
        - Handle case when no houses match
        """
        filtered_houses = [
            house for house in self.houses
            if house.price <= max_price and house.segment == segment
        ]
        
        if not filtered_houses:
            return None
        
        return filtered_houses