from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1


@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True
    segment: Optional[str] = None

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.
        Returns 0.0 if area is zero.
        """
        if self.area == 0:
            return 0.0
        return round(self.price / self.area, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if house is considered new construction (< 5 years old).
        """
        return current_year - self.year_built < 5

    def get_quality_score(self) -> QualityScore:
        """
        Generate a quality score based on house attributes if none is provided.
        """
        if self.quality_score is not None:
            return self.quality_score
        if self.year_built > 2000 and self.bedrooms >= 3 and self.area >= 1500:
            return QualityScore.EXCELLENT
        elif self.year_built > 1980:
            return QualityScore.GOOD
        return QualityScore.FAIR

    def sell_house(self) -> None:
        """
        Mark house as sold.
        """
        self.available = False