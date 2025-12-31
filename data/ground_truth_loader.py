"""
Ground Truth Data Loader for the Agentic Text2Cypher pipeline.

Loads and provides access to the ground truth question-query pairs.
"""
import csv
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthItem:
    """A single ground truth question-query pair."""

    id: int
    reasoning_level: str  # Tingkat Penalaran
    sublevel: str  # Sublevel
    complexity: str  # Tingkat Kompleksitas
    question: str  # Pertanyaan
    cypher_query: str  # Cypher Query

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "reasoning_level": self.reasoning_level,
            "sublevel": self.sublevel,
            "complexity": self.complexity,
            "question": self.question,
            "cypher_query": self.cypher_query,
        }


class GroundTruthLoader:
    """
    Loader for ground truth data.

    Provides methods to load and filter ground truth question-query pairs.
    """

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            csv_path: Path to the ground truth CSV file.
                     If None, uses default path.
        """
        if csv_path is None:
            # Default path relative to this file
            csv_path = Path(__file__).parent / "ground_truth" / "ground-truth_refined.csv"

        self.csv_path = Path(csv_path)
        self.items: List[GroundTruthItem] = []
        self._loaded = False

    def load(self) -> List[GroundTruthItem]:
        """
        Load ground truth data from CSV.

        Returns:
            List of GroundTruthItem objects
        """
        if self._loaded:
            return self.items

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.csv_path}")

        self.items = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                item = GroundTruthItem(
                    id=idx + 1,
                    reasoning_level=row.get("Tingkat Penalaran", ""),
                    sublevel=row.get("Sublevel", ""),
                    complexity=row.get("Tingkat Kompleksitas", ""),
                    question=row.get("Pertanyaan", ""),
                    cypher_query=row.get("Cypher Query", ""),
                )
                self.items.append(item)

        self._loaded = True
        logger.info(f"Loaded {len(self.items)} ground truth items from {self.csv_path}")

        return self.items

    def get_all(self) -> List[GroundTruthItem]:
        """Get all ground truth items."""
        if not self._loaded:
            self.load()
        return self.items

    def get_by_id(self, item_id: int) -> Optional[GroundTruthItem]:
        """Get a specific item by ID."""
        if not self._loaded:
            self.load()

        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_by_complexity(self, complexity: str) -> List[GroundTruthItem]:
        """
        Filter items by complexity level.

        Args:
            complexity: One of "Easy", "Medium", "Hard"

        Returns:
            Filtered list of items
        """
        if not self._loaded:
            self.load()

        return [item for item in self.items if item.complexity.lower() == complexity.lower()]

    def get_by_reasoning_level(self, level: str) -> List[GroundTruthItem]:
        """
        Filter items by reasoning level.

        Args:
            level: One of "Fakta Eksplisit", "Fakta Implisit", "Inferensi"

        Returns:
            Filtered list of items
        """
        if not self._loaded:
            self.load()

        return [item for item in self.items if item.reasoning_level.lower() == level.lower()]

    def get_by_sublevel(self, sublevel: str) -> List[GroundTruthItem]:
        """
        Filter items by sublevel.

        Args:
            sublevel: One of "Nodes", "One-hop", "Multi-hop"

        Returns:
            Filtered list of items
        """
        if not self._loaded:
            self.load()

        return [item for item in self.items if item.sublevel.lower() == sublevel.lower()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the ground truth data.

        Returns:
            Dictionary with statistics
        """
        if not self._loaded:
            self.load()

        stats = {
            "total_items": len(self.items),
            "by_complexity": {},
            "by_reasoning_level": {},
            "by_sublevel": {},
        }

        for item in self.items:
            # Count by complexity
            stats["by_complexity"][item.complexity] = \
                stats["by_complexity"].get(item.complexity, 0) + 1

            # Count by reasoning level
            stats["by_reasoning_level"][item.reasoning_level] = \
                stats["by_reasoning_level"].get(item.reasoning_level, 0) + 1

            # Count by sublevel
            stats["by_sublevel"][item.sublevel] = \
                stats["by_sublevel"].get(item.sublevel, 0) + 1

        return stats

    def __len__(self) -> int:
        """Return number of items."""
        if not self._loaded:
            self.load()
        return len(self.items)

    def __iter__(self):
        """Iterate over items."""
        if not self._loaded:
            self.load()
        return iter(self.items)


def load_ground_truth(csv_path: Optional[str] = None) -> List[GroundTruthItem]:
    """
    Convenience function to load ground truth data.

    Args:
        csv_path: Optional path to CSV file

    Returns:
        List of GroundTruthItem objects
    """
    loader = GroundTruthLoader(csv_path)
    return loader.load()
