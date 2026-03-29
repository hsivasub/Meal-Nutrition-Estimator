import unittest
import os
import json
import tempfile
import sys

# Ensure src module is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nutrition_retriever import NutritionRetriever

class TestNutritionRetriever(unittest.TestCase):
    def setUp(self):
        # Create a temporary category map for isolated testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.map_path = os.path.join(self.temp_dir.name, "temp_map.json")
        self.mock_map = {
            "pizza": {"fdc_id": 1104332, "description": "Cheese pizza"},
            "sushi": {"fdc_id": 1102871, "description": "Sushi"}
        }
        with open(self.map_path, 'w') as f:
            json.dump(self.mock_map, f)
            
        self.retriever = NutritionRetriever(category_map_path=self.map_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_match(self):
        """Test retrieving nutrition for a known food category."""
        result = self.retriever.get_nutrition("pizza")
        self.assertIn("match_status", result)
        self.assertEqual(result["match_status"], "mock_match") # Mock since no USDA CSV
        self.assertEqual(result["fdc_id"], 1104332)
        self.assertEqual(result["description"], "Cheese pizza")
        self.assertIn("nutrition", result)

    def test_no_match(self):
        """Test handling cases where the food category doesn't exist."""
        result = self.retriever.get_nutrition("unknown_food")
        self.assertEqual(result["match_status"], "no_match")
        self.assertIn("error", result)

if __name__ == '__main__':
    unittest.main()
