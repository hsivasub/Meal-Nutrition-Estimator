import json
import os
import pandas as pd

class NutritionRetriever:
    """
    Retrieves nutritional information for a given food label by mapping
    it to a USDA FoodData Central ID (FDC ID) and looking up the macros.
    """
    def __init__(self, category_map_path="data/processed/food_category_map.json", usda_csv_path=None):
        self.category_map = self._load_json(category_map_path)
        self.usda_data = None
        
        if usda_csv_path and os.path.exists(usda_csv_path):
            # Assumes CSV has columns like 'fdc_id', 'calories', 'protein', etc.
            self.usda_data = pd.read_csv(usda_csv_path)

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing category map at {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def get_nutrition(self, predicted_label):
        """
        Retrieves nutritional information for a predicted food label.
        
        Args:
            predicted_label (str): The class name predicted by our model.
            
        Returns:
            dict: Dictionary containing nutrition info, matched FDC ID, and status.
        """
        # Edge Case 1: No match found for the predicted label
        if predicted_label not in self.category_map:
            return {
                "error": f"Label '{predicted_label}' not found in category map",
                "match_status": "no_match"
            }
            
        mapping = self.category_map[predicted_label]
        fdc_id = mapping.get("fdc_id")
        description = mapping.get("description", "Unknown Description")
        
        # If we have a local USDA database loaded
        if self.usda_data is not None:
            # Edge Case 2: Handling multiple matches or exact match in USDA DB
            matches = self.usda_data[self.usda_data['fdc_id'] == fdc_id]
            if not matches.empty:
                # We take the first match if there are multiple
                record = matches.iloc[0].to_dict()
                return {
                    "nutrition": record, 
                    "description": description,
                    "match_status": "exact_match", 
                    "fdc_id": fdc_id
                }
            else:
                return {
                    "error": f"FDC ID {fdc_id} not found in USDA local database", 
                    "match_status": "no_match"
                }
        
        # Fallback/Mock return if no CSV is present (e.g., MVP testing without full DB)
        return {
            "nutrition": {
                "calories": 250.0, # Placeholder
                "protein_g": 10.0,
                "carbs_g": 30.0,
                "fat_g": 10.0
            },
            "description": description,
            "match_status": "mock_match",
            "fdc_id": fdc_id
        }
