class HealthScoreEngine:
    """
    A rule-based analytics engine that evaluates the nutritional layout of a meal.
    Computes a score from 0-100 and applies a traffic-light label (Green, Yellow, Red)
    for easy user comprehension.
    """
    def __init__(self):
        # Recommended base limits for a single meal 
        # (Assuming an average 2000 kcal daily diet spread over 3 meals)
        self.meal_targets = {
            "calories": 600.0,
            "protein_g": 20.0,
            "fat_g": 20.0,
            "carbs_g": 60.0,
            "sugar_g": 10.0,
            "sodium_mg": 600.0
        }

    def calculate_score(self, nutrition_dict):
        """
        Calculates a health score between 0 and 100 based on parsed macros.
        """
        score = 100.0
        
        # Calories penalty: Over 800 heavily docks points, under 150 docks a bit
        cals = nutrition_dict.get('calories', 0)
        if cals > 800:
            score -= (cals - 800) * 0.05
        elif cals < 150:
            score -= 10
            
        # Protein bonus/penalty
        protein = nutrition_dict.get('protein_g', 0)
        if protein < 10:
            score -= 10  # Needs more protein
        elif protein > self.meal_targets["protein_g"]:
            score += min(10, (protein - self.meal_targets["protein_g"]) * 0.5) # Bonus capped at 10 pts
            
        # Fat penalty: Deduct for overflowing threshold
        fat = nutrition_dict.get('fat_g', 0)
        if fat > self.meal_targets["fat_g"]:
            score -= (fat - self.meal_targets["fat_g"]) * 1.5
            
        # Carbs penalty
        carbs = nutrition_dict.get('carbs_g', 0)
        if carbs > self.meal_targets["carbs_g"]:
            score -= (carbs - self.meal_targets["carbs_g"]) * 0.5
            
        # Sugar penalty: High penalty multiplier
        sugar = nutrition_dict.get('sugar_g', 0)
        if sugar > self.meal_targets["sugar_g"]:
            score -= (sugar - self.meal_targets["sugar_g"]) * 2.0
            
        # Sodium penalty
        sodium = nutrition_dict.get('sodium_mg', 0)
        if sodium > self.meal_targets["sodium_mg"]:
            score -= (sodium - self.meal_targets["sodium_mg"]) * 0.02
        
        # Clamping score strictly between 0 and 100
        final_score = max(0.0, min(100.0, score))
        return round(final_score, 1)

    def assign_traffic_light(self, score):
        """
        Returns Green (Healthy), Yellow (Moderate), or Red (Unhealthy).
        """
        if score >= 80:
            return "Green"
        elif score >= 50:
            return "Yellow"
        else:
            return "Red"

    def evaluate_meal(self, nutrition_dict):
        """
        End-to-end evaluation providing both the score and the traffic light label.
        
        Args:
            nutrition_dict (dict): Dictionary comprising calories, protein_g, fat_g, carbs_g, etc.
            
        Returns:
            dict: Evaluated score and derived label.
        """
        score = self.calculate_score(nutrition_dict)
        label = self.assign_traffic_light(score)
        
        return {
            "health_score": score,
            "traffic_light": label
        }
