import os
import torch
from torchvision import transforms
from PIL import Image

from src.food_classifier import FoodEfficientNet
from src.nutrition_retriever import NutritionRetriever
from src.portion_estimator import ReferenceObjectPortionEstimator, scale_nutrition
from src.health_scorer import HealthScoreEngine

class FullInferencePipeline:
    """
    Wires up the entire prediction pipeline end-to-end:
    Image -> Food Label -> Baseline Nutrition -> Portion Scaling -> Health Score.
    Includes input validation and isolated error handling.
    """
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize PyTorch Food Classifier
        self.classifier = FoodEfficientNet(num_classes=20, freeze_backbone=True).to(self.device)
        self.classifier.eval()
        if model_path and os.path.exists(model_path):
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            
        # 2. Extract Data via Nutrition Retriever
        self.nutrition_retriever = NutritionRetriever()
        
        # 3. OpenCV Distance/Contour Scaling Estimator
        self.portion_estimator = ReferenceObjectPortionEstimator()
        
        # 4. Health Score Analytic Engine
        self.health_engine = HealthScoreEngine()
        
        # Preprocessing block for inference inputs
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # MVP Static Mapping (Typically loaded dynamically via configuration)
        self.idx_to_class = {
            0: "pizza", 1: "hamburger", 2: "sushi", 3: "french_fries", 4: "hot_dog",
            5: "fried_rice", 6: "ramen", 7: "ice_cream", 8: "donuts", 9: "macarons",
            10: "tacos", 11: "steak", 12: "spaghetti_bolognese", 13: "chicken_wings",
            14: "omelette", 15: "caesar_salad", 16: "dumplings", 17: "grilled_cheese_sandwich",
            18: "pancakes", 19: "waffles"
        }

    def predict(self, image_path):
        """
        Executes the full chain on a target image input payload.
        
        Returns dictionary containing explicit success/failure statues and cascading metadata.
        """
        # Validate Input Integrity
        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image file not found: {image_path}"}
            
        try:
            # Safely attempt to read Image
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                return {"status": "error", "error": "Invalid image file format", "details": str(e)}

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Step 1: Execute Forward Pass on Classifier
            with torch.no_grad():
                output = self.classifier(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                
            predicted_label = self.idx_to_class.get(predicted_idx, "pizza")
            
            # Step 2: Grab Core Nutrition Facts
            nutrition_resp = self.nutrition_retriever.get_nutrition(predicted_label)
            if nutrition_resp.get("match_status") == "no_match":
                return {"status": "error", "error": "Failed to retrieve baseline nutrition facts", "details": nutrition_resp}
                
            base_nutrition = nutrition_resp.get("nutrition", {})
            
            # Step 3: Compute Scale based on Portion Size Estimation
            estimated_grams = self.portion_estimator.estimate_grams(image_path)
            scaled_nutrition = scale_nutrition(base_nutrition, estimated_grams)
            
            # Step 4: Run Health Analytics Rating
            health_eval = self.health_engine.evaluate_meal(scaled_nutrition)
            
            # Step 5: Format Consolidated Summary
            return {
                "status": "success",
                "predicted_label": predicted_label,
                "description": nutrition_resp.get("description", ""),
                "portion_grams": estimated_grams,
                "scaled_nutrition": scaled_nutrition,
                "health_score": health_eval["health_score"],
                "traffic_light": health_eval["traffic_light"]
            }
            
        except Exception as e:
            return {"status": "error", "error": "Unhandled Pipeline Exception Occurred", "details": str(e)}

if __name__ == "__main__":
    pipeline = FullInferencePipeline()
    print("Inference Pipeline Successfully Initialized.")
