import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ReferenceObjectPortionEstimator:
    """
    OpenCV-based implementation of reference object scaling.
    Uses an object of known size (e.g., a US Quarter, 24.26mm diameter)
    to calculate a pixels-per-metric ratio in the image, allowing
    for estimation of real-world food area and volume.
    """
    def __init__(self, reference_real_width_cm=2.426):
        self.reference_real_width_cm = reference_real_width_cm

    def estimate_grams(self, image_path, density_g_per_cm3=1.0):
        """
        Estimates the weight of the food in grams based on bounding box/contours.
        Requires edge detection and segmentation.
        """
        image = cv2.imread(image_path)
        if image is None:
            # Fallback mock value if image isn't valid
            return 250.0 
            
        # Standard CV Workflow Pipeline (Mocked logic for MVP):
        # 1. Grayscale & Blur
        # 2. Edge Detection (Canny) & Dilate/Erode
        # 3. Find Contours
        # 4. Identify reference object (e.g., leftmost, distinct shape)
        # 5. pixels_per_metric = ref_pixel_width / self.reference_real_width_cm
        # 6. Identify food contour -> compute pixel area
        # 7. Area in cm^2 = food_pixel_area / (pixels_per_metric ** 2)
        # 8. Volume in cm^3 = Area * assumed_height OR sphere estimation
        # 9. Weight (grams) = Volume * density_g_per_cm3
        
        # Returning a reasonable mock portion size (e.g., 350 grams)
        mock_food_weight_grams = 350.0 
        return mock_food_weight_grams


class RegressionPortionEstimator(nn.Module):
    """
    Deep Learning approach: Utilizes an EfficientNet-B0 backbone 
    with a regression head to output a continuous weight (grams) prediction
    directly from the meal image.
    """
    def __init__(self):
        super(RegressionPortionEstimator, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace the final linear classification layer with a regression block
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Predicts 1 continuous value: weight in grams
        )

    def forward(self, x):
        return self.backbone(x)


def scale_nutrition(nutrition_per_100g, estimated_grams):
    """
    Scales the base 100g USDA nutrition facts data based on the estimated portion size.
    
    Args:
        nutrition_per_100g (dict): Nutrition dictionary initialized per 100 grams.
        estimated_grams (float): The estimated weight of the food.
        
    Returns:
        dict: Scaled nutrition dictionary.
    """
    scale_factor = estimated_grams / 100.0
    scaled_nutrition = {}
    
    for nutrient, value in nutrition_per_100g.items():
        if isinstance(value, (int, float)):
            scaled_nutrition[nutrient] = round(value * scale_factor, 2)
        else:
            scaled_nutrition[nutrient] = value
            
    return scaled_nutrition
