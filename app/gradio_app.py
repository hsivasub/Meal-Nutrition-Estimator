import gradio as gr
import os
import sys
import json

# Ensure project root is in the path so we can import src modules natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import FullInferencePipeline

# Initialize the centralized inference pipeline instance globally for Gradio events
print("Binding ML models...")
pipeline = FullInferencePipeline()
print("Models successfully loaded to memory.")

def analyze_meal(image_filepath):
    """
    Gradio target function that receives the path of an uploaded image and 
    returns a formatted HTML output summarizing the inference, along with raw JSON metrics.
    """
    if image_filepath is None:
        return "Please upload an image to start.", "{}"

    # Execute end-to-end inference pipeline
    result = pipeline.predict(image_filepath)
    
    if result.get("status") == "error":
        error_msg = result.get("error", "Unknown Error")
        details = result.get("details", "")
        return f"<h3 style='color:red;'>Error Processing Image</h3><p>{error_msg} -> {details}</p>", "{}"
    
    # Successful prediction parsing
    label = result.get("predicted_label", "Unknown").title()
    desc = result.get("description", "No description")
    portion = result.get("portion_grams", "N/A")
    score = result.get("health_score", "N/A")
    traffic_light = result.get("traffic_light", "Yellow")
    nutrition = result.get("scaled_nutrition", {})
    
    # Set up traffic light thematic coloring
    color_map = {
        "Green": "#28a745",    # Healthy
        "Yellow": "#ffc107",   # Moderate
        "Red": "#dc3545"       # Unhealthy
    }
    hex_color = color_map.get(traffic_light, "black")
    
    # Structure HTML Display
    html_output = f"""
    <div style="font-family: sans-serif; background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
        <h2 style="color: #343a40; margin-top: 0;">Predicted Meal: <span style="color: #007bff;">{label}</span></h2>
        <p style="color: #6c757d; font-style: italic;">{desc}</p>
        <hr style="border: 1px solid #dee2e6;">
        <h3 style="color: #495057;">Health Analytics</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 8px;">🍽️ <b>Estimated Portion:</b> {portion}g</li>
            <li style="margin-bottom: 8px;">❤️ <b>Health Score:</b> {score} / 100 
                <span style="color: white; background-color: {hex_color}; padding: 3px 8px; border-radius: 12px; margin-left: 5px; font-weight: bold;">
                {traffic_light}
                </span>
            </li>
        </ul>
        <hr style="border: 1px solid #dee2e6;">
        <h3 style="color: #495057;">Scaled Macronutrients</h3>
        <ul style="line-height: 1.6;">
            <li><b>Calories:</b> {nutrition.get('calories', 'N/A')} kcal</li>
            <li><b>Protein:</b> {nutrition.get('protein_g', 'N/A')} g</li>
            <li><b>Carbs:</b> {nutrition.get('carbs_g', 'N/A')} g</li>
            <li><b>Fat:</b> {nutrition.get('fat_g', 'N/A')} g</li>
            <li><b>Sugar:</b> {nutrition.get('sugar_g', 'N/A')} g</li>
            <li><b>Sodium:</b> {nutrition.get('sodium_mg', 'N/A')} mg</li>
        </ul>
    </div>
    """
    
    # Serialize raw python dict to pretty JSON string for developers module
    raw_json = json.dumps(result, indent=2)
    return html_output, raw_json

# Constructing the Gradio Blocks Interface
with gr.Blocks(title="Meal Nutrition Estimator Web UI") as demo:
    gr.Markdown(
        """
        # 🥗 AI Meal Nutrition Estimator
        Upload a photo of your meal. Our platform uses an EfficientNet backbone to classify the food, OpenCV 
        scaling to estimate portion weight, and USDA mapping to generate a precise nutritional and health scoring breakdown.
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Meal Photo")
            submit_btn = gr.Button("Analyze Meal", variant="primary")
            
        with gr.Column():
            html_result = gr.HTML(label="Dashboard Report")
            
    with gr.Accordion("Developer Raw Data JSON", open=False):
        json_dump = gr.Code(language="json", label="Inference Payload")
            
    submit_btn.click(fn=analyze_meal, inputs=image_input, outputs=[html_result, json_dump])

if __name__ == "__main__":
    # Launch locally for dev
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, theme=gr.themes.Base())
