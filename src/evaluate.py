import os
import json
from src.predict import FullInferencePipeline

def evaluate_pipeline(data_dir):
    """
    Evaluates the full inference pipeline across a directory of images.
    Helpful for checking robustness, simulating stress tests, and ensuring modules string together beautifully.
    """
    print(f"Initializing pipeline engine...")
    pipeline = FullInferencePipeline()
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist for evaluation.")
        return
        
    results = []
    
    print(f"Scanning target directory: {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_dir, filename)
            
            # Run prediction; explicit exceptions are neatly caught inside predict
            resp = pipeline.predict(image_path)
            
            results.append({
                "image": filename,
                "result": resp
            })
            
    # Output robust metric summary
    success_count = sum(1 for r in results if r.get("result", {}).get("status") == "success")
    total = len(results)
    
    print(f"\n--- Evaluation Core Summary ---")
    print(f"Total Target Images: {total}")
    print(f"Successful Inferences: {success_count}")
    print(f"Failed Inferences: {total - success_count}")
    
    if results:
        print("\nSample Output From Pipeline:")
        print(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    # Test execution evaluating the pipeline over our mock data folder inside data/sample
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "..", "data", "sample")
    
    evaluate_pipeline(sample_dir)
