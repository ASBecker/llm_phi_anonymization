import os
import shutil
from datetime import datetime
from llm_anonymization_pipeline import AnonymizationPipeline
from pathlib import Path
import logging

def run_pipeline_for_model(model_name, excel_file, report_column, n_reports=1000, results_dir="results_m4timing_"):
    # Create model-specific directory
    model_dir = f"{results_dir}{model_name.replace(':', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Clear existing handlers before creating new pipeline
    logger = logging.getLogger("AnonymizationPipeline")
    logger.handlers.clear()

    # Define model-specific files
    regex_file = os.path.join(model_dir, "regex_rules.json")
    results_file = os.path.join(model_dir, f"{Path(excel_file).stem}_anonymized_{timestamp}.xlsx")

     # Process reports
    print(f"Results will be saved to: {results_file}")
    
    # Copy base regex rules if they exist
    if os.path.exists("regex_rules.json") and not os.path.exists(regex_file):
        shutil.copy("regex_rules.json", regex_file)
    
    # Initialize pipeline with model-specific settings
    pipeline = AnonymizationPipeline(
        regex_file=regex_file,
        llm_model=model_name,
        results_dir=results_dir
    )
    
    # Process first n reports
    df = pipeline.process_excel_file(
        excel_file,
        report_column,
        max_reports=n_reports
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(model_dir, f"anonymized_reports_{timestamp}.xlsx")
    df.to_excel(results_file, index=False)
    
    return output_file

def main():
    # Configuration
    models = [        
        #"qwen2.5-coder:32b",
        #"qwen2.5-coder:14b",
        #"qwen2.5-coder:7b",
        #"llama3.3",
        "llama3.1:8b",        
        #"llama3.2:1b",
        #"llama3.2:3b",
        "phi3:14b",
        "phi4",
        "llama3.1:70b"
        #"phi3:3.8b"
    ]
    
    excel_file = "radiology-full-text.xlsx"  # your input file
    report_column = "narrative"  # your report column name
    n_reports = 100
    
    # Process with each model
    results = {}
    for model in models:
        print(f"\nProcessing with model: {model}")
        try:
            output_file = run_pipeline_for_model(model, excel_file, report_column, n_reports)
            results[model] = {
                "status": "success",
                "output_file": output_file
            }
        except Exception as e:
            print(f"Error processing with {model}: {str(e)}")
            results[model] = {
                "status": "error",
                "error": str(e)
            }
    
    # Print summary
    print("\nProcessing Summary:")
    for model, result in results.items():
        status = result["status"]
        if status == "success":
            print(f"{model}: Success - Output saved to {result['output_file']}")
        else:
            print(f"{model}: Failed - {result['error']}")

if __name__ == "__main__":
    main() 