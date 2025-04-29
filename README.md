# LLM Report Anonymization Pipeline

This repository contains the code accompanying the paper "Keeping Private Patient Data Off the Cloud: A Comparison of Local LLMs for Anonymizing Radiology Reports" (DOI: [10.1016/j.ejrai.2025.100020](https://doi.org/10.1016/j.ejrai.2025.100020)).

The goal of this project is to evaluate and provide a pipeline for using open-source, locally deployed Large Language Models (LLMs) to anonymize radiology reports. This approach preserves clinical data while ensuring compliance with privacy regulations like HIPAA, without relying on potentially insecure cloud-based solutions.

## Methodology

The core method involves using LLMs to generate Regular Expression (RegEx) rules based on identified Protected Health Information (PHI) in sample reports. These rules are then iteratively applied and validated to anonymize a larger dataset efficiently.

- Models are prompted to identify PHI according to HIPAA definitions.
- For identified PHI, the LLM generates a corresponding RegEx pattern.
- Generated RegEx rules are collected and applied to subsequent reports.
- The process repeats for each report to ensure thorough anonymization (min 3 passes, max 25).
- Models were run using the Ollama framework on both an NVIDIA A100 GPU and an Apple M4 Max.

## Features

*   Processes reports from Excel files.
*   Uses LLMs to automatically generate RegEx rules for PHI removal.
*   Applies generated and manual RegEx rules iteratively.
*   Supports comparison across different locally run LLM models.
*   Designed for privacy-preserving, offline execution (no cloud dependency).
*   Saves anonymized reports.

## Key Findings (from the paper)

*   **Qwen Models Excel:** Qwen-2.5-coder models (7B and 32B) significantly outperformed others, achieving high accuracy in both PHI removal (100% for 32B) and preservation of clinical data (97.6% for 7B).
*   **Llama Models Struggle:** Llama models (v3.1 8B/70B, v3.3 70B) frequently misidentified clinical numbers as PHI and failed to remove all patient names in some cases.
*   **Phi Models Unusable:** Phi3/4 models produced overly aggressive rules or hallucinated, rendering their output unusable for this task.
*   **Efficiency:** Qwen models demonstrated significantly faster runtimes compared to Llama models on both A100 and M4 Max hardware.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ASBecker/llm_phi_anonymization
    cd llm_phi_anonymization
    ```
2.  **Install dependencies:** Requires Python and pip.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Ollama:** Ensure Ollama (v0.5.0 or compatible) is installed and running. Pull the desired models:
    ```bash
    ollama pull qwen2.5-coder:7b
    ollama pull qwen2.5-coder:32b
    # ... pull other models as needed
    ```
4.  **Prepare your data:**
    *   Place your Excel file (e.g., `radiology-reports.xlsx`) in the project root directory.
    *   Ensure the Excel file has a column containing the text reports to be anonymized (default column name: `narrative`).
5.  **(Optional) Customize Initial Regex Rules:**
    *   Modify the `regex_rules.json` file to add initial manual rules if desired.

## Usage

There are two main ways to use this pipeline:

1.  **Comparing Multiple Models (Evaluation):**
    To evaluate and compare the performance of different LLMs (as done in the paper), use the `run_multi_model_pipeline.py` script. Configure the desired models within the script.
    ```bash
    python run_multi_model_pipeline.py
    ```
    This will create separate output directories for each model specified in the script's `models` list.

2.  **Anonymizing with a Single Model:**
    To anonymize a report file using a single, pre-configured LLM (defaults to `qwen2.5-coder:32b` in the pipeline class, but check `llm_anonymization_pipeline.py` for the current default), run the main pipeline script directly:
    ```bash
    python llm_anonymization_pipeline.py [path/to/your_excel_file.xlsx] [report_column_name]
    ```
    *   `[path/to/your_excel_file.xlsx]`: Optional. Defaults to `radiology-full-text.xlsx` if omitted.
    *   `[report_column_name]`: Optional. Defaults to `narrative` if omitted.

    Example:
    ```bash
    python llm_anonymization_pipeline.py my_reports.xlsx report_text
    ```

    Use the `--continue` flag to resume processing from the last report mentioned in the `anonymization.log` file:
    ```bash
    python llm_anonymization_pipeline.py --continue
    ```
    This mode will create a single output file (e.g., `radiology-full-text_anonymized_<timestamp>.xlsx`) and a single `anonymization.log` in the default results directory for that model.

### Configuration

*   **For Multi-Model Evaluation:** Modify `run_multi_model_pipeline.py` to set:
    *   **`models`**: List of LLM models available via Ollama to test. The paper evaluated:
        *   `qwen2.5-coder:7b`
        *   `qwen2.5-coder:32b`
        *   `llama3.1:8b`
        *   `llama3.1:70b`
        *   `llama3.3` (Assumed 70B based on paper context, adjust if needed)
        *   `phi3:14b`
        *   `phi4` (Specific size variant not mentioned, adjust if needed)
    *   **`excel_file`**: Path to your input Excel file.
    *   **`report_column`**: Name of the column containing the report text.
    *   **`n_reports`**: Maximum number of reports to process from the Excel file (Paper used 1000 for evaluation).
    *   **`results_dir`**: Base directory name prefix for storing results (a subdirectory will be created for each model).

*   **For Single-Model Usage:** The default LLM model used when running `llm_anonymization_pipeline.py` directly is set within the `AnonymizationPipeline` class constructor in `llm_anonymization_pipeline.py` (`llm_model` parameter).

## Output

*   **Multi-Model Evaluation:** Creates a results directory for *each model* tested (e.g., `results_m4timing_qwen2.5-coder_7b`), each containing:
    *   An Excel file containing the anonymized reports (e.g., `anonymized_reports_<timestamp>.xlsx`).
    *   A JSON file containing the final set of RegEx rules used (`regex_rules.json`).
    *   A log file (`anonymization.log`) detailing the process.
*   **Single-Model Usage:** Creates one anonymized Excel file (e.g., `your_excel_file_anonymized_<timestamp>.xlsx`) and one log file (`anonymization.log`) in the results directory corresponding to the default model used by the pipeline.

## Citation

If you use this code in your research, please cite the original paper:

McIntosh, F., Murina, S., Chen, L., Vargas, H. A., & Becker, A. S. (2025). Keeping Private Patient Data Off the Cloud: A Comparison of Local LLMs for Anonymizing Radiology Reports. *European Journal of Radiology Artificial Intelligence*, 100020. https://doi.org/10.1016/j.ejrai.2025.100020

You can also use the following BibTeX entry:

```bibtex
@article{mcintosh2025keeping,
  title={Keeping Private Patient Data Off the Cloud: A Comparison of Local LLMs for Anonymizing Radiology Reports},
  author={McIntosh, Fiona and Murina, Sofya and Chen, Luoyao and Vargas, H Alberto and Becker, Anton S},
  journal={European Journal of Radiology Artificial Intelligence},
  pages={100020},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 