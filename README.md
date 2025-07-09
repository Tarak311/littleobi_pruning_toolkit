# Model Pruning Calibration Toolkit

This toolkit provides a modular and configurable way to calibrate pre-trained Hugging Face transformer models for pruning. It supports both sequence-to-sequence (Seq2Seq) and causal language models, allowing for various quantization options and analysis of activation-weighted scores to identify prune-ready parameters.

## Features

- **Modular Design:** Organized into a Python package (`pruning_toolkit/`) for reusability.
- **Configurable:** All parameters are loaded from JSON configuration files (`.json`), making it easy to switch models, quantization settings, and calibration data without code changes.
- **Model Type Support:** Handles both `AutoModelForSeq2SeqLM` (e.g., IndicTrans2) and `AutoModelForCausalLM` (e.g., GPT-2).
- **Quantization Options:** Supports `bitsandbytes` 4-bit and 8-bit quantization configurations.
- **Activation-Weighted Scoring:** Implements a forward hook mechanism to collect scores based on the product of absolute weights and mean absolute activations.
- **Pruning Efficiency Analysis:** Simulates pruning at various sparsity percentages and reports the number of parameters removed/remaining.
- **Score Distribution Plotting:** Generates histograms of collected scores to visualize their distribution.
- **Flexible Data Input:** Calibration sentences are loaded from external JSON files.
- **Target Layer Selection:** By default, targets `torch.nn.Linear` and `bitsandbytes.nn.Linear*` layers, but can be extended.

## Project Structure



pruning_toolkit_project/
├── pruning_toolkit/
│   ├── init.py           # Initializes the package and sets up basic logging.
│   ├── calibrator.py         # Main class for model loading, calibration, and analysis.
│   ├── config.py             # Handles loading and accessing configuration from JSON files.
│   ├── utils.py              # Contains utility functions, CalibrationDataset, and hook logic.
│   └── data/                 # Directory for calibration data files.
│       └── calibration_data_seq2seq.json
│       └── calibration_data_causal.json
├── config_seq2seq.json       # Example configuration for a Seq2Seq model.
├── config_causal.json        # Example configuration for a Causal model.
├── main_calibrate.py         # Entry point script to run calibration using config files.
└── README.md                 # This documentation file.


## Setup Instructions

### 1. Clone/Create the Project

If you haven't already, create the `pruning_toolkit_project` directory and the internal structure as described above. Copy the provided code snippets into their respective files.

### 2. Install Dependencies

Navigate to the `pruning_toolkit_project` directory and install the necessary Python packages. It's highly recommended to use a virtual environment.

```bash
cd pruning_toolkit_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch transformers accelerate bitsandbytes matplotlib numpy IndicTransToolkit











