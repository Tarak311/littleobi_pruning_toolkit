# main_calibrate.py
import json
import os
from pruning_toolkit.calibrator import ModelPruningCalibrator
from pruning_toolkit.config import PruningConfig
from pruning_toolkit.utils import TARGET_LINEARS # Import default target layers for reference

import logging

# Configure root logger to see all messages from the library
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_full_path(relative_path):
    """Converts a relative path to an absolute path based on the script's directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

def calibrate_model_from_config(config_file: str):
    """
    Runs the pruning calibration process for a model specified by a configuration file.
    """
    logger.info(f"\n--- Starting calibration for configuration: {config_file} ---")

    try:
        # Load configuration
        main_config = PruningConfig(get_full_path(config_file))
        logger.info(f"Loaded configuration:\n{main_config}")

        # Get calibration data path from config
        calibration_data_path = main_config.get("calibration_data_path")
        if not calibration_data_path:
            logger.error("Calibration data path not specified in config. Cannot proceed.")
            return

        calibration_data_path_abs = get_full_path(calibration_data_path)

        # Load calibration sentences from JSON file
        calibration_sentences = []
        if not os.path.exists(calibration_data_path_abs):
            logger.error(f"Calibration data file not found at {calibration_data_path_abs}. Please ensure the path is correct.")
            return
        try:
            with open(calibration_data_path_abs, 'r', encoding='utf-8') as f:
                data = json.load(f)
                calibration_sentences = data.get("calibration_sentences", [])
            logger.info(f"Loaded {len(calibration_sentences)} calibration sentences from {calibration_data_path_abs}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from calibration data file {calibration_data_path_abs}: {e}")
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading calibration data from {calibration_data_path_abs}: {e}")
            return

        # Get calibrator parameters from config
        model_name = main_config.get("model_name")
        model_type = main_config.get("model_type", "seq2seq") # Default to seq2seq
        output_scores_path = get_full_path(main_config.get("output_scores_path", f"activation_scores_{os.path.basename(config_file)}.pt"))
        plot_save_path = get_full_path(main_config.get("plot_save_path", f"score_distribution_histogram_{os.path.basename(config_file)}.png"))
        batch_size = main_config.get("batch_size", 1)
        src_lang = main_config.get("calibration_src_lang")
        tgt_lang = main_config.get("calibration_tgt_lang")
        max_length = main_config.get("calibration_max_length", 128)
        
        # Handle target_layers option:
        # The target_layers field in config.json is illustrative.
        # To actually *change* the targeted layers from config, you'd need a more
        # sophisticated mechanism (e.g., mapping strings to class objects).
        # For now, it's set to null by default, meaning TARGET_LINEARS from utils.py is used.
        # If you wanted to override, you'd uncomment and modify this:
        # target_layers_override = (torch.nn.Linear,) # Example: only target standard Linear layers
        target_layers_override = None # Default: use TARGET_LINEARS from utils.py

        # Initialize calibrator
        calibrator = ModelPruningCalibrator(
            model_name=model_name,
            model_type=model_type,
            config_path=get_full_path(config_file), # Pass the specific config path
            target_layers_override=target_layers_override # Pass the override if applicable
        )

        # Perform calibration
        calibrator.calibrate(
            calibration_sentences=calibration_sentences,
            batch_size=batch_size,
            save_scores_path=output_scores_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=max_length
        )

        # Analyze pruning efficiency
        results = calibrator.analyze_pruning_efficiency(plot_save_path=plot_save_path)

        if results:
            logger.info("\n--- Pruning Analysis Summary ---")
            for p, res in results.items():
                logger.info(f"  {p}% Pruning: Removed {res['params_removed']:,} params ({res['percent_removed']:.2f}%), Remaining {res['params_remaining']:,} params ({res['percent_remaining']:.2f}%)")
        
        logger.info(f"--- Calibration for {config_file} finished successfully ---")

    except Exception as e:
        logger.error(f"An error occurred during calibration for {config_file}: {e}", exc_info=True)


if __name__ == "__main__":
    # Define the configuration files you want to run
    calibration_configs = [
        "config_seq2seq.json",
        #"config_causal.json"
        # Add more config files here to calibrate different models
        # "config_my_new_model.json",
    ]

    for config_file in calibration_configs:
        calibrate_model_from_config(config_file)