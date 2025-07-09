# pruning_toolkit/calibrator.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from .utils import (
    CalibrationDataset, make_pruning_hook, plot_score_distribution,
    TARGET_LINEARS, DEVICE, INDIC_PROCESSOR_AVAILABLE, IndicProcessor
)
from .config import PruningConfig

logger = logging.getLogger(__name__)

class ModelPruningCalibrator:
    """
    A class to calibrate a pre-trained model for pruning.
    It collects activation-weighted scores for linear layers.
    Supports both Sequence-to-Sequence and Causal Language Models.
    """
    def __init__(self, model_name: str, model_type: str = "seq2seq",
                 config_path: str = None, cache_dir: str = None,
                 target_layers_override: tuple = None): # New parameter for targeting layers
        """
        Initializes the ModelPruningCalibrator.

        Args:
            model_name (str): Hugging Face model identifier (e.g., "ai4bharat/indictrans2-indic-indic-dist-320M").
            model_type (str): Type of the model ("seq2seq" or "causal"). Defaults to "seq2seq".
            config_path (str, optional): Path to a JSON configuration file. Defaults to None.
            cache_dir (str, optional): Directory to cache Hugging Face models. Defaults to None.
            target_layers_override (tuple, optional): A tuple of torch.nn.Module types to target for scoring.
                                                      If None, uses the default TARGET_LINEARS from utils.py.
                                                      Example: (torch.nn.Linear, bnb_nn.Linear4bit)
        """
        if model_type not in ["seq2seq", "causal"]:
            raise ValueError(f"Invalid model_type: '{model_type}'. Must be 'seq2seq' or 'causal'.")

        self.model_name = model_name
        self.model_type = model_type
        self.cache_dir = cache_dir
        
        # Load configuration from provided path or create an empty one
        self.config = PruningConfig(config_path)

        # Set target layers, allowing override
        self.target_layers = target_layers_override if target_layers_override is not None else TARGET_LINEARS
        logger.info(f"Targeting layers of type: {[cls.__name__ for cls in self.target_layers]}")


        self.tokenizer = None
        self.model = None
        self.ip = None 
        self.score_dict = {}
        self.total_score_params = 0

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Loads the model and tokenizer based on model_type and configuration."""
        logger.info(f"Loading {self.model_type} model: {self.model_name}")
        
        tokenizer_args = {"trust_remote_code": True}
        model_args = {"device_map": "auto", "trust_remote_code": True}

        if self.cache_dir:
            tokenizer_args["cache_dir"] = self.cache_dir
            model_args["cache_dir"] = self.cache_dir

        # Quantization configuration from config.py (nested keys)
        load_in_4bit = self.config.get("quantization.load_in_4bit", True)
        bnb_4bit_quant_type = self.config.get("quantization.bnb_4bit_quant_type", "nf4")
        bnb_4bit_compute_dtype_str = self.config.get("quantization.bnb_4bit_compute_dtype", "bfloat16")
        bnb_4bit_use_double_quant = self.config.get("quantization.bnb_4bit_use_double_quant", True)
        llm_int8_threshold = self.config.get("quantization.llm_int8_threshold", 6.0)
        llm_int8_has_fp16_weight = self.config.get("quantization.llm_int8_has_fp16_weight", False)
        llm_int8_skip_modules = self.config.get("quantization.llm_int8_skip_modules", None)
        load_in_8bit = self.config.get("quantization.load_in_8bit", False)


        quant_config = None
        if load_in_4bit or load_in_8bit:
            bnb_compute_dtype = torch.bfloat16 if bnb_4bit_compute_dtype_str == "bfloat16" else torch.float16

            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                load_in_8bit=load_in_8bit,
                llm_int8_threshold=llm_int8_threshold,
                llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
                llm_int8_skip_modules=llm_int8_skip_modules,
            )
            model_args["quantization_config"] = quant_config
            logger.info(f"Model will be loaded with quantization config: {quant_config}")
        else:
            logger.info("Model will be loaded without quantization.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_args)
        
        if self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_args)
            if INDIC_PROCESSOR_AVAILABLE:
                self.ip = IndicProcessor(inference=True)
                logger.info("IndicProcessor initialized for Seq2Seq model.")
            else:
                logger.warning("IndicTransToolkit not available, cannot perform language-specific preprocessing.")
        elif self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_args)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning(f"Tokenizer for causal model '{self.model_name}' has no pad token. Setting pad_token to eos_token: '{self.tokenizer.eos_token}'")
        else:
            # Fallback for other model types if AutoModel can handle them, though not explicitly supported for calibration logic
            self.model = AutoModel.from_pretrained(self.model_name, **model_args)
            logger.warning(f"Loading model as generic AutoModel. Ensure its forward pass is compatible with calibration inputs.")


        self.model.eval()
        logger.info(f"{self.model_type.capitalize()} model and tokenizer loaded successfully.")

    def _initialize_score_dict(self):
        """
        Initializes the score dictionary for relevant linear layers based on `self.target_layers`.
        """
        self.score_dict = {}
        logger.info(f"Building score dictionary for target layers: {[cls.__name__ for cls in self.target_layers]}...")
        for name, mod in self.model.named_modules():
            # Check if the module is an instance of any class in self.target_layers
            if isinstance(mod, self.target_layers) and hasattr(mod, "weight"):
                W = mod.weight
                if W.dim() == 2 and W.shape[1] > 1: # Ensure it's a typical linear layer weight
                    # Initialize with float32 on CPU to avoid device issues during accumulation
                    self.score_dict[f"{name}.weight"] = torch.zeros_like(W, dtype=torch.float32, device="cpu")
                    logger.debug(f"  [+] Tracking {name}: {W.shape}")
                else:
                    logger.debug(f"  [-] Skip {name} (not a 2D weight or input dim <= 1): {getattr(W, 'shape', 'N/A')}")
            else:
                logger.debug(f"  [-] Skip {name} (not in target layer types).")

        self.total_score_params = sum(s.numel() for s in self.score_dict.values())
        if self.total_score_params == 0:
            logger.warning("No parameters found in target layers for scoring. Check 'target_layers_override' or model architecture.")
        else:
            logger.info(f"Total linear modules to score: {len(self.score_dict)}")
            logger.info(f"Total parameters collected for scoring analysis: {self.total_score_params:,}")


    def calibrate(self, calibration_sentences: list, batch_size: int = 1, 
                  save_scores_path: str = "activation_scores.pt",
                  src_lang: str = None, tgt_lang: str = None,
                  max_length: int = 128):
        """
        Performs the calibration run to collect activation-weighted scores.

        Args:
            calibration_sentences (list): A list of sentences to use for calibration.
            batch_size (int): Batch size for the DataLoader.
            save_scores_path (str): Path to save the collected scores.
            src_lang (str, optional): Source language for IndicProcessor (only for seq2seq). Defaults to None.
            tgt_lang (str, optional): Target language for IndicProcessor (only for seq2seq). Defaults to None.
            max_length (int): Max tokenization length for inputs.
        """
        if not calibration_sentences:
            logger.warning("No calibration sentences provided. Calibration skipped.")
            return

        self._initialize_score_dict()

        dataset = CalibrationDataset(
            tokenizer=self.tokenizer,
            sentences=calibration_sentences,
            # Pass IndicProcessor only if model is seq2seq AND IndicProcessor is available
            ip=self.ip if self.model_type == "seq2seq" and INDIC_PROCESSOR_AVAILABLE else None,
            src_lang=src_lang if self.model_type == "seq2seq" else None,
            tgt_lang=tgt_lang if self.model_type == "seq2seq" else None,
            max_length=max_length
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        # Register hooks
        handles = []
        for name, mod in self.model.named_modules():
            pname = f"{name}.weight"
            if pname in self.score_dict: # Only register hooks for parameters we initialized in score_dict
                handles.append(mod.register_forward_hook(make_pruning_hook(self.score_dict, pname)))
        logger.info(f"Registered {len(handles)} forward hooks on relevant layers.")
        if len(handles) == 0 and self.total_score_params > 0:
            logger.warning("No hooks were registered despite parameters being initialized for scoring. This might indicate an issue with layer type matching or the model structure.")


        # Calibration run
        logger.info(f"Starting calibration pass with {len(calibration_sentences)} sentences (batch size: {batch_size})...")
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_on_device = {k: v.to(DEVICE) for k, v in batch.items()}
                
                # Model-specific forward pass
                try:
                    if self.model_type == "seq2seq":
                        # For Seq2Seq, we need to provide decoder_input_ids to activate the decoder
                        # A common practice for calibration is to use the encoder inputs as decoder inputs
                        # to ensure all parts of the model are activated.
                        batch_on_device["decoder_input_ids"] = batch_on_device["input_ids"]
                        _ = self.model(**batch_on_device)
                    elif self.model_type == "causal":
                        # Causal models only take input_ids
                        _ = self.model(input_ids=batch_on_device["input_ids"], attention_mask=batch_on_device["attention_mask"])
                    else:
                        # Generic AutoModel or other types
                        _ = self.model(**batch_on_device)
                except Exception as e:
                    logger.error(f"Error during forward pass for batch {i}: {e}. Make sure the model type is correct and inputs are suitable.")
                    for h in handles: h.remove() # Clean up hooks on error
                    raise
        logger.info("Calibration done.")

        # Clean up hooks
        for h in handles: h.remove()
        logger.info("Hooks removed.")

        # Save scores
        if save_scores_path:
            torch.save(self.score_dict, save_scores_path)
            logger.info(f"Scores saved to {save_scores_path}")

    def analyze_pruning_efficiency(self, plot_save_path: str = "score_distribution_histogram.png") -> dict:
        """
        Analyzes the collected scores to simulate pruning efficiency.

        Args:
            plot_save_path (str): Path to save the score distribution plot.

        Returns:
            dict: A dictionary containing pruning simulation results.
        """
        if not self.score_dict:
            logger.warning("No scores collected. Run calibration first.")
            return {}

        logger.info("\n--- Pruning Efficiency Analysis ---")

        all_scores_list = [s.flatten().numpy() for s in self.score_dict.values()]
        if not all_scores_list:
            logger.warning("No scores available for analysis.")
            return {}

        all_scores = np.concatenate(all_scores_list)
        
        if self.total_score_params == 0:
            logger.info("No parameters scored. Cannot perform analysis.")
            return {}

        # Basic Statistics
        logger.info(f"Score Statistics (Abs Mean * Abs Weight):")
        logger.info(f"  Min: {np.min(all_scores):.6f}")
        logger.info(f"  Max: {np.max(all_scores):.6f}")
        logger.info(f"  Mean: {np.mean(all_scores):.6f}")
        logger.info(f"  Median: {np.median(all_scores):.6f}")
        logger.info(f"  Std Dev: {np.std(all_scores):.6f}")
        logger.info(f"  Sum of all scores: {np.sum(all_scores):.2f}")

        # Simulate Pruning Thresholds
        logger.info("\nSimulating pruning at various percentages:")
        sorted_scores = np.sort(all_scores)

        # Use percentages from config, with a default if not specified
        pruning_percentages = self.config.get("pruning_percentages", [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98])
        analysis_results = {}

        for p_percent in pruning_percentages:
            if p_percent < 0 or p_percent > 100:
                logger.warning(f"Invalid pruning percentage: {p_percent}. Skipping.")
                continue

            # Calculate the index for the threshold. Ensure it's within bounds.
            threshold_idx = min(int(self.total_score_params * (p_percent / 100.0)), self.total_score_params - 1)
            pruning_threshold = sorted_scores[threshold_idx]

            # Count parameters that are less than or equal to the threshold
            num_pruned_params = np.sum(all_scores <= pruning_threshold)
            num_remaining_params = self.total_score_params - num_pruned_params
            
            percent_removed = (num_pruned_params / self.total_score_params) * 100 if self.total_score_params > 0 else 0
            percent_remaining = (num_remaining_params / self.total_score_params) * 100 if self.total_score_params > 0 else 0

            logger.info(f"  Pruning {p_percent}% (threshold score <= {pruning_threshold:.6f}):")
            logger.info(f"    Parameters removed: {num_pruned_params:,} ({percent_removed:.2f}%)")
            logger.info(f"    Parameters remaining: {num_remaining_params:,} ({percent_remaining:.2f}%)")
            
            analysis_results[p_percent] = {
                "threshold_score": float(pruning_threshold),
                "params_removed": int(num_pruned_params),
                "percent_removed": float(percent_removed),
                "params_remaining": int(num_remaining_params),
                "percent_remaining": float(percent_remaining)
            }
        
        # Plot distribution based on config
        if self.config.get("plot_distribution", True):
            plot_score_distribution(all_scores, save_path=plot_save_path)

        non_zero_scored_modules = sum(1 for s in self.score_dict.values() if s.sum().item() > 0)
        logger.info(f"\nSummary: Modules with non-zero collected scores (received active inputs): {non_zero_scored_modules}/{len(self.score_dict)}")
        
        return analysis_results

    def get_raw_scores(self):
        """Returns the raw collected scores."""
        return self.score_dict