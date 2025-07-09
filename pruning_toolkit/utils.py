# pruning_toolkit/utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig # Use AutoModel for generality
import logging
import numpy as np
import matplotlib.pyplot as plt

# Try importing IndicProcessor conditionally, as it might not be needed for all models
try:
    from IndicTransToolkit.processor import IndicProcessor
    INDIC_PROCESSOR_AVAILABLE = True
except ImportError:
    IndicProcessor = None
    INDIC_PROCESSOR_AVAILABLE = False
    logging.getLogger(__name__).warning("IndicTransToolkit not available. Preprocessing for Indic languages will be skipped.")


logger = logging.getLogger(__name__)

try:
    import bitsandbytes.nn as bnb_nn
    BNB_LINEARS = (bnb_nn.Linear4bit, bnb_nn.Linear8bitLt)
    logger.info("BitsAndBytes detected for quantized linear layers.")
except ImportError:
    BNB_LINEARS = ()
    logger.warning("BitsAndBytes not available. Will use only nn.Linear for score collection.")

# This tuple defines which layer types will be targeted for score collection.
# By default, it includes standard Linear layers and BitsAndBytes quantized Linear layers.
# You can customize this list if you want to target other specific layer types.
# For example, to only target torch.nn.Linear layers: TARGET_LINEARS = (torch.nn.Linear,)
TARGET_LINEARS = (torch.nn.Linear,) + BNB_LINEARS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

class CalibrationDataset(Dataset):
    """
    Dataset for calibration, handling text preprocessing and tokenization.
    Supports optional IndicProcessor for language-specific preprocessing.
    """
    def __init__(self, tokenizer, sentences, ip=None, src_lang=None, tgt_lang=None, max_length=128):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.ip = ip
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        logger.info(f"CalibrationDataset initialized with {len(sentences)} sentences.")
        if self.ip and self.src_lang and self.tgt_lang:
            logger.info(f"Using IndicProcessor for {src_lang} to {tgt_lang} preprocessing.")
        else:
            logger.info("No IndicProcessor provided or language parameters missing; sentences will be tokenized directly.")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        if self.ip and self.src_lang and self.tgt_lang:
            # Preprocess if IndicProcessor is available and language parameters are provided
            processed_text = self.ip.preprocess_batch([text], self.src_lang, self.tgt_lang)
            # preprocess_batch returns a list of preprocessed strings, we take the first
            text = processed_text[0]
        
        inputs = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

def make_pruning_hook(score_dict, param_name):
    """
    Factory function to create a forward hook for collecting activation-weighted scores.
    This hook is attached to linear layers to monitor their inputs (activations)
    and output weight-activation scores.
    """
    def hook(module, inputs, outputs):
        W = module.weight
        # Only consider 2D weights (typical for linear layers) and layers with more than 1 output feature
        if W.dim() != 2 or W.shape[1] <= 1:
            return

        A = inputs[0] # The input activation to the linear layer
        # Calculate the mean of absolute activations along all dimensions except the last one (feature dimension)
        # This gives a single value per input feature.
        A_mean = A.abs().mean(dim=list(range(A.dim()-1)))
        
        # Ensure the activation mean has the same number of elements as the input features of the weight matrix
        if A_mean.shape[0] != W.shape[1]:
            logger.debug(f"[Hook Skip] Shape mismatch for {param_name}: A_mean {A_mean.shape} vs W in_dim {W.shape[1]}. This layer's inputs might not be suitable for this type of scoring.")
            return

        # Calculate the score: Absolute Weight * Absolute Mean Activation
        # Unsqueeze A_mean to allow broadcasting for element-wise multiplication with W
        score = W.abs().cpu() * A_mean.cpu().unsqueeze(0)
        score_dict[param_name] += score
        logger.debug(f"Hook fired for {param_name}. Score added. Current total for this param: {score_dict[param_name].sum().item():.2f}")
    return hook

def plot_score_distribution(all_scores, save_path="score_distribution_histogram.png"):
    """
    Generates and saves a histogram of the score distribution.
    """
    try:
        if len(all_scores) == 0:
            logger.warning("No scores available for plotting histogram.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=50, density=True, alpha=0.7, color='blue', log=True)
        plt.title('Distribution of Activation-Weighted Scores (Log Scale)')
        plt.xlabel('Score Value (Absolute Activation * Absolute Weight)')
        plt.ylabel('Density (Log Scale)')
        plt.grid(True, which="both", ls="--", c='0.7')
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Score distribution histogram saved to '{save_path}'")
        plt.close() # Close the plot to free memory
    except Exception as e:
        logger.error(f"Could not generate histogram plot: {e}. Make sure matplotlib is installed and scores are valid.")