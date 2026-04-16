"""Entrypoint to train/evaluate soft-prompt entity typing."""

from argparse import ArgumentParser
import tempfile
import logging
import os
import tomllib

import mlflow
import torch

from owner.training.entity_typing_soft_prompt import SoftPromptEntityTypingTrainer
from owner.utils.mlflow import absolutify, log_config
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("mlflow")


def main(config: dict):
    """Run soft-prompt entity typing train/eval cycle."""
    save_state = config.get("save_state", "none")
    save_path = config.get("save_path", None)

    model = SoftPromptEntityTypingTrainer(config)

    logger.info("Loading and preprocessing data")
    if save_state == "load_finetuned":
        model.load_data(False)
    else:
        model.load_data(True)

    torch.manual_seed(config["seed"])

    if save_state == "load_finetuned":
        logger.info('>>> Loading model from "%s" <<<', save_path)
        model.load_model(save_path)
    else:
        logger.info(">>> Training model <<<")
        model.train()

    logger.info(">>> Evaluating model <<<")
    model.evaluate()

    if save_state == "save_finetuned":
        logger.info('>>> Saving model to "%s" <<<', save_path)
        model.save_model(save_path)


if __name__ == "__main__":
    parser = ArgumentParser("SoftPromptEntityTyping")
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, "rb") as file:
        toml_config = tomllib.load(file)
        absolutify(toml_config)
        log_config(toml_config)
    mlflow.log_artifact(args.config_file, "")

    with tempfile.TemporaryDirectory() as tmpdir_name:
        os.chdir(tmpdir_name)
        main(toml_config)

