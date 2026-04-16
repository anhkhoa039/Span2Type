"""NER Trainer
"""
import logging
from typing import List
from ..data.model import MiniDocument
from ..data.serialization import from_owner
from ..evaluation.entity_typing import evaluate_entity_typing
from ..evaluation.base import convert_document_to_entities, \
    merge_dataset_with_predictions
from .base import BaseTrainer
from .mention_detection import MentionDetectionTrainer
from .entity_typing import EntityTypingTrainer
from .entity_typing_soft_prompt import SoftPromptEntityTypingTrainer

logger = logging.getLogger('mlflow')


class NerTrainer(BaseTrainer):
    """NER trainer
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.mention_detection = MentionDetectionTrainer(config)
        et_config = config.get('entity_typing', {})
        use_soft_prompt = et_config.get('use_soft_prompt', False)
        self.entity_typing = (
            SoftPromptEntityTypingTrainer(config)
            if use_soft_prompt else EntityTypingTrainer(config)
        )

    def load_data(self, training: bool = True):
        self.mention_detection.load_data(training)
        self.entity_typing.load_data(training)

    def train(self):
        self.mention_detection.train()
        self.entity_typing.train()

    def evaluate(self):
        self.mention_detection.evaluate()
        self.entity_typing.evaluate()

        logger.info('Evaluating end-to-end NER')

        # Ground truth
        test_dataset = self.mention_detection.test_dataset.dataset
        test_dataset_v2 = self.mention_detection.test_dataset_v2.dataset
        truth: List[MiniDocument] = []
        for document in test_dataset.documents:
            truth.append(convert_document_to_entities(document))

        # Predict entities
        pred_entities = self.mention_detection.predict_dataset(test_dataset)
        pred_entities_dataset = merge_dataset_with_predictions(
            test_dataset, test_dataset_v2.documents)

        # Predict entity types
        pred_entities = self.entity_typing.predict_dataset(
            pred_entities_dataset, 'test', prefix='ner')

        evaluate_entity_typing(
            truth, pred_entities, test_dataset.metadata.entity_types, None, 'test', None, 'ner')

    def save_model(self, folder: str):
        super().save_model(folder)
        self.mention_detection.save_model(folder)
        self.entity_typing.save_model(folder)

    def load_model(self, folder: str):
        self.mention_detection.load_model(folder)
        self.entity_typing.load_model(folder)
