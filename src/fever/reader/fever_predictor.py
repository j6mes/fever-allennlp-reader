import json

from allennlp.common import JsonDict
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
from typing import Dict, Union, Iterable, List, Tuple, Optional

from allennlp.data import Tokenizer, TokenIndexer, Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

from fever.reader.document_database import FEVERDocumentDatabase
from fever.reader.preprocessing import FEVERInstanceGenerator, ConcatenateEvidence
from fever.reader.simple_random import SimpleRandom

import numpy as np

@Predictor.register("fever")
class FEVERPredictor(Predictor):
    
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        claim_id: int = json_dict['id'] if "id" in json_dict else None
        claim: str = json_dict['claim']
        label: str = json_dict['label'] if 'label' in json_dict else None
        evidence: List[Tuple[str, int]] = json_dict['predicted_sentences']
        evidence: List[List[Tuple[str,int]]] = [[(None, item[0],item[1]) for item in evidence]]

        generated = self._dataset_reader._instance_generator.generate_instances(self._dataset_reader, evidence, claim)[0]

        return self._dataset_reader.text_to_instance(claim_id, None, generated["evidence"], claim, label)

    def predict(self, json_line:str) -> JsonDict:
        return self.predict_json(json.loads(json_line))

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        out_dict = {}
        if "label_logits" in outputs:
            out_dict["label_logits"] = outputs["label_logits"]
            out_dict["predicted_label"] = self._model.vocab.get_token_from_index(np.argmax(outputs["label_logits"]))

        if "label_probs" in outputs:
            out_dict["label_probs"] = outputs["label_probs"]
            out_dict["predicted_label"] = self._model.vocab.get_token_from_index(np.argmax(outputs["label_probs"]))

        return json.dumps(out_dict) + "\n"