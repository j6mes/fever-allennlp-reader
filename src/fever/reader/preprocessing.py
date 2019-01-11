from typing import List, Tuple

from allennlp.common import Registrable
from overrides import overrides


class FEVERPreprocessing(Registrable):

    def preprocess(self, evidence:List[List[str]], claim:str)->Tuple[str,str]:
        raise NotImplemented("This preprocessing function should be implemented")


@FEVERPreprocessing.register("concatenate")
class ConcatenateEvidence(FEVERPreprocessing):

    @staticmethod
    def _flatten(l):
        return [item for sublist in l for item in sublist]

    @overrides
    def preprocess(self, evidence:List[List[str]], claim:str)->Tuple[str,str]:
        items = self._flatten(evidence)

        # Key sets in dictionaries preserve insert order whereas python sets do not
        evidence = dict()
        for item in items:
            evidence[item] = 1
        evidence = " ".join(evidence.keys())
        return evidence, claim

