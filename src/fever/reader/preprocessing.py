from typing import List, Tuple

from allennlp.common import Registrable
from overrides import overrides

class FEVERInstanceGenerator(Registrable):

    def generate_instances(self, reader, evidence:List[List[Tuple[str, int]]], claim:str):
        raise NotImplemented("This preprocessing function should be implemented")


@FEVERInstanceGenerator.register("concatenate")
class ConcatenateEvidence(FEVERInstanceGenerator):

    @staticmethod
    def _flatten(l):
        return [item for sublist in l for item in sublist]

    @overrides
    def generate_instances(self, reader, evidence:List[List[Tuple[str, int]]], claim:str):
        evidence_text: List[List[str]] = [[reader.get_doc_line(item[0],item[1]) for item in group] for group in evidence]
        flat_evidence_text: List[str] = self._flatten(evidence_text)

        # Key sets in dictionaries preserve insert order whereas python sets do not
        evidence_dict = dict()
        for item in flat_evidence_text:
            evidence_dict[item] = 1
        evidence = " ".join(evidence_dict.keys())
        return [{"evidence":evidence, "claim":claim}]

