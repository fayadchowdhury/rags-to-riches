from abc import ABC, abstractmethod
from typing import Dict

class BaseEvaluator(ABC):
    '''
    Abstract base class for evaluators
    '''

    def __init__(self, **kwargs):
        '''
        Initialize an evaluator with optional configuration parameters
        '''
        self.config = kwargs

    def precision_k(self, candidate_list: str, gold_list: list) -> float:
        '''
        Take a candidate list of retrieved documents and a gold list of relevant documents and k value
        Return precision@k
        '''
        deduped_candidate_list = set(candidate_list) if candidate_list and len(candidate_list) > 0 else set()
        deduped_gold_list = set(gold_list) if gold_list and len(gold_list) > 0 else set()
        intersection = deduped_gold_list.intersection(deduped_candidate_list)
        return len(intersection) / len(deduped_candidate_list) if len(deduped_candidate_list) > 0 else 0
    
    def recall_k(self, candidate_list: str, gold_list: list) -> float:
        '''
        Take a candidate list of retrieved documents and a gold list of relevant documents and k value
        Return recall@k
        '''
        deduped_candidate_list = set(candidate_list) if candidate_list and len(candidate_list) > 0 else set()
        deduped_gold_list = set(gold_list) if gold_list and len(gold_list) > 0 else set()
        intersection = deduped_gold_list.intersection(deduped_candidate_list)
        return len(intersection) / len(deduped_gold_list) if len(deduped_gold_list) else 0
    
    def f1_score(self, precision: float, recall: float) -> float:
        '''
        Take precision and recall values
        Return f1 score
        '''
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # Add more concrete methods to be shared between evaluators
    
    @abstractmethod
    def evaluate_answer(self, system_prompt: str, prompt: str, results: Dict) -> Dict:
        '''
        Take a system prompt, a chat prompt and a results dictionary to pass to LLM
        Return the output JSON
        '''
        pass