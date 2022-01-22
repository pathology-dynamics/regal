from enum import Enum
import torch
import numpy as np
from abc import abstractmethod
from snorkel.labeling import labeling_function
from snorkel_utils import make_keyword_lf

from typing import Union, List

class RuleType(Enum):
    KEYWORD = 1
    DEP_TREE = 2


class Rule:
    """
    Base Class for all types of rules
    """
    def __init__(
            self,
            rule_type: RuleType,
            label: int,
    ):
        """ Minimum info for initializing a rule and for the functionality of a
        single rule

        Parameters
        ----------
        rule_type: the type of rule it is using (more for UI/documentation use)
        label: the label of the class it is
        """
        self.rule_type = rule_type
        self.class_label = label

        # a snorkel labeling function
        self.lf = None
        # a list to store some positive instances to display user
        self.pos_instances = []

    @abstractmethod
    def create_mlm_mask(self, seq):
        """
        Used to create a mlm mask for training
        Parameters
        ----------
        seq: the sequence given by the dataloader

        Returns
        -------
        mlm_mask: the masked language modeling mask for training
        """
        pass

    @abstractmethod
    def str_rep(self):
        """

        Returns
        -------
        str_rep: A string representation of the rule

        """
        pass

    def __str__(self):
        return self.str_rep()

class KeywordRule(Rule):
    """
    Class for rules that classify based on keyword
    """
    def __init__(
            self,
            rule_type: RuleType,
            label: str,
            keywords: Union[List[str], str],
    ):
        super(KeywordRule, self).__init__(rule_type, label)

        # Question: Do we want a KeywordRule object for a collection of keywords?
        # Note: The current implementation accepts creating multiple keyword lfs
        self.keywords = set(keywords)

        # creating snorkel lf with the snorkel_utils.py helper function
        self.lf = make_keyword_lf(keywords,label)

    def create_mlm_mask(self, seq) -> torch.Tensor:
        """
        MLM mask with 1's in the positions of the keywords of the rule in the
        sequence

        Parameters
        ----------
        seq: individual sequence of tokens given by the dataloader.

        Returns
        -------
        token_mask: MLM mask described above
        """
        tokens_mask = torch.tensor([w in self.keywords for w in seq]).bool()
        if torch.any(tokens_mask):
            self.pos_instances.append(str(seq))
        return tokens_mask

    def str_rep(self):
        return f"Keyword Rule using: {self.keywords}. Example: {self.pos_instances[0]}"


class RuleList:

    def __init__(self, cache_limit=1000, promotion_thres=20):
        self.all_rules = []
        self.cache_limit = cache_limit
        self.promotion_thres = promotion_thres
        self.ranking_metrics = {'labeled_score':''}

    def cache_new_rules(self, clean_input_ids, labels):
        """ given sequence batches, add new possible rules (multitoken ones) to the cache
        Also, promote rules that have been seen more than the promotion threshold to be
        shown to the user (if they have a good confidence score)

        Parameters
        ----------
        clean_input_ids: batch input ids for the words in a sequence
        labels: the labels for each sequence
        """
        raise NotImplementedError("See TODO in rpn.py")

    def propose_multi_token_rules(self, weighted_sums, potential_rules):
        """ given the current single token potential rules and the weighted sums
        of each token, see which multi token rules are best and return the updated
        potential rules

        Parameters
        ----------
        weighted_sums: torch.tensor (self.vocab_size x n_classes)
                Tensor of weights corresponding to how relevant each word is to each class.
                Output of get_weighted_sums.
        potential_rules: dict
                Dictionary of potential single token rules, sorted by individual class
                Each rule is a tuple of (rule_keyword, confidence_score)

        Returns
        -------
        proposed_rules: dict
                Dictionary of potential rules, sorted by individual class
                Each rule is a tuple of (rule description, confidence_score)
        """
        raise NotImplementedError("See TODO in rpn.py")

    # TODO: with this class, implementing the method to get the statistics of a
    #       rule should be more straightforward
