'''
Rule proposal module for ReGAL
'''
# Imports
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
import pickle
import logging

# Submodules
from torch import nn
from tqdm.auto import tqdm, trange
from texttable import Texttable
from abc import abstractmethod
from pytorch_lightning.callbacks import Callback

# Set up logging
logger = logging.getLogger('__file__')


# Rule Generator
class RuleGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def extract_patterns(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    # TODO: implement this method + self.rules.cache_limit and self.rules.new_rules and self.promotion_thres
    # Basically new proposed rules will go to cache and be promoted to new_rules once
    # We have a good enough count
    # self.rules.cache_new_rules(clean_input_ids, labels)


class AttentionKeywordRuleGenerator(RuleGenerator):
    def __init__(self):
        pass


class GradientKeywordRuleGenerator(RuleGenerator):
    pass


class NgramRuleGenerator(RuleGenerator):
    pass

# TODO: implement RuleList to add the multi_token_rules according to the weights


# Rule Ranker Class
class RuleRanker():
    def __init__(self,):
        self.ranking_metrics
        pass

    @abstractmethod
    def rank_rules(self,):
        pass


class Regal(pl.LightningModule):
    def __init__(self, args):
        # Get hyperparameters from argparse
        self.hparams = args

        # Rule dict
        # Each key in list is rule type and value is list of accepted rules of that type
        self.rule_dict = {
                'keyword': [],
                'phrase': [],
                'crowd_worker': []
                          }
        self.rule_proposers = {'keyword':...}


        # Labled samples
        # Samples correspond to each rule and are given at time of proposition
        self.labeled_samples = {}


        # Rule ranker


        # Model


        # Clustering of datapoints


        # Tokenizer


        # Data Module


    # Cache if if using with streamlit
    def _rule_proposal_inputs(self):
        '''
        Compute all inputs needed by each rule proposal module
        May involve pass through network
        '''
        # I'm that this may be best implemented as a set of pytorch-lightning callbacks
        pass


    def _tune_rule_metrics(self):
        '''
        Tune thresholds that balance between coverage, accuracy, specificity, etc
        during the rule generation process
        '''
        pass

    def _display_proposed_rules_summary(self):
        '''
        Display list of proposed rules for inspection
        '''
        pass

    def _display_single_rule_detailed(self, n=5):
        '''
        Display single rule with $n examples of matched instances and 
        statistics of coverage, agreement, 
        '''
        pass

    def _update_rules_and_labels(self):
        pass

    def propose_rules(self):
        '''
        Propose new rules for user evaluation
        Consists of 3 steps:
            1. Create rules from each generator
            2. Pool rules from all generators and rank 
            3. Allow user to choose which rules are good
        '''
        pass

    
    # Pytorch-Lightning specific functions
    @staticmethod
    def add_model_specific_args(parent_parser):
        '''
        Add arguments to argument parser specific to ReGAL model
        '''
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


    def forward(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.scheduler.OneCycleLR()
        return optimizer

    def training_step(self, batch, batch_idx):
        # inputs, ... = batch
        loss = loss(x, y)

        # Log training loss

        # Return loss and anything else necessary
        return loss


    def _shared_eval_step(self, batch):
        inputs = batch
        loss = loss(x, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch)

        # Log loss and other metrics

        # Return
        return loss

    def _eval_epoch_end(self, eval_step_outputs):
        '''
        Compute metrics of interest on eval set
        '''
        pass

    
        

#################
### Callbacks ###
#################

class KeywordRuleInputCallback(Callback):
    pass

class PhraseRuleInputCallback(Callback):
    pass

class DepencencyRuleInputCallback(Callback):
    pass