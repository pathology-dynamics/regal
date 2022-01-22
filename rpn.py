# Imports
import pandas as pd 
import numpy as np 
import ujson
import datetime
import pickle
import torch
import os
import logging
import pprint
import string
import copy
import time

# Library submodules
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from os.path import join as pathjoin
from tqdm import tqdm, trange
from collections import defaultdict
from transformers import BertModel, BertForMaskedLM
from texttable import Texttable
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from snorkel.labeling.model import LabelModel

# My submodules
from dataloader import RPNDataset, RPNCollate, prep_data
from train import train, evaluate, log_metrics, save_checkpoint
from data_utils import prepare_data, get_noisy_labels
from snorkel_utils import make_keyword_lf, make_token_lf
# from utils import get_model_info, get_tokenizer
from utils import get_model_and_tokenizer, get_model_output
from models import AssembleModel
from phrase_rules import get_word_scores, accumulate_phrase_scores


STOPWORDS = stopwords.words('english')

logger = logging.getLogger('__file__')


# Move to _display_proposed_rules_summary
def display_sample_rules(potential_rules, count_dict, k=3, class_names=None, ):

    '''
    Display sample rules for a user to verify his/her choice of alpha

    Inputs:
    --------------------
        potential_rules: dict
            List of potential rules returned from propose_single_token_rules
        
    Outputs:
    --------------------
        None
    '''    
    sample_list = [['Class', 'Rule', 'Score', 'Count']]

    if len(potential_rules) == 0:
        print("No rules to choose! Please press 'y' to continue.")
        return

    # Collect a sample rule from each class
    for j, (key, class_dict) in enumerate(potential_rules.items()):
        # Get class name if available
        # if 'name' in class_dict.keys():
        #     name = class_dict['name']
        if class_names is not None:
            name = class_names[key]
        else:
            name = key
        
        # Get top rule/score
        if len(class_dict['rules']) == 0:
            sample_list.append([name, "No rules above threshold", "N/A"])
        else:
            sorted_rules = sorted(class_dict['rules'], key=lambda x: x[1])[::-1]
            sample_list.extend([[name, f'HAS({rule}) -> {name}', score, count_dict[rule]] for (rule, score) in sorted_rules[:k]])

    # Display rules in pretty way
    t = Texttable()
    t.add_rows(sample_list)
    print(t.draw())



class RuleProposer():
    # Move appropriate items to __init__ of Regal class
    def __init__(self, args, model, tokenizer, cuda=False, init_alpha=.7):
        '''
        Initialize RPN
        '''
        self.args = args 
        self.model = model
        self.tokenizer = tokenizer
        self.word2id = tokenizer.vocab
        self.id2word = {val: key for key, val in self.word2id.items()}
        self.vocab_size = np.max(list(self.id2word.keys())) + 1
        self.cuda = cuda
        self.init_alpha = args.alpha
        self.n_classes = args.num_classes
        self.k = self.args.rules_per_epoch
        self.min_count_cutoff = args.min_count_cutoff
        self.polarity_thresh = args.polarity_thresh
        self.rules = []

        # Get mapping of words to stems
        self.stemmer = PorterStemmer()
        self.construct_token_stems()

        self.rpn_collate_fn = RPNCollate(self.tokenizer)
        self.weighted_phrase_sums = {}


        # Make label model
        self.label_model = LabelModel(cardinality=self.n_classes)
        self.min_lf = 1
        
        # Get place to save data
        date = str(datetime.date.today())
        save_dir = os.path.join('output', args.output_dataset, date)
        self.parent_save_dir = save_dir
        if not os.path.isdir(save_dir):
            save_dir = os.path.join(save_dir, '0')
            self.save_dir = save_dir
        else:
            subdirs = [int(d) for d in os.listdir(save_dir) if d.isnumeric()] + [-1]
            new_dir = str(max(subdirs) + 1)
            self.save_dir = os.path.join(save_dir, new_dir)    
        os.makedirs(self.save_dir)
        os.makedirs(pathjoin(self.save_dir, args.logdir))
        os.makedirs(pathjoin(self.save_dir, args.checkpoint_dir))

        # Save a model checkpoint for model refresh after training
        torch.save(self.model, pathjoin(self.save_dir, 'model.pt'))
        
        # Load data
        self.load_dataset(args.data_path)
        logger.debug(f"Used Keywords: {self.used_kwds}")
        self.n_rules = self.train.noisy_labels.size(1)
        logger.debug(f'RPN n_rules: {self.n_rules}')

        # Make dataloaders for deep learning
        self.make_dataloaders()


        # Type of used word list (i.e. is it words or tokens?)
        if len(self.used_kwds) > 0:
            used_word_type = type(list(self.used_kwds)[0])
        else:
            used_word_type = None

        # Get list of words to exclude
        self.words_to_exclude = set(STOPWORDS) | set(string.punctuation) | {'[CLS]', '[SEP]','[UNK]','[PAD]','[MASK]'}
        if used_word_type == str:
            self.words_to_exclude |= self.used_kwds

        # Precompute counts for faster training
        self.precompute_phrase_counts()

        # Get list of tokens to exclude
        self.tokens_to_exclude = {self.word2id[word] for word in self.words_to_exclude if word in self.word2id}
        if used_word_type == int:
            self.tokens_to_exclude |= self.used_kwds



    def precompute_phrase_counts(self):
        '''
        Precompute word counts for faster model training
        '''
        # phrase_counts = defaultdict(int)
        phrase_inds = defaultdict(set)
        normalized_text = []
        logger.info("Precomputing phrase counts")
        for j, word_list in enumerate(tqdm(self.train.word_lists)):
            normalized_text.append(" ".join(word_list))
            # normalized_text.append(self.tokenizer.decode(self.tokenizer.encode(word_list)[1:-1]))
            for l in range(1, 1 + self.args.max_rule_length):

                phrases = [" ".join(word_list[i:i+l]) for i in range(len(word_list) - l + 1)]
                for p in phrases:
                    if any([punct in p for punct in '.,!?"\\']):
                        continue
                    # phrase_counts[p] += 1
                    phrase_inds[p].add(j)

        self.train.text = normalized_text
        self.phrase_counts = {k:len(v) for k, v in phrase_inds.items() if len(v) >= self.min_count_cutoff and k not in self.words_to_exclude}
        logger.debug(f"Num Phrases: {len(self.phrase_counts)}")
        self.phrase_inds = {k:list(phrase_inds[k]) for k in self.phrase_counts.keys()}

    # def save_proposed_rules(self, epoch=0)

    def save_rules(self, epoch=0):
        '''
        Save model and constituent pieces to file in save_dir
        
        Things that get saved:
            * Model with ckpt
            * Labeling functions
        '''
        # Save dict of labeling functions
        rule_dict = defaultdict(list)
        for label, rule in self.rules:
            rule_dict[label].append(rule)

        self.rule_dict = rule_dict
        # logger.debug(f"Rule Dict: {rule_dict}")
        with open(os.path.join(self.save_dir, f'rules_{epoch}.json'), 'w') as f:
            ujson.dump(rule_dict, f)

    def save_df(self, df, rule_length, epoch):
        '''
        Save rule df during autoeval
        '''
        save_path = pathjoin(self.save_dir, 'rule_dfs')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        df.to_csv(pathjoin(save_path, f'rules_epoch{epoch}_len{rule_length}.tsv'), index=False, sep='\t')

    def save_scores(self):
        '''
        Save phrase scores to debug our model
        '''
        save_path = pathjoin(self.save_dir, 'debugging')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.multitoken_scores, pathjoin(save_path, 'scores.pt'))
        torch.save(self.multitoken_counts, pathjoin(save_path, 'counts.pt'))
        torch.save(self.id2phrase, pathjoin(save_path, 'id2phrase.pt'))
        torch.save(self.phrase_counts, pathjoin(save_path, 'phrase_counts.pt'))

    def refresh_model(self):
        logger.debug("***REFRESHING MODEL PARAMS***")
        model = torch.load(pathjoin(self.save_dir, 'model.pt'))
        if self.cuda:
            model = model.cuda()

        self.model = model
        return model


    def construct_token_stems(self):
        '''
        Get all tokens coming from same stem
        '''
        # Init dicts
        self.stem2token = defaultdict(list)
        self.stem2id = defaultdict(list)

        # Get mapping of stem to all of its tokens
        for word, id_val in self.word2id.items():
            # We only care if token starts a word
            if not word[0].isalpha():
                continue
            stem = self.stemmer.stem(word)
            self.stem2token[stem].append(word)
            self.stem2id[stem].append(id_val)


        # return stem2token, stem2id

    def find_similar_tokens(self, subword):
        '''
        Get tokens similar to subword (i.e. having the same stem)
        '''
        # Make sure first token letter of token is letter
        if not subword[0].isalpha():
            return []
        else:
            stem = self.stemmer.stem(subword)
            similar = self.stem2token[stem]
            return similar

    # # TODO: FINISH
    # def clarify_rule(self, rule_phrase, label):
    #     '''
    #     Clarify rules by mapping each word to particular tokens.  
    #     This creates rules that are more specific than substring matching.

    #     Inputs:
    #     -------
    #         rule_phrases: list of str
    #             Phrases corresponding to each accepted rule

    #         label:
    #             Label corresponding to rule

    #     Returns:
    #     --------
    #         Output dict with three values:
    #             * rule_tokens: list of str
    #                 Tokens to zero out and/or mask during future iterations
    #             * rule_phrases: list of str
    #                 Phrases to zero out when constructing future rules
    #             labeling_functions: list of lf
    #                 Labeling functions to be used in future iterations
    #     '''
    #     rule_tokens = []
    #     rule_phrases = []
    #     labeling_functions = []
    #     tokens = self.tokenizer.tokenize(rule_phrase)
    #     # Single token rule
    #     if len(tokens) == 1:
    #         # Find other tokens that start with same root
    #         # parent_token = 
    #         similar = self.find_similar_tokens(phrase)
    #         additions = [self.evaluate_token(phrase, t) for t in similar]


    #         # Find other subwords containing token, e.g. "unpleasant" for "pleasant"
    #         # similar.extend(self.tokens_containing_subword(phrase))

    #         # Make token-based keyword for rule
    #         if len(similar) == 0:
    #             return

    #     # Single word rule (but split into multiple subwords)
    #     elif len(phrase.split()) == 1:
    #         # See if any tokens start with same root
    #         similar = self.find_similar_tokens(phrase)

    #         # If so, create token-based rules


    #         # Otherwise, create substring-based rule


    #     # Multi-word rules are handled via substring matching
    #     else:
    #         # Create substring-based rule
    #         rule_phrases.append(phrase)
    #         labeling_functions.append(make_keyword_lf(phrase))

    #         # (Optional) Add stemming to improve coverage




    # def user_rule_clarify(self, starting_word, similar_tokens):
    #     pass


    def lookup_matching_examples(self, token=None, phrase=None, alg='random', n=5):
        '''
        Get matching examples for a particular rule

        Inputs:
        -------
            token: int
                Token to match in input_ids

            phrase: str
                Substring to match in text example

            alg: choice of ['random', 'unlabeled', 'conflicting', 'mixed'] (default: 'random')
                Algorithm used to select which examples to display
                * 'random':     Randomly chooses n matching examples
                * 'unlabeled':  Randomly chooses n matching examples from datapoints not 
                                currently matched by any labeling function
                * 'conflicting': Randomly chooses n matching examples from datapoints that have 
                                labeling functions
                * 'mixed':      Selects half of datapoints using 'unlabeled' stategy and half 
                                using 'conflicting' strategy

            n: int
                Number of examples to match
        '''
        if token is not None and phrase is not None:
            print("Cannot match with `token` and `phrase` simultaneously.  Defaulting to token.")

        elif token is None and phrase is None:
            raise ValueError("Must provide `token` or `phrase` on which to match!")

        # Token-based match
        if token is not None:
            examples, inds = self.train.token_match(token, alg=alg, n=n)

        # Phrase based
        elif phrase is not None:
            examples, inds = self.train.phrase_match(phrase, alg=alg, n=n)

        return examples, inds

    def _grab_true_label(self, ind):
        '''
        Solicit true label from an element at index [ind] of training data
        '''
        pass

    def update_true_label(self, ind, label):
        '''
        Give ground truth label to element at index [ind] of training data
        '''
        pass  
    
    # Move to data module
    def load_dataset(self, filepath):
        '''
        Load dataset with train, test, and validation dicts.  
        Should also contain keywords used in original rules
        '''
        output_path = os.path.join(self.parent_save_dir, filepath.split('/')[-1])
        data = torch.load(filepath)
        if os.path.isfile(output_path) and not self.args.refresh_data:
            cached_data = torch.load(output_path)
            train_dataset = cached_data['train']
            valid_dataset = cached_data['valid']
            test_dataset = cached_data['test']
            labeled = cached_data['labeled']

        else:
            # Load data splits
            train_dataset = RPNDataset(data['train'], 
                                self.tokenizer, 
                                rule_keywords=data['rule_keywords'], 
                                min_lf=self.min_lf)
            valid_dataset = RPNDataset(data['valid'], 
                                self.tokenizer, 
                                rule_keywords=data['rule_keywords'], 
                                min_lf=self.min_lf)
            test_dataset = RPNDataset(data['test'], 
                                self.tokenizer, 
                                rule_keywords=data['rule_keywords'], 
                                min_lf=self.min_lf)
            labeled = copy.deepcopy(train_dataset)
            labeled._use_labeled()

            # Save cached training data for future experiments
            data_dict = {
                'train': train_dataset,
                'valid': valid_dataset,
                'test': test_dataset,
                'labeled': labeled
                }
            torch.save(data_dict, output_path)


        # Train label model
        self.label_model.fit(labeled.noisy_labels)
        # self.label_model.fit(data['train']['noisy_labels'], class_balance=np.ones(self.n_classes)/self.n_classes)

        # Get soft labels
        train_dataset.soft_labels = self.label_model.predict_proba(train_dataset.noisy_labels)
        valid_dataset.soft_labels  = self.label_model.predict_proba(valid_dataset.noisy_labels)
        test_dataset.soft_labels = self.label_model.predict_proba(train_dataset.noisy_labels)
        labeled.soft_labels = self.label_model.predict_proba(labeled.noisy_labels)

        self.train = train_dataset
        self.valid = valid_dataset
        self.test = test_dataset
        self.labeled = labeled

            # # Update which data points are labeled
            # labeled, unlabeled = self.split_labeled()
            # self.labeled = labeled

        # TODO: Modernize for current version
        # This entails:
        #   * Masking whole words/phrases in dataloader
        # self.used_kwds = set(key_subwords)
        self.used_kwds = set([w for (k, v) in data['rule_keywords'].items() for w in v])
        logger.debug(f"used_kwds: {self.used_kwds}")
        self.class_names = data['class_names']
        # logger.info(f"Excluding the following keywords: {[self.word2id[i] for i in self.used_kwds]}")
        if 'lfs' in data:
            self.lfs = lfs
        else:
            self.lfs = []

            


        # TODO: Grab and clarify rules

        # # Add needed fields if missing
        # if ('encoded_text' not in data['train'].keys() 
        #     or 'attention_masks' not in data['train'].keys() 
        #     or 'word_starts' not in data['train'].keys()
        #     or 'word_ends' not in data['train'].keys()
        #     or 'word_lists' not in data['train'].keys()
        #     or 'soft_labels' not in data['train'].keys()
        #     ):
        #     for key in ['train', 'test','valid']:
        #         # Run one-time data prep
        #         input_ids, attention_masks, word_starts, word_ends = prep_data(data[key]['text'], self.tokenizer, max_length=self.args.max_len)

        #         soft_labels = torch.tensor(self.label_model.predict_proba(data[key]['noisy_labels']))

        #         # Grab appropriate fields
        #         data[key]['encoded_text'] = input_ids
        #         data[key]['attention_masks'] = attention_masks
        #         data[key]['word_starts'] = word_starts
        #         data[key]['word_ends'] = word_ends
        #         data[key]['soft_labels'] = soft_labels

        #         if 'labels' not in data.keys():
        #             data[key]['labels'] = data[key]['label']

        #         rpn_dataset = RPNDataset(data[key], self.tokenizer, min_lf=self.min_lf)
        #         data[key]['word_lists'] = rpn_dataset.word_lists
                

        #         # Remap noisy labels
        #     # Save updated data so we don't need to do this again
        #     torch.save(data, filepath)

        # self.train = data['train']
        # self.labeled = data['labeled']
        # self.test = data['test']
        # self.valid = data['valid']


        

    # Move to ruleKeyword module
    def tokens_containing_subword(self, subword):
        return [k for k in self.word2id.keys() if subword in k]

    # Move to data module
    def make_dataloaders(self):
        '''
        Make data loaders for training.  Also updates dataloaders after addition of new rules
        '''
        # Make datasets
        rpn_datasets = {}
        for data, name in zip([self.train, self.test, self.valid, self.labeled], ['train','test','valid', 'labeled']):
            # if 'labels' not in data.keys():
            #     data['labels'] = data['label']
            # if 'soft_labels' not in data.keys():
            data.soft_labels = torch.tensor(self.label_model.predict_proba(data.noisy_labels))
            # dataset = RPNDataset(data, self.used_kwds, self.tokenizer, min_lf=self.min_lf)
            # rpn_datasets[name] = dataset
        # self.rpn_datasets = rpn_datasets

        # Make dataloaders
        self.train_loader = DataLoader(self.train, 
                                        shuffle=True, 
                                        batch_size=self.args.batch_size, 
                                        num_workers=self.args.num_workers, 
                                        collate_fn=self.rpn_collate_fn)
        self.labeled_loader = DataLoader(self.labeled, 
                                        shuffle=True, 
                                        batch_size=self.args.batch_size, 
                                        num_workers=self.args.num_workers, 
                                        collate_fn=self.rpn_collate_fn)
        self.test_loader = DataLoader(self.test, 
                                        shuffle=True, 
                                        batch_size=self.args.batch_size, 
                                        num_workers=self.args.num_workers, 
                                        collate_fn=self.rpn_collate_fn)
        self.valid_loader = DataLoader(self.valid, 
                                        shuffle=True, 
                                        batch_size=self.args.batch_size, 
                                        num_workers=self.args.num_workers, 
                                        collate_fn=self.rpn_collate_fn)


    def split_labeled(self):
        '''
        Split data into labeled and unlabeled according to a particular rule-induced label set
        '''
        label_matrix = self.train.noisy_labels
        mask = ((torch.LongTensor(label_matrix) >= 0).sum(dim=1) >= self.min_lf).bool()
        # labeled = {}
        # unlabeled = {}
        # for key, val in self.train.items():
        #     if torch.is_tensor(val):
        #         labeled[key] = val[mask]
        #         unlabeled[key] = val[~mask]
        #     else:
        #         labeled[key] = [t for (i,t) in zip(mask, val) if i == 1]
        #         unlabeled[key] = [t for (i,t) in zip(mask, val) if i == 0]

        self.label_model.fit(self.train.noisy_labels)

        for data_slice in [self.train, self.valid, self.test]:
            soft_labels = torch.tensor(self.label_model.predict_proba(data_slice.noisy_labels))
            data_slice.soft_labels = soft_labels

        self.train.min_lf = self.min_lf
        self.labeled = copy.deepcopy(self.train)
        self.labeled._use_labeled()
        # return labeled, unlabeled

    # Move to _update_rules_and_labels function
    def update_noisy_labels(self, new_lfs):
        '''
        Add to noisy label tensor from new labeling functions
        '''
        logger.debug("\n\n***Updating Noisy Labels****\n\n")
        if len(new_lfs) == 0:
            logger.debug("**No New LFS**")
            return
        # Add new rules for each dataset
        for d in [self.train, self.test, self.valid]:
            new_labels = torch.LongTensor(get_noisy_labels(d.text, new_lfs))
            d.full_noisy_labels = torch.cat([d.full_noisy_labels, new_labels], dim=1)
            d.balance_noisy_labels()

        # Update which data points are labeled
        self.split_labeled()
        # labeled, unlabeled = self.split_labeled()
        # self.labeled = labeled

    def update_rule_phrase_kwds(self, rule_phrases, proposed_kwds=None):
        '''
        Add new keywords to list of keywords used in rules
        '''
        if proposed_kwds is None:
            logger.debug("No proposed kwds given!")
            proposed_kwds = rule_phrases
        
        words = [w for phrase in rule_phrases for w in self.tokenizer.tokenize(phrase)]
        self.used_kwds.update(words)
        self.tokens_to_exclude.update([self.word2id[w] for w in words])


    def get_token_scores(self, masked_attention_probs, noised_ids):
        masked_attention_probs = masked_attention_probs.reshape(-1, self.n_classes)
        flat_ids = noised_ids.flatten()

        # Get whole word scores

        # Create mask to exclude tokens already proposed in rules
        id_mask = torch.stack([(flat_ids == i) for i in self.tokens_to_exclude], dim=1)
        id_mask = ~(id_mask.sum(dim=1).bool())

        # Get filtered tokens with mask
        filtered_ids = flat_ids[id_mask]

        # Filter token scores by unproposed tokens
        probs_mask = id_mask.unsqueeze(-1) * torch.ones_like(masked_attention_probs, dtype=torch.bool)
        filtered_scores = masked_attention_probs[probs_mask].view(-1, self.n_classes)

        # Update token counts
        unique, value_counts = torch.unique(filtered_ids, return_counts=True)
        self.all_counts[unique] += value_counts 

        # Update prob sums
        for token_id in unique:
            token_mask = (filtered_ids == token_id)
            token_score = filtered_scores[token_mask].sum(axis = 0)
            self.all_scores[token_id, :] += token_score


    # Move to _rule_proposal_inputs
    # @torch.inference_mode()
    def get_weighted_token_vals(self):
        '''
        Get token attention for each token weighted by confidence of each class
        '''
        # Put model in eval mode for speed
        self.model.eval()

        # Token score lists we want to fill
        self.all_scores = torch.zeros(self.vocab_size, self.n_classes)
        self.all_counts = torch.zeros(self.vocab_size)

        # Dicts in which to accumulate scores
        self.phrase_scores = defaultdict(lambda: defaultdict(float))

        # We don't need gradients for backprop
        with torch.no_grad():
            for i, batch_dict in enumerate(tqdm(self.train_loader),1):
                if i > self.args.max_rule_iters:
                    break

                clean_input_ids = batch_dict['input_ids']
                attention_masks = batch_dict['attention_masks']
                noisy_labels = batch_dict['noisy_labels']
                noised_ids = batch_dict['noised_ids']
                mlm_labels = batch_dict['mlm_labels']
                starts = batch_dict['word_starts']
                ends = batch_dict['word_ends']
                batch_inds = batch_dict['batch_inds']
                word_inds = batch_dict['word_inds']
                word_mask = batch_dict['word_mask']
                batch_words = batch_dict['batch_words']
                soft_labels = batch_dict['soft_labels']

                if 'labels' in batch_dict:
                    labels = batch_dict['labels']

                n_rules = torch.LongTensor([self.n_rules])
                # covered_mask = torch.BoolTensor((noisy_labels != -1).sum(dim=1) > 0)

                # Use GPU if available
                if self.cuda:
                    noised_ids = noised_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    labels = labels.cuda()
                    # noisy_labels = noisy_labels.cuda()
                    # n_rules = n_rules.cuda()
                    # covered_mask = covered_mask.cuda()
                    # all_scores = all_scores.cuda()
                    # all_counts = all_counts.cuda()
                    clean_input_ids = clean_input_ids.cuda()
                    mlm_labels = mlm_labels.cuda()
                    # starts = starts.cuda()
                    # ends = ends.cuda()
                    batch_inds = batch_inds.cuda()
                    word_inds = word_inds.cuda()
                    word_mask = word_mask.cuda()
                    soft_labels = soft_labels.cuda()

                # Calculate outputs (specifically probs and attentions)
                # output_dict = self.model(noised_ids, attention_masks, noisy_labels, n_rules, covered_mask, mlm_labels)
                output_dict = self.model(noised_ids, attention_masks, soft_labels, mlm_labels)
                probs = output_dict['probs']
                attentions = output_dict['attention']

                # Get attention, weighted by class confidence
                attention_probs = attentions * probs.unsqueeze(1)
                masked_attention_probs = (attention_probs * attention_masks.unsqueeze(2))

                # Compute single token rule recommendations
                # self.get_token_scores(masked_attention_probs, noised_ids)

                # Whole phrase recommendations
                whole_word_scores = get_word_scores(masked_attention_probs, 
                                                         batch_inds, 
                                                         word_inds, 
                                                         word_mask)
                
                # Get scores/counts for each phrase_length
                for phrase_len in range(1, self.args.max_rule_length+1):
                    # scores, counts = accumulate_phrase_scores(batch_words, 
                    #                                           whole_word_scores, 
                    #                                           batch_inds, 
                    #                                           phrase_len)
                    scores = accumulate_phrase_scores(batch_words, 
                                                        whole_word_scores, 
                                                        batch_inds, 
                                                        phrase_len,
                                                        self.phrase_counts)
                    for phrase, score in scores.items():
                        if phrase in self.phrase_counts:
                            self.phrase_scores[phrase_len][phrase] += score


    def prune_low_count_phrase_scores(self):
        '''
        Remove phrases with low count
        '''
        for phrase in self.phrase_counts.keys():
            if any([subphrase in phrase for subphrase in self.used_kwds]):
                self.phrase_counts[phrase] = 0
        pruned_scores = {l:{phrase:scores for (phrase, scores) in d.items() 
                            if self.phrase_counts[phrase] >= self.min_count_cutoff}
                         for l, d in self.phrase_scores.items()}
        # pruned_counts = {phrase:count for (phrase, count) in self.phrase_counts.items() 
        #                  if count >= self.min_count_cutoff}

        self.phrase_scores = pruned_scores
        # self.phrase_counts = pruned_counts
        
    def get_multitoken_scores(self):
        '''
        Convert phrase scores to tensor for fast operations
        '''
        # logger.debug(f"Phrase Scores: {list(self.phrase_scores.items())[:50]}")
        # logger.debug(f"Phrase counts: {list(self.phrase_counts.items())[:50]}")
        self.phrase2id = {i:{} for i in range(1, self.args.max_rule_length+1)}
        self.id2phrase = {i:{} for i in range(1, self.args.max_rule_length+1)}
        self.multitoken_scores = {}
        self.multitoken_counts = {}
        for l in range(1, self.args.max_rule_length+1):
            num_phrases = len(self.phrase_scores[l])
            multitoken_scores = torch.zeros((num_phrases, self.n_classes))
            multitoken_counts = torch.zeros((num_phrases))
            for i, (phrase, scores) in enumerate(self.phrase_scores[l].items()):
                self.phrase2id[l][phrase] = i
                self.id2phrase[l][i] = phrase
                # print("i:", i)
                # print("scores:", scores)
                multitoken_scores[i,:] = scores
                multitoken_counts[i] = self.phrase_counts[phrase]

            self.multitoken_scores[l] = multitoken_scores
            self.multitoken_counts[l] = multitoken_counts
        # return phrase_scores, phrase_counts


    def get_weighted_phrase_sums(self, alpha, rule_length=1, eps=1e-12):
        '''
        Return attention and class-prob weighted sum of phrase relevance for rule creation.
        This is separated from get_weighted_token_vals to allow for quick tuning of alpha
        '''
        # Make sure alpha is in acceptable range
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must lie in [0,1]")
        
        

        # for l in range(1, self.args.max_rule_length+1):
        self.weighted_phrase_sums[rule_length] = self.multitoken_scores[rule_length] / ((eps + 
                                                            self.multitoken_counts[rule_length]) ** alpha).unsqueeze(-1)
        # logger.debug(f"Weighted phrase sums: {self.weighted_phrase_sums}")

    def prune_existing_rules(self):
        '''
        Eliminate rules that contain used tokens
        '''
        for l, d in self.multitoken_scores.items():
            for phrase, scores in d:
                if any([w in phrase for w in phrase_kwds]):
                    multitoken_scores[l][phrase] *= 0

    def propose_phrase_rules(self, rule_length, use_synonyms=False):
        '''
        Create phrase-based rules
        '''
        potential_rules = {}

        # Scores normalized by counts
        # Run after choosing alpha
        count_normed_scores = self.weighted_phrase_sums[rule_length]

        # Normalize rows to sum to 1
        ordered_vals, _ = (count_normed_scores / count_normed_scores.sum(dim=1, keepdims=True)).topk(dim=1, k=2)

        # Only keep scores with substantial polarity
        polarity_vals = ordered_vals[:,0] - ordered_vals[:,1]
        polarity_mask = polarity_vals > self.polarity_thresh

        # Prune scores below polarity thresh
        count_normed_scores = count_normed_scores[polarity_mask]
        count_normed_mean = count_normed_scores.mean(dim=1, keepdims=True)

        # Normalize to have mean 0
        normed_scores = (count_normed_scores - count_normed_mean)
        # logger.debug(f"\n\nNormed Scores: {normed_scores}" )

        # Indices with sufficient polarity to one class
        masked_inds = torch.arange(polarity_mask.shape[0])[polarity_mask]

        # Get topk values/indices.  Also get row maxes
        k = self.k
        if normed_scores.size(0) < self.k:
            k = normed_scores.size(0)
        # k = normed_scores.size(0)

        if k == 0:
            return None

        rule_vals, rule_inds = normed_scores.topk(dim=0, k=k)


        for i in trange(rule_vals.size(1)):
            # Get indices of top rules
            vals = rule_vals[:, i]
            inds = rule_inds[:, i]


            # Make sure indices are for desired class
            val_ind_mask = (vals > 0)
            vals = vals[val_ind_mask]
            inds = inds[val_ind_mask]

            # Get condidence score for each chosen rule
            confidence_scores = (vals + count_normed_mean.flatten()[inds]) / count_normed_scores[inds].sum(dim=1)

            # Get dict of rules for current class
            class_dict = {'rules':[(self.id2phrase[rule_length][masked_inds[j].item()], score.item()) for j, score in zip(inds, confidence_scores)]}
            potential_rules[i] = class_dict
        

            # Name our classes for understandability
            if self.class_names is not None:
                print("Class Names:", self.class_names[i])
                class_dict['name'] = self.class_names[i]

            potential_rules[i] = class_dict
        
        return potential_rules


    # TODO: Move to _tune_rule_metric
    def check_phrase_alpha(self, alpha, rule_length):
        # Make sure user is happy with choice of alpha
        self.get_weighted_phrase_sums(alpha=alpha, 
                                      rule_length=rule_length)
        self.potential_rules = self.propose_phrase_rules(rule_length=rule_length)
        if self.potential_rules is None:
            return True, alpha
        logger.debug("Rules proposed!")

        print("Sample rules for this choice of alpha")
        if self.class_names is not None:
            display_sample_rules(self.potential_rules, count_dict=self.phrase_counts, class_names=self.class_names)
        else:
            display_sample_rules(self.potential_rules, count_dict=self.phrase_counts,)

        undecided = True

        while undecided:
            satisfied = input("\nAre you satisfied with the proposed rules? [y/n/new_alpha]\n").lower()
            if satisfied == 'y' or satisfied == 'yes':
                return True, alpha
            # elif satisfied == 'n' or satisfied == 'no':
            #     return False
            # else:
            #     "You must choose a value in {y, n}"
            else:
                return False, satisfied

    # TODO: Move to _tune_rule_metric
    def choose_phrase_alpha(self, rule_length):
        '''
        Tune alpha to choose rules that balance coverage and rule specificity
        '''
        # Allow user to choose alpha
        print(f"\nAlpha controls how much to downweight common words.  Lower values favor high coverage while higher values favor greater individual weight.  \n\tDefault: {self.init_alpha}\n")

        # Let user change alpha if they desire
        new_alpha = None
        alpha_unspecified = True
        while alpha_unspecified:
            if new_alpha is not None:
                try:
                    alpha = float(new_alpha)
                except:
                    alpha = input(f"\nEnter a new value of alpha if you would like to change it\n Press 'enter' to accept default value of {self.init_alpha}\n")
            else:
                alpha = input(f"\nEnter a new value of alpha if you would like to change it\n Press 'enter' to accept default value of {self.init_alpha}\n")

            # Show user rules for default alpha
            if alpha == '':
                alpha = self.init_alpha
                satisfied, new_alpha = self.check_phrase_alpha(alpha, rule_length=rule_length)
                if satisfied:
                    alpha_unspecified = False

            # Show user rules for custom alpha
            else:
                try:
                    alpha = float(alpha)

                    # Make sure alpha is valid
                except:
                    logger.debug("Entered except loop")
                    logger.debug(alpha)
                    print("Alpha must be a number in the closed interval [0,1]")
                    continue
              
                if alpha < 0 or alpha > 1:
                    logger.info(alpha)
                    print("alpha must be in the closed interval [0,1]")
                    alpha = self.init_alpha

                # Determine if user is happy with alpha
                else:
                    # logger.debug("Entering check_phrase_alpha")
                    satisfied, new_alpha = self.check_phrase_alpha(alpha, rule_length=rule_length)
                    if satisfied:
                        alpha_unspecified = False


    def user_rule_input(self):
        '''
        Let user choose which rules to add to the ruleset

        Inputs:
        --------------------
            potential_rules: dict
                List of potential rules returned from propose_single_token_rules
            
        Outputs:
        --------------------
            new_rules: list of LabelingFunction
                List of all new rules chosen by user
        '''
        new_lfs = []
        new_rules = []
        chosen_words = []
        proposed_words = []
        if self.potential_rules is not None:
    
            k = len(self.potential_rules)
            print("***Choose new rules***")

            # Loop through classes
            for j, (key, class_dict) in enumerate(self.potential_rules.items()):
                # class_dict = potential_rules[key]

                # Get class name if available
                logger.debug(f"class dict keys: {class_dict.keys()}")
                if 'name' in class_dict.keys():
                    name = class_dict['name']
                else:
                    name = key

                unchosen = True
                while unchosen:
                    # Display rules for current class
                    print (f"Currently choosing rules for class: {name} ({j+1} of {k})")
                    n = len(class_dict['rules'])

                    # Put in table for readability
                    sorted_rules = sorted(class_dict['rules'], key=lambda x: x[1])[::-1]
                    t = Texttable()
                    t.add_rows([['Number','Rule','Score', 'Count']] + 
                            [[i, f'HAS({rule}) -> {name}', score, len(self.phrase_inds[rule])] 
                                for i, (rule, score) in enumerate(sorted_rules)]
                            )
                    print(t.draw())

                    # Let user pick desired rules
                    rules_str = input("Which of the above rules are applicable? \n(please enter all numbers separated by commas)\n")
                    try:
                        if len(rules_str.strip()) == 0:
                            chosen_rules = []
                            unchosen = False
                        else:
                            chosen_rules = [int(i.strip()) for i in rules_str.split(',')]
                            unchosen = False
                    except:
                        print("Invalid input.  Please enter rules as comma-delimited set of integers corresponding to the rules you want to select.")


                # Map chosen rules to label functions
                rule_words = [sorted_rules[ind][0] for ind in chosen_rules]
                logger.debug(f"Rule Words: {rule_words}")
                chosen_words.extend(rule_words)
                if len(rule_words) > 0:
                    new_rules.append((key, rule_words))
                    new_lfs.append(make_keyword_lf(rule_words, key))
                
                # Keep track of proposed rule words so they don't keep coming up
                proposed_words.extend([r[0] for r in class_dict['rules']])


                logger.debug(f"Chosen phrases: {chosen_words}")
        return new_lfs, new_rules, chosen_words, proposed_words

    def get_phrase_rule_score(self, phrase, class_label):
        '''
        Evaluate phrase rule score based on % correctly labeled from random sample of data points
        '''
        # Find examples that match rule
        n = self.args.num_autoeval_examples
        matched_inds = np.array(self.phrase_inds[phrase])

        # Use oracle accuracy or random sampling
        if self.args.oracle:
            random_inds = matched_inds
        else:
            m = len(matched_inds)
            if m < n:
                # logger.debug(f"length: {len(self.train['text'])}")
                logger.debug(f"m: {m} {matched_inds}")
            random_inds = torch.LongTensor(np.random.choice(matched_inds, size=min([n,m]), replace=False))

        # Get labels and calculate proportion that match
        chosen_labels = self.train.labels[random_inds]
        score = (chosen_labels == class_label).float().mean().item()
        return score



    def autoeval(self, rule_length, epoch=0):
        '''
        Automatic evaluation of proposed rules
        '''
        k = len(self.potential_rules)
        all_dfs = []
        new_lfs = []
        new_rules = []
        chosen_words = []
        proposed_words = []

        # Loop through classes
        for j, (key, class_dict) in enumerate(self.potential_rules.items()):
            # class_dict = potential_rules[key]
            curr_label = key

            # Get class name if available
            # logger.debug(f"class dict keys: {class_dict.keys()}")
            if 'name' in class_dict.keys():
                name = class_dict['name']
            else:
                name = key

            unchosen = True
            # Display rules for current class
            print (f"Currently choosing rules for class: {name} ({j+1} of {k})")
            n = len(class_dict['rules'])

            # Put in table for readability
            sorted_rules = sorted(class_dict['rules'], key=lambda x: x[1])[::-1]
            rule_df = pd.DataFrame([[key, name, rule, score, len(self.phrase_inds[rule])] 
                                    for i, (rule, score) in enumerate(sorted_rules)], 
                                    columns=['class_number', 
                                            'class_name',
                                            'rule',
                                            'score', 
                                            'count'])

            rule_df['query_score'] = rule_df['rule'].map(lambda x: self.get_phrase_rule_score(x, class_label=curr_label))

            # Determine which rules get chosen
            chosen = rule_df['query_score'] >= self.args.autoeval_thresh
            rule_df['chosen'] = chosen
            rule_words = rule_df.loc[chosen, 'rule'].tolist()

            # Keep dfs for use
            all_dfs.append(rule_df)

            # Map chosen rules to label functions
            # rule_words = [sorted_rules[ind][0] for ind in chosen_rules]
            # logger.debug(f"Rule Words: {rule_words}")
            chosen_words.extend(rule_words)
            if len(rule_words) > 0:
                new_rules.append((key, rule_words))
                for word in rule_words:
                    new_lfs.append(make_keyword_lf(word, key))
            
            # Keep track of proposed rule words so they don't keep coming up
            proposed_words.extend([r[0] for r in class_dict['rules']])


            # logger.debug(f"Chosen phrases: {chosen_words}")

        log_df = pd.concat(all_dfs)
        self.save_df(log_df, rule_length, epoch)

        return new_lfs, new_rules, chosen_words, proposed_words



    def add_new_rules(self, epoch=0):
        # Get token weights
        logger.info("Getting word importance weights")
        
        self.get_weighted_token_vals()
        logger.info("Getting phrase scores")
        added_lfs = []

        # Make sure we add new rules
        rules_updated = False
        # Choose new rules
        for rule_length in range(1, self.args.max_rule_length+1):
            print(f"Adding rules of length {rule_length} ({rule_length} of {self.args.max_rule_length})")
            # Prune used phrases and rescore
            self.prune_low_count_phrase_scores()
            self.get_multitoken_scores()

            if not self.args.autoeval:
                self.choose_phrase_alpha(rule_length)
                new_lfs, new_rules, new_rule_phrases, proposed_phrases = self.user_rule_input()
            else:
                self.get_weighted_phrase_sums(alpha=self.init_alpha, 
                                      rule_length=rule_length)
                self.potential_rules = self.propose_phrase_rules(rule_length=rule_length)
                if self.potential_rules is None:
                    continue
                # logger.debug("Rules proposed!")
                # logger.debug(f"potential rules: {self.potential_rules}")
                new_lfs, new_rules, new_rule_phrases, proposed_phrases = self.autoeval(rule_length, epoch=epoch)
            # new_lfs, new_rules, new_rule_kwds, proposed_kwds = self.user_rule_input()


            
            # Update data sources
            self.rules.extend(new_rules)
            self.update_noisy_labels(new_lfs)
            self.make_dataloaders()
            # self.update_rule_kwds(new_rule_kwds, proposed_kwds)
            self.update_rule_phrase_kwds(new_rule_phrases)
            self.n_rules += len(new_lfs)
            # logger.debug(f'Updated RPN n_rules: {self.n_rules}')
            self.lfs.extend(new_lfs)
            self.save_scores()
            self.prune_low_count_phrase_scores()
            if len(new_lfs) > 0:
                rules_updated = True

        return rules_updated


def run_one_epoch(args, 
                  model, 
                  rpn, 
                  optimizer, 
                  epoch, 
                  best_score,
                  scheduler=None, 
                  cuda=True, 
                  do_re=False,
                  warmup=False):


    # Train model
    if warmup:
        train_metrics = train(args, 
            model, 
            rpn.train_loader, 
            rpn.n_rules, 
            optimizer, 
            epoch, 
            scheduler=scheduler, 
            cuda=cuda, 
            # max_iter=None, 
            do_re=do_re,
            warmup=warmup)
    else:
        logger.debug("**run_one_epoch training**")
        train_metrics = train(args, 
            model, 
            rpn.labeled_loader, 
            rpn.n_rules, 
            optimizer, 
            epoch, 
            scheduler=scheduler, 
            cuda=cuda, 
            # max_iter=None, 
            do_re=do_re,
            warmup=warmup)

    # Update minimum number of LFs required for matched samples after first epoch of training
    if rpn.min_lf != args.min_lf:
        rpn.min_lf = args.min_lf


    # Get metrics on dev set
    # val_loader = DataLoader(valid_tuple)
    if not args.debug:
        eval_metrics = evaluate(args, 
                                model, 
                                rpn.valid_loader, 
                                epoch, 
                                rpn.n_rules, 
                                cuda=cuda)

        # Get test metrics if eval set gets best score
        if eval_metrics[args.metric] > best_score:
            best_epoch = epoch
            best_score = eval_metrics[args.metric]
            test_metrics = evaluate(args, 
                                    model, 
                                    rpn.test_loader, 
                                    epoch, 
                                    rpn.n_rules, 
                                    cuda=cuda)
            test_metrics['best_epoch'] = best_epoch
            logger.info(pprint.pformat(test_metrics))
            log_metrics(test_metrics, pathjoin(rpn.save_dir, args.logdir, f'test_metrics_{epoch}.json'))

        # Save model and metrics
        log_metrics(eval_metrics, pathjoin(rpn.save_dir, args.logdir, f'metrics_{epoch}_eval.json'))
        save_checkpoint(model, optimizer, epoch, rpn.lfs, pathjoin(rpn.save_dir, args.checkpoint_dir))
        # rpn.save_rules(epoch)

        return best_score

    else:

        return 0


# def test_dataloader(args):
#     logger.info("Loading Data")
#     # Get model info
#     tokenizer, basemodel = get_model_and_tokenizer(model_no)
#     basemodel.resize_token_embeddings(len(tokenizer))

#     cuda = torch.cuda.is_available()

#     # Create full model
#     model = AssembleModel(basemodel, 
#                           args.fc_size, 
#                           args.max_rules, 
#                           args.hidden_size, 
#                           args.num_classes)
#     if cuda:
#         model.cuda()

#     rpn = RuleProposer(args, model, tokenizer, cuda=cuda)

#     for i, batch in enumerate(tqdm(rpn.train_loader)):
#         if i == 24:
#             # logger.debug(batch)
#             logger.debug([x for x in batch])
        

def main(args):
    start = time.time()

    # Set random states
    torch.manual_seed(args.seed)

    # Make sure necessary files exist
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    logger.info("Loading Data")

    # Load model and tokenizer
    tokenizer, basemodel = get_model_and_tokenizer(args.model_no)
    basemodel.resize_token_embeddings(len(tokenizer))

    cuda = torch.cuda.is_available()
        
    # Deal with frozen models
    if args.freeze == 1:
        logger.info("FREEZING MOST HIDDEN LAYERS...")
        if args.model_no >= 1:
            # unfrozen_layers = ['classifier','lm_linear','cls','blanks_linear','pooler']
            unfrozen_layers = ["classifier", 
                               "pooler", 
                               "encoder.layer.11", 
                               "encoder.layer.10",
                               "encoder.layer.9", 
                               "encoder.layer.8"
                               "blanks_linear", 
                               "lm_linear", 
                               "cls"]
        elif args.model_no == 1:
            unfrozen_layers = ["classifier", "pooler", "embeddings", "attention",\
                               "blanks_linear", "lm_linear", "cls",\
                               "albert_layer_groups.0.albert_layers.0.ffn"]
            
        for name, param in basemodel.named_parameters():
            if not any([layer in name for layer in unfrozen_layers]):
                # print("[FROZE]: %s" % name)
                param.requires_grad = False
            else:
                # print("[FREE]: %s" % name)
                param.requires_grad = True

    # Create full model
    model = AssembleModel(basemodel, 
                          args.fc_size, 
                          args.max_rules, 
                        #   args.hidden_size, 
                          args.num_classes)
    if cuda:
        model.cuda()

    # Create optimizer object
    optimizer = optim.Adam(model.parameters(), 
                            lr=args.lr, 
                            # weight_decay=args.weight_decay
                            )

    
    rpn = RuleProposer(args, model, tokenizer, cuda=cuda)

    # logger.info("Starting training process...")
    best_score = -1

    ### Old location of warmup training ###

    
    ### End old loc of warmup ###

    # for epoch in trange(warmup, args.num_epochs + warmup):
    for epoch in trange(args.num_epochs):
        # Get updated dataloaders
        # train_loader = DataLoader(TensorDataset(train_tuple))

        if args.refresh:
            model = rpn.refresh_model()
            # Create optimizer object
            optimizer = optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    # weight_decay=args.weight_decay
                                    )

        ### Testing Warmup Training ###
        if args.warmup_epochs > 0:
            logger.info("Running warmup training")
            warmup = args.warmup_epochs
            for epoch in trange(warmup):
                best_score = run_one_epoch(args, 
                        model, 
                        rpn, 
                        optimizer, 
                        epoch, 
                        best_score,
                        cuda=cuda,
                        warmup=True)
        
        else:
            warmup = 0

        ### End Test ###
            

        # best_score = run_one_epoch(args, 
        run_one_epoch(args,
                    model, 
                    rpn, 
                    optimizer, 
                    epoch, 
                    best_score,
                    cuda=cuda)

        if len(rpn.lfs) < args.max_rules:
                
            rules_updated = rpn.add_new_rules(epoch=epoch)
            if not rules_updated:
                logger.info("***\nNo rules updated.  Stopping now.\n***")
                break

            # logger.debug(f"Rules: {rpn.rules}")
            rpn.save_rules(epoch=epoch)
            output_dict = get_model_output(rpn)
            name = output_dict['dataset']
            output_dict['runtime'] = time.time() - start
            pickle.dump(output_dict, open(f'./run_metrics/{name}.pickle', 'wb'))

