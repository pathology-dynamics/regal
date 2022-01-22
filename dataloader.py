# Imports
import torch
import numpy as np
import logging
import pickle
import os
import pytorch_lightning as pl

# Submodules
from typing import Union, List
from tqdm import tqdm, trange
from torch.utils.data import Dataset, TensorDataset
from snorkel.labeling import LFApplier
from snorkel_utils import make_keyword_lf

# from allennlp.nn.utils import flatten_and_batch_shift_indices


# Need to set tokenizers_parallelism environment variable to avoid lots of warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logger = logging.getLogger('__file__')

# Collate function for RPNDataset
class RPNCollate():
    def __init__(self, tokenizer):
        # self.id2word = id2word
        self.tokenizer = tokenizer

    def __call__(self, batch):
        '''
        Collate function to turn batch from dataloader into clean dict of output
        '''
        # print(batch)
        # print("Length", len(batch))
        # seq, attn_mask, labels, noisy_labels, noised_ids, mlm_labels, starts, ends = *batch


        input_ids = torch.stack(tuple([x['input_ids'] for x in batch]))
        attn_mask = torch.stack(tuple([x['attention_masks'] for x in batch]))
        labels = torch.stack(tuple([x['labels'] for x in batch]))
        noisy_labels = torch.stack(tuple([x['noisy_labels'] for x in batch]))
        soft_labels = torch.stack(tuple([x['soft_labels'] for x in batch]))
        noised_ids = torch.stack(tuple([x['noised_ids'] for x in batch]))
        mlm_labels = torch.stack(tuple([x['mlm_labels'] for x in batch]))
        starts = [x['word_starts'] for x in batch]
        ends = [x['word_ends'] for x in batch]


        # Get batch indices and start/end indices of each word
        batch_inds = torch.cat(tuple([i*torch.ones_like(s).long() for i, s in enumerate(starts)])).reshape(-1,1)
        starts = torch.cat(tuple(starts)).reshape(-1,1)
        ends = torch.cat(tuple(ends)).reshape(-1,1)

        # Get tensor to select ids and/or embeddings for each word from a tensor
        word_lengths = ends-starts
        max_len = word_lengths.max()
        selector_inds = starts + torch.arange(max_len)
        selector_mask = (selector_inds < ends)
        selector_inds[~selector_mask] = 0

        # Get all words in the batch to be used for creating phrase-based rules
        batch_words = reconstruct_words(input_ids, starts, ends, self.tokenizer, batch_inds=batch_inds)


        output_dict = {
                        'input_ids': input_ids, 
                        'attention_masks': attn_mask, 
                        'labels': labels, 
                        'noisy_labels': noisy_labels, 
                        'noised_ids': noised_ids, 
                        'mlm_labels': mlm_labels,
                        'batch_inds': batch_inds,
                        'word_starts':starts, 
                        'word_ends': ends,
                        'word_inds': selector_inds,
                        'word_mask': selector_mask,
                        'batch_words': batch_words,
                        'soft_labels': soft_labels
                        }
        return output_dict


# Helper functions
# def reconstruct_words(input_ids, starts, ends, id2word, batch_inds=None):
def reconstruct_words(input_ids, starts, ends, tokenizer, batch_inds=None):
    '''
    Reconstruct all words in text from their input ids
    '''
    words = []
    ss = starts.flatten()
    es = ends.flatten()

    if batch_inds is not None:
        bs = batch_inds.flatten()
        words = [tokenizer.decode(input_ids[b, s:e]) for b, s, e in zip(bs, ss, es)]
        # for (b, s, e) in zip(bs, ss, es):
            # if s - e == 1:
            #     words.append[id2word[input_ids[b, s:e].item()]]
            # else:
            #     subword_ids = input_ids[b, s:e].numpy()
            #     words.append(tokenizer.decode(subword_ids))
                # words.append(merge_tokens(subword_ids, id2word))
    else:
        words = [tokenizer.decode(input_ids[s:e]) for s, e in zip(ss, es)]
        # for (s, e) in zip(ss, es):
            # if s - e == 1:
            #     words.append[id2word[input_ids[s:e].item()]]
            # else:
            #     subword_ids = input_ids[s:e].numpy()
            #     words.append(tokenizer.decode(subword_ids))
                # words.append(merge_tokens(subword_ids, id2word))

    return words


# def merge_tokens(subword_ids, id2word):
#     '''
#     Merge tokens from subword units
#     '''
    # tokens = [id2word[i] for i in subword_ids]
    # s = tokens[0]
    # for t in tokens[1:]:
    #     if t.startswith('##'):
    #         s += t[2:]
    #     else:
    #         s += ' ' + t

    # return s

def get_word_spans(word_ids, punct_inds=None):
        '''
        Get spans of whole words list of wordpiece -> word mappings

        Params:
        -------
            word_ids: List
                List of which word is mapped to each individual token
          

        Returns:
        --------
            span_starts: torch.LongTensor
                Array of starts of word spans

            span_ends: torch.LongTensor
                Array of ends of word spans

        Example:
        --------
            Sentence:   "the dog jumped excitedly"
            -> Tokenized:  ['[CLS]', 'the','dog', 'jump', '##ed', 'excit', '##ed', '##ly', '[SEP]']
            -> word_ids:   [None, 0, 1, 2, 2, 3, 3, 3, None]
            -> Spans: [(0,0), (1,2), (2,3), (3,5), (5,8), (0,0)]


            Usage: self.get_word_spans(word_ids) #word_ids as above
                -> returns: (tensor([1, 2, 3, 5]), tensor([2, 3, 5, 8]))
        
        '''
        prev_ind = None
        starts = []
        ends = []

        # Gather start and end indices
        for i, ind in enumerate(word_ids):
            if prev_ind != ind:
                if prev_ind != None:
                    ends.append(i)
                if ind != None:
                    starts.append(i)
            prev_ind = ind

        # Return tensors
        return (torch.LongTensor(starts), torch.LongTensor(ends))

def prep_data(text, tokenizer, max_length=128):
    '''
    Prep data for RPN usage
    '''
    enc = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors='pt', return_token_type_ids=False)

    # Portion out different vaues
    encoded_text = enc['input_ids']
    attention_masks = enc['attention_mask']

    # Get word start/end indices
    word_spans = [get_word_spans(enc.word_ids(i)) for i in trange(len(text))]
    word_starts = [s[0] for s in word_spans]
    word_ends = [s[1] for s in word_spans]
    return encoded_text, attention_masks, word_starts, word_ends


class RPNDataset(Dataset):
    # RPN Dataset to mask keywords used in rules
    def __init__(self, 
                 data, 
                 tokenizer,
                 rule_keywords, 
                 rule_tokens=[],
                 mask_prob=.1, 
                 rule_mask_prob=.5,
                 seed_labels=None, 
                 filter_labels=True,
                 max_length=128, 
                 min_lf=1,
                 ):
        self.text = data['text']
        self.tokenizer = tokenizer
        if 'rule_keywords' in data:
            self.rule_keywords = data['rule_keywords']
        else:
            self.rule_keywords = rule_keywords

        # Tokenizer attributes
        self.word2id = tokenizer.vocab
        self.mask_id = self.word2id['[MASK]']
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.max_length = max_length

        # Make sure data is ready for deep learning models
        if 'encoded_text' not in data.keys():
            self.prepare_data()
        else:
            self.encoded_text = data['encoded_text']
            self.attention_masks = data['attention_masks']
            self.word_starts = data['word_starts']
            self.word_ends = data['word_ends']
        self.labels = data['labels']

        if 'word_lists' in data.keys():
            self.word_lists = data['word_lists']
        else:
            logger.info("Computing word lists")
            self.word_lists = [reconstruct_words(ids, starts, ends, self.tokenizer) 
                                for (ids, starts, ends) in tqdm(zip(self.encoded_text, 
                                                                self.word_starts, 
                                                                self.word_ends))]

        # Make sure noisy labels are there
        self.min_lf = min_lf
        if 'noisy_labels' not in data:
            self.make_lfs(rpn_generated=False)
            self.make_noisy_labels()

        else:
            self.noisy_labels = data['noisy_labels']

        self.balance_noisy_labels()

        if 'soft_labels' in data:
            self.soft_labels = data['soft_labels']
        else:
            soft_labels = None
        # self.soft_labels = data['soft_labels']
        
        
        # labeled_inds = ((self.noisy_labels >= 0).sum(dim=1) >= min_lf).nonzero().flatten()
        # logger.debug(labeled_inds.size)
        # logger.debug(f'Proportion labeled: {labeled_inds.size(0)/self.noisy_labels.size(0)}')
        # self.labeled_inds = labeled_inds

        
        # Get vocab size
        self.vocab_size = int(np.max(list(self.word2id.values())) + 1)
        self.num_special_tokens = int(np.max([val for key, val in self.word2id.items() if key.startswith('[')]) + 1)

        # Rule attributes
        self.rule_tokens = rule_tokens
        self.rule_map = {val:val for val in self.word2id.values()}
        self.update_rule_map(rule_tokens)
        self.is_rule = {val:0 for val in self.word2id.values()}
        for w in rule_tokens:
            if w.strip() in self.word2id:
                self.is_rule[self.word2id[w.strip()]] = 1
    
        # Misc attributes
        self.p = mask_prob
        self.rule_p = rule_mask_prob
        self.length = len(self.text)
        self.idx_map = {i:i for i in range(self.length)}

    def prepare_data(self,):
        '''
        Prepare data by tokenizing, padding, and getting word start/end indices

        Params:
        -------
            text: List[str]
                List of text of each instance
        '''
        # Encode text
        enc = self.tokenizer(self.text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt', return_token_type_ids=False)

        # Portion out different vaues
        self.encoded_text = enc['input_ids']
        self.attention_masks = enc['attention_mask']

        # Get word start/end indices
        word_spans = [get_word_spans(enc.word_ids(i)) for i in trange(len(self.text))]
        self.word_starts = [s[0] for s in word_spans]
        self.word_ends = [s[1] for s in word_spans]

    # Make more general to apply to n-grams/phrases
    def make_lfs(self, rpn_generated=True):
        '''
        Make labeling functions from keywords/phrases
        '''
        self.keyword_lfs = [make_keyword_lf(w, label, rpn_generated=rpn_generated) for label, words in self.rule_keywords.items() for w in words if not ' ' in w]
        self.phrase_lfs = [make_keyword_lf(w, label, rpn_generated=rpn_generated) for label, words in self.rule_keywords.items() for w in words if ' ' in w]
        

    def make_noisy_labels(self):
        '''
        Make noisy labels from labeling functions
        '''
        if len(self.keyword_lfs) > 0:
            keyword_applier = LFApplier(lfs=self.keyword_lfs)
            keyword_noisy_labels = torch.LongTensor(keyword_applier.apply(self.word_lists))
            noisy_labels = keyword_noisy_labels
        if len(self.phrase_lfs) > 0:
            phrase_applier = LFApplier(lfs=self.phrase_lfs)
            phrase_noisy_labels = torch.LongTensor(phrase_applier.apply(self.text))
            noisy_labels = phrase_noisy_labels

        if len(self.keyword_lfs) > 0 and len(self.phrase_lfs) > 0:
            noisy_labels = torch.cat((keyword_noisy_labels, phrase_noisy_labels), dim=1)

        self.full_noisy_labels = noisy_labels


    def balance_noisy_labels(self):
        '''
        Balance number of noisy labels for each class to prevent model imbalance
        '''
        self.noisy_labels = self.full_noisy_labels.clone()
        label_counts = [(self.noisy_labels == label).sum().item() for label in self.rule_keywords.keys()]
        logger.debug(f"Old label counts: {label_counts}")  

        # Balance classes
        count_min = min(label_counts)
        for label in self.rule_keywords.keys():
            count = (self.noisy_labels == label).sum()
            cutoff = (count - count_min)/count
            mask = (torch.rand(self.noisy_labels.size()) < cutoff) & (self.noisy_labels == label)
            self.noisy_labels[mask] = -1

        label_counts = [(self.noisy_labels == label).sum() for label in self.rule_keywords.keys()]
        logger.debug(f"New label counts: {label_counts}")  

        labeled_inds = ((self.noisy_labels >= 0).sum(dim=1) >= self.min_lf).nonzero().flatten()
        # logger.debug(labeled_inds.size)
        logger.debug(f'Proportion labeled: {labeled_inds.size(0)/self.noisy_labels.size(0)}')
        self.labeled_inds = labeled_inds

        
    def _use_labeled(self):
        '''
        Switches model to only iterate through labeled data
        '''
        labeled_inds = ((self.noisy_labels >= 0).sum(dim=1) >= self.min_lf).nonzero().flatten()
        self.labeled_inds = labeled_inds
        self.length = self.labeled_inds.size(0)
        self.idx_map = {i:self.labeled_inds[i] for i in range(self.length)}

        # Debugging statements
        # logger.debug(labeled_inds.size)
        logger.debug(f'Proportion labeled: {labeled_inds.size(0)/self.noisy_labels.size(0)}')
        

        # return noisy_labels

    # def precompute_phrase_counts(self):
    #     '''
    #     Precompute word counts for faster model training
    #     '''
    #     phrase_counts = defaultdict(int)
    #     phrase_inds = defaultdict(set)
    #     normalized_text = []
    #     logger.info("Precomputing phrase counts")
    #     for j, word_list in enumerate(tqdm(self.train['word_lists'])):
    #         normalized_text.append(" ".join(word_list))
    #         # normalized_text.append(self.tokenizer.decode(self.tokenizer.encode(word_list)[1:-1]))
    #         for l in range(1, 1 + self.args.max_rule_length):

    #             phrases = [" ".join(word_list[i:i+l]) for i in range(len(word_list) - l + 1)]
    #             for p in phrases:
    #                 if any([punct in p for punct in '.,!?"\\']):
    #                     continue
    #                 phrase_counts[p] += 1
    #                 phrase_inds[p].add(j)

    #     self.train['text'] = normalized_text
    #     self.phrase_counts = {k:v for k, v in phrase_counts.items() if v >= self.min_count_cutoff and k not in self.words_to_exclude}
    #     logger.debug(f"Num Phrases: {len(self.phrase_counts)}")
    #     self.phrase_inds = {k:list(phrase_inds[k]) for k in self.phrase_counts.keys()}

    def update_rule_map(self, kwds):
        for kwd in kwds:
            self.rule_map[kwd] = self.mask_id

    def token_match(self, token, alg='random', n=5):
        '''
        Match examples based on token
        '''
        pass

    def phrase_match(self, phrase, alg='random', n=5):
        '''
        Match examples based on phrase
        '''
        pass

    # Needs updating for whole words/phrases
    def noise_input_tokens(self, seq, p=1):
        '''
        Add noise to input sequences for MLM loss

        Inputs:
        -------
            seq: Input sequence on which to mask tokens

            p: Probability with which to mask each token from a rule
        '''
        rule_tokens = torch.tensor([self.is_rule[w.item()] for w in seq]).bool()
        # rule_mask_ps = (torch.ones_like(rule_tokens) * p)
        # rule_draws = torch.bernoulli(rule_mask_ps).bool()
        # masked_rule_tokens = (rule_tokens & rule_draws)

        # MLM Loss
        ps = self.p * torch.ones_like(seq)
        mlm_mask = (torch.bernoulli(ps).bool() & (seq >= self.num_special_tokens))
        # mask = (mlm_mask | masked_rule_tokens)
        mask = (mlm_mask | rule_tokens)

        # # Debugging
        # if rule_tokens.sum() > 0:
        #     logger.debug(rule_tokens.sum())
        # if mlm_mask.sum() != mask.sum():
        #     logger.debug(f"mlm_mask: {mlm_mask.sum()}")
        #     logger.debug(f"mask: {mask.sum()}")
        #     logger.debug("mask should be larger")

        # Labels
        mlm_labels = seq.clone()
        mlm_labels[~mask] = -100

        # Get masks of how to noise tokens
        a = torch.rand(seq.size())
        mask_token_locs = (mask & (a < .8))
        random_token_locs = (mask & (a > .9))
        num_random = random_token_locs.sum()
        random_tokens = torch.randint(low=self.num_special_tokens, 
                                      high=self.vocab_size, 
                                      size=(num_random.item(),))

        # Noise input ids
        noised_ids = seq.clone()
        noised_ids[mask_token_locs] = self.mask_id
        noised_ids[random_token_locs] = random_tokens

        return noised_ids, mlm_labels
        

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        idx = self.idx_map[i]
        # logger.debug(f"INDEX: {i}, {idx}")

        seq = self.encoded_text[idx]
        attn_mask = self.attention_masks[idx]
        labels = self.labels[idx]
        noisy_labels = self.noisy_labels[idx]
        noised_ids, mlm_labels = self.noise_input_tokens(seq)
        starts = self.word_starts[idx]
        ends = self.word_ends[idx]
        soft_labels = self.soft_labels[idx]
        # noised_ids, mlm_labels = self.noise_input(seq, starts, ends)

        output_dict = {'input_ids': seq, 
                       'attention_masks': attn_mask, 
                       'labels': labels, 
                       'noisy_labels':noisy_labels, 
                       'noised_ids': noised_ids, 
                       'mlm_labels': mlm_labels, 
                       'word_starts':starts, 
                       'word_ends': ends,
                       'soft_labels': soft_labels,
                       }

        # return seq, attn_mask, labels, noisy_labels, noised_ids, mlm_labels, starts, ends
        return output_dict 

    def save(self, filepath):
        '''
        Save data module to file
        '''
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(self, filepath):
        '''
        Load data module from file
        '''
        with open(filepath, 'wb') as f:
            self.__dict__ = pickle.load(f)



class RegalDataset(Dataset):
    # RPN Dataset to mask keywords used in rules
    def __init__(self, 
                 text,
                 encoded_text,
                 attention_masks, 
                 labels,
                 tokenizer, 
                 rules,
                 mask_prob=.1):
        '''
        Initialize dataset class

        Inputs:
            text: List of str
                Input text of datapoints to classify

            labels: List of torch.LongTensor
                Labels corresponding to each datapoint

            tokenizer: 
                Huggingface tokenizer object to encode text

            Rules: List of Rule
                Labeling functions to create noisy labels
        '''
        self.text = data['text']
        self.encoded_text = data['encoded_text']
        self.attention_masks = data['attention_masks']
        self.labels = data['labels']
        self.noisy_labels = data['noisy_labels']

        # Tokenizer attributes
        self.tokenizer = tokenizer
        self.word2id = tokenizer.vocab
        self.mask_id = self.word2id['[MASK]']
    
        # Get vocab size
        self.vocab_size = int(np.max(list(self.word2id.values())) + 1)
        self.num_special_tokens = int(np.max([val for key, val in self.word2id.items() if key.startswith('[')]) + 1)

        # Rule attributes
        self.rule_tokens = rule_tokens
        self.rule_map = {val:val for val in self.word2id.values()}
        self.update_rule_map(rule_tokens)
        self.is_rule = {val:0 for val in self.word2id.values()}
        for w in rule_tokens:
            self.is_rule[self.word2id[w]] = 1
    
        # Misc attributes
        self.p = mask_prob
        self.length = len(self.text)


    def __len__(self):
        '''
        Length attribute
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        Return items from dataset for dataloader
        '''
        seq = self.encoded_text[idx]
        attn_mask = self.attention_masks[idx]
        labels = self.labels[idx]
        noisy_labels = self.noisy_labels[idx]
        noised_ids, mlm_labels = self.noise_input_tokens(seq)

        return seq, attn_mask, labels, noisy_labels, noised_ids, mlm_labels


# class RegalDataModule(pl.LightningDataModule):
#     def __init__(self, 
#                  data_dir, 
#                  tokenizer_name='bert-base-uncased', 
#                  tokenizer_path=None,
#                  ):
#         super(RegalDataModule, self).__init__()

#         if tokenizer_path is not None:
#             self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))

#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    
#     def prepare_data(self,):
#         # Tokenize dataset
#         pass


