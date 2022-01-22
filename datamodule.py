# Imports
import torch
import numpy as np
import logging
import pickle
import os
# import pytorch_lightning as pl

# Submodules
from typing import Union, List
from tqdm import tqdm, trange
from torch.utils.data import Dataset, TensorDataset

# Need to set tokenizers_parallelism environment variable to avoid lots of warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logger = logging.getLogger('__file__')

class RPNCollate():
    def __init__(self, id2word):
        self.id2word = id2word

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
        batch_words = reconstruct_words(input_ids, starts, ends, self.id2word, batch_inds=batch_inds)


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