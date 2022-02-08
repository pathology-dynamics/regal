import pandas as pd
import torch
from snorkel.labeling import PandasLFApplier, LFApplier
# from snorkel.labeling.apply.dask import PandasParallelLFApplier
from tqdm.auto import tqdm, trange
import pickle
# tqdm.pandas()


def prepare_data(args, dict_path, tokenizer, preprocessed_dir=None):
    '''
    Load data dict.  Should have the following structure:
        {
        train: {
            text: torch.Tensor
                Text of sentence/document to be classified
            labels: torch.Tensor (optional)
                Tensor of ground-truth labels 
            noisy_labels: torch.Tensor (optional)
                Precomputed noisy labels
            }

        valid: {
            text: torch.Tensor
                Text of sentence/document to be classified
            labels: torch.Tensor (optional)
                Tensor of ground-truth labels 
            }

        test: {
            text: torch.Tensor
                Text of sentence/document to be classified
            labels: torch.Tensor (optional)
                Tensor of ground-truth labels 
            }

        lfs: list of func
            List of label functions to apply to dataset
        }
    '''
    d = torch.load(dict_path)

    data_tuples = []
    lfs = d['lfs']
    for key in ['train','test','valid']:
        data = d[key]
        ids, attn_masks = prepare_text(data['text'], tokenizer)
        noisy_labels = get_noisy_labels(data['text'], lfs)
        noisy_labels = torch.tensor(noisy_labels)
        data_tuples.append((ids, attn_masks, noisy_labels))

    return data_tuples, lfs


def get_noisy_labels(text, lfs, pandas=False, n_cores=16):
    if pandas:
        applier = PandasLFApplier(lfs=lfs)
        noisy_labels = applier.apply(text, n_cores)
    else:
        applier = LFApplier(lfs=lfs)
        noisy_labels = applier.apply(text)
    return noisy_labels

def seed_rule_tokens(seed_kwds, tokenizer):
    '''
    Get all of the tokens in keywords used to make initial rules
    '''
    tokens = tokenizer.encode(" ".join(seed_kwds), add_special_tokens=False, truncation=True)
    return tokens


def prepare_text(sentences, 
                tokenizer,
                tokenizer_save_path=None,
                new_tokens=None, 
                entity_tags=None,
                # substrings_to_mask=None, 
                max_len=128):

    # Add new tokens if available
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        if tokenizer_save_path is not None:
            pickle.dump(tokenizer, open(tokenizer_save_path, 'wb'))
    
    if entity_tags:
        tag_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(entity_tags)))


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    entity_starts = []

    # Encode, pad, and truncate sentences
    for sent in tqdm(sentences):
        encoded_dict = tokenizer(
                            sent,                      # Sentence to encode.
                            # add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length=max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation=True,
                    )
        
        # Get entity tags if available
        if entity_tags:
            entity_inds = [i for i, e in enumerate(encoded_dict['input_ids'][0]) if e in tag_tokens]
            if len(entity_inds) < 2:
                if len(entity_inds) < 1:
                    raise Exception("No entity markers found in sentence!")
                else:
                    entity_inds.append(max_len-1)
            entity_starts.append(entity_inds)


        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])


    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if entity_tags:
        entity_starts = torch.tensor(entity_starts)
        return input_ids, attention_masks, entity_starts
    
    return input_ids, attention_masks


