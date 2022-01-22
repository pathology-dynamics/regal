import torch
import numpy as np
import pandas as pd
import glob
import ujson
import os

from snorkel.labeling import LabelingFunction
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier, LFApplier
from snorkel.preprocess import preprocessor
from tqdm.auto import tqdm, trange
from argparse import ArgumentParser
from snorkel_utils import keyword_lookup, make_keyword_lf


def split_data(data, label_matrix):
    '''
    Split data into labeled and unlabeled according to a particular rule-induced label set
    '''
    train = data['train']
    
    mask = ((torch.LongTensor(label_matrix) >= 0).sum(dim=1) > 0).bool()
    print(mask.sum())
    data['labeled'] = {}
    data['unlabeled'] = {}
    for key, val in train.items():
        if torch.is_tensor(val):
            data['labeled'][key] = val[mask]
            data['unlabeled'][key] = val[~mask]
            # print(data['unlabeled'][key].size())
        else:
            data['labeled'][key] = [t for (i,t) in zip(mask, val) if i == 1]
            data['unlabeled'][key] = [t for (i,t) in zip(mask, val) if i == 0]
            # print(len(data['unlabeled'][key]))
    # data['labeled'] = {key: val[mask] for key, val in train.items() if key != 'text'}
    # data['labeled']['text'] = [t for (i,t) in zip(mask, train['text']) if i == 1]
    # data['unlabeled'] = {key: val[~mask] for key, val in train.items() if key != 'text'}
    # data['unlabeled']['text'] = [t for (i,t) in zip(mask, train['text']) if i == 0]

    return data



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data dictionary with train, test, and validation data")
    parser.add_argument('--rule_dict_path', type=str, required=True, help='Path to .json file of keyword rules for each class')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output data')
    parser.add_argument('--split', action='store_true', help='Split data into labeled and unlabeled for use with multi-source weak supervision model')

    args = parser.parse_args()

    data = torch.load(args.data_path)
    lf_kwds = ujson.load(open(args.rule_dict_path, 'r'))
    # print([(w, int(key)) for (key, vals) in lf_kwds.items() for w in vals])
    lfs = [make_keyword_lf(w, int(key), rpn_generated=False) for (key, vals) in lf_kwds.items() for w in vals]

    label_matrix = LFApplier(lfs).apply(data['train']['text'])
    analysis_df = LFAnalysis(label_matrix).lf_summary(Y=data['train']['labels'].numpy())
    print(analysis_df)
    # print(data['train']['noisy_labels'].size())
    # print(torch.LongTensor(label_matrix).size())
    data['train']['noisy_labels'] = torch.LongTensor(label_matrix)

    for data_slice in ['test','valid']:
        noisy_labels = LFApplier(lfs).apply(data[data_slice]['text'])
        data[data_slice]['noisy_labels'] = torch.LongTensor(noisy_labels)

    used_kwds = [w for (key, vals) in lf_kwds.items() for w in vals]
    data['rule_keywords'] = used_kwds
    

    
    # Split data if desired
    if args.split:
        data = split_data(data, label_matrix)

    # Save data
    dir = os.path.dirname(args.save_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    torch.save(data, args.save_path)
        
        
        
