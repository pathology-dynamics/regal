import torch
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
from transformers import BertModel, BertForMaskedLM, BertTokenizer
from transformers import MobileBertModel, MobileBertForMaskedLM, MobileBertTokenizer
from collections import defaultdict
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score


def get_tokenizer(tokenizer_path=None, model=None, lower_case=None):
    # Load saved tokenizer
    if tokenizer_path:
        print("***\n\n\n***WRONG TOKENIZER***\n\n\n")
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    # Load tokenizer from specific model
    else:
        print("***\n\nLoading Autotokenizer\n\n***")
        tokenizer = AutoTokenizer.from_pretrained(model)

    # Put some stuff here to deal with extra tokens when it becomes relevant

    return tokenizer

def get_model_info(model_no):
    if model_no == 0:
        # from .model.BERT.modeling_bert import BertModel as Model
        model = 'bert-base-uncased'
        lower_case = True
        model_name = 'BERT'
    elif model_no == 1:
        # from .model.ALBERT.modeling_albert import AlbertModel as Model
        model = 'albert-base-v2'
        lower_case = False
        model_name = 'ALBERT'
    elif model_no == 2:
        # from .model.BERT.modeling_bert import BertModel as Model
        model = 'scibert-scivocab-uncased'
        lower_case = False
        model_name = 'SciBERT'
    elif model_no == 3:
        model = 'biobert-biovocab-cased'
        lower_case = False
        model_name = 'BioBERT'

    return model, lower_case, model_name


def get_model_and_tokenizer(model_no):
    '''
    Return pretrained model and accompanying tokenizer from huggingface
    '''
    if model_no == 0:
        tokenizer = PreTrainedTokenizerFast.from_pretrained('google/mobilebert-uncased')
        model = MobileBertForMaskedLM.from_pretrained('google/mobilebert-uncased', 
                                                output_attentions=False,
                                                output_hidden_states=True,
                                                return_dict=True)
    else:
        model_cards = {1: 'bert-base-uncased', 2: 'scibert-scivocab-uncased', 3:'biobert-biovocab-cased'}
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_cards[model_no])
        model = BertForMaskedLM.from_pretrained(model_cards[model_no], 
                                                output_attentions=False,
                                                output_hidden_states=True,
                                                return_dict=True)

    tokenizer.pad_token = '[PAD]'
    return tokenizer, model

def get_model_output(rpn):
    dataset_name = rpn.args.data_path.split('/')[-1].split('.')[0]
    lf_dict = defaultdict(list)
    for (class_label, phrase_list) in rpn.rule_dict.items(): 
        print(phrase_list)
        for phrase in phrase_list:
            lf_dict[class_label].append(phrase)
    lf_dict = dict(lf_dict)

    # Train snorkel model
    label_matrix = rpn.train.full_noisy_labels.numpy()
    mask = np.array((label_matrix >= 0).sum(1) > 0).flatten()

    # convert to snorkel format
    noisy_labels = np.array(label_matrix)[mask]

    # True labels
    labels = rpn.train.labels.numpy()[mask]

    # Get stats of 
    majority_vote = (noisy_labels > 0).sum(axis=1) / (noisy_labels >= 0).sum(axis=1)

    # Get analysis of labeling functions
    analysis = LFAnalysis(noisy_labels).lf_summary(Y=labels)

    # Fit label model
    lm = LabelModel(cardinality=rpn.n_classes)
    lm.fit(noisy_labels)
    preds = lm.predict(noisy_labels)
    probs = lm.predict_proba(noisy_labels)

    # Get accuracy and AUC
    accuracy = accuracy_score(y_pred=preds, y_true=labels)
    mv_accuracy = accuracy_score(y_pred=majority_vote.round(), y_true=labels)
    if rpn.n_classes == 2:
        auc = roc_auc_score(y_score=probs[:,1], y_true=labels)
        mv_auc = roc_auc_score(y_score=majority_vote, y_true=labels)
    else:
        mv_auc = 'N/A'
        auc = 'N/A'

    # Collect outputs in dictionary
    output_dict = {
        'dataset': dataset_name,
        'label_matrix': label_matrix,
        'labeled_mask': mask,
        'labels': labels, 
        'coverage': mask.sum()/labels.shape[0],
        'analysis_df': analysis,
        'lm_accuracy': accuracy,
        'lm_auc': auc,
        'mv_accuracy': mv_accuracy,
        'mv_auc': mv_auc,
        'lf_dict': lf_dict,
        'n_lfs':noisy_labels.shape[1],
    }

    return output_dict