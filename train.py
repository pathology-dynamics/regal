import numpy as np 
import torch
from torch import nn
import pandas as pd 
from tqdm import tqdm, trange
import os
from os.path import join as pathjoin
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import ujson as json
import logging 
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer

logger = logging.getLogger('__file__')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

def train(args, 
          model, 
          dataloader, 
          n_rules, 
          optimizer, 
          epoch,
          scheduler=None, 
          cuda=False, 
          warmup=False,
        #   max_iter=None, 
          do_re=False):
    '''
    Perform 1 epoch of RPN training

    Inputs:
    -----------------
        args: dict
            Argparse dictionary of input args into model

        model: nn.Module
            Pytorch model to be trained

        datloader: torch.data.utils.DataLoader
            Loader with training data, noisy labels, etc

        n_rules: int
            How many rules are currently in use

        optimizer: torch.optim object
            Pytorch optimizer used to train the model

        epoch: int
            Epoch number

        scheduler: torch.optim object (Optional: default None)
            Learning rate scheduler to dynamically adjust learning during training

        cuda: bool (Optional: default False)
            Whether to use GPU 

        max_iter:int (Optional: default None)
            Maximum number of training iterations before proposing rules

        do_re: bool (Optional: default False)
            Whether to perform relation extraction

    Outputs:
    -----------------
        metrics: dict
            Training metrics to evaluate performance
    '''
    # Set model in train mode
    model.train()

    all_logits = []
    all_labels = []
    loss = 0
    n = 0
    tot_loss = 0
    tot_n = 0

    max_iter = args.batches_per_epoch

    # pbar = tqdm(dataloader, desc="Loss: None")
    logger.debug("**Entering Train Loop**")
    for i, batch_dict in enumerate(tqdm(dataloader),1):
        if i > args.max_iters:
            i = 0
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

        n_rules = torch.LongTensor([n_rules])
        # covered_mask = torch.BoolTensor((noisy_labels != -1).sum(dim=1) > 0)

        # Use GPU if available
        if cuda:
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

        
        # # Debugging model
        # input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        # input_ids = input_ids.cuda()
        
        # print("Debugging test", input_ids)
        # outputs = model.basemodel(input_ids, labels=input_ids)
        # print("Debugging test output:", len(outputs))

        # # Calculate loss
        # logger.debug(f"Noisy input ids: {noised_ids}")
        # logger.debug(f"mlm_labels: {mlm_labels}")
        # output = model.basemodel(noised_ids, labels=mlm_labels)
        # print("Real model output", output)

        # output_dict = model(noised_ids, attention_masks, noisy_labels, n_rules, covered_mask, mlm_labels)
        output_dict = model(noised_ids, attention_masks, soft_labels, mlm_labels)

        # Get batch loss and update training totals
        soft_label_loss = output_dict['soft_label_loss']
        mlm_loss = output_dict['mlm_loss']
        reg_loss = 0.1 * output_dict['reliability_regularization']

        if warmup:
            batch_loss = mlm_loss
        else:
            batch_loss = soft_label_loss + mlm_loss # + reg_loss 
        loss += batch_loss
        tot_loss += loss.detach()
        
        # Update progress bar
        # pbar.set_description(f"soft_label: {soft_label_loss}\t  MLM: {mlm_loss}")
        # pbar.refresh()

        # Update logits/labels
        all_logits.append(output_dict['logits'].detach())
        all_labels.append(labels)

        # n += output_dict['n']
        # tot_n += output_dict['n'].data

        # Update model params and learning rate
        loss.backward()
        loss = 0
        # n = 0

        # Update model
        optimizer.step()
        optimizer.zero_grad()

        # Update LR scheduler
        if scheduler:
            scheduler.step()

        # if not warmup:
        #     # if n > args.update_size:
        #         # Get average loss across batches
        #         # loss /= n.float() 
        #     loss.backward()
        #     loss = 0
        #     # n = 0

        #     # Update model
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     # Update LR scheduler
        #     if scheduler:
        #         scheduler.step()

        # else:
        #     loss.backward()
        #     loss = 0
        #     # n = 0

        #     # Update model
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     # Update LR scheduler
        #     if scheduler:
        #         scheduler.step()

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels)

    metric_dict = get_metrics(labels, logits, epoch)
    metric_dict['loss'] = tot_loss.detach().cpu()
    return metric_dict
            

def evaluate(args, model, dataloader, epoch, n_rules, cuda=True):
    # Set model in train mode
    model.eval()
    eval_metrics = defaultdict()

    # Init loss and number matched samples
    loss = 0
    n = 0

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, batch_dict in enumerate(tqdm(dataloader),1):
            # Get batch of input
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

            n_rules = torch.LongTensor([n_rules])
            # covered_mask = torch.BoolTensor((noisy_labels != -1).sum(dim=1) > 0)

            # Use GPU if available
            if cuda:
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

            # Calculate loss
            # output_dict = model(clean_input_ids, attention_masks, noisy_labels, n_rules, covered_mask, mlm_labels)
            output_dict = model(clean_input_ids, attention_masks, soft_labels, mlm_labels)

            loss += output_dict['soft_label_loss'] + output_dict['mlm_loss'] + 0.1 * output_dict['reliability_regularization']
            # n += output_dict['n']

            all_logits.append(output_dict['logits'])
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels)

    metric_dict = get_metrics(labels, logits, epoch)
    metric_dict['loss'] = (loss/n).item()
    return metric_dict
    

def evaluate_rules(model, dataloader, tot_rules, start_rule=0):
    pass

def get_metrics(labels, logits, epoch):
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')

    accuracy = accuracy_score(labels, preds)

    metric_dict = {'epoch': epoch,
                   'precision': {i:x for i, x in enumerate(precision)},
                   'recall': {i:x for i, x in enumerate(recall)},
                   'f1': {i:x for i, x in enumerate(f1)},
                   'accuracy': accuracy,
                   'avg_precision': micro_p,
                   'avg_recall': micro_r,
                   'avg_f1': micro_f1}
    return metric_dict

def update_metric_totals():
    pass

def log_metrics(metric_dict, filepath):
    json.dump(metric_dict, open(filepath, 'w'))

def save_checkpoint(model, optimizer, epoch, lfs, model_dir):
    model_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'lfs': lfs
            }
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model_dict, pathjoin(model_dir, f'{epoch}.pt'))
