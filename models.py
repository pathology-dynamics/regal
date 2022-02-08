from typing import List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from snorkel.classification import cross_entropy_with_probs
import numpy as np
import logging

logger = logging.getLogger('__file__')


Outputs = Mapping[str, List[torch.Tensor]]


def masked_softmax(vector,
                   mask,
                   dim=-1,
                   memory_efficient=False,
                   mask_fill_value=-1e32) -> torch.Tensor:
    """
    Performs softmax over values not excluded by mask.

    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(-1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            if mask.size(-1) != vector.size(-1):
                mask = torch.cat(vector.size(-1) * [mask], dim=-1)
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

def masked_cross_entropy_with_probs(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.BoolTensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    '''
    Cross entropy loss with probabilistic targets where 
    masked values are omitted from computation

    Returns:
        loss: 
            Cross entropy loss with soft labels on unmasked subset
        n: 
            Number of unmasked tokens
    '''
    logits_subset = logits[mask]
    target_subset = target[mask]
    n = mask.sum()
    if n > 0:
        loss = cross_entropy_with_probs(logits_subset, target_subset, reduction=reduction)
    else:
        loss = 0
    return loss, n


class WordAttention(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size):
        super(WordAttention, self).__init__()
        # self.attn_mat = nn.init.xavier_normal_(torch.empty(n_classes, input_size))
        self.n_classes = n_classes
        self.class_mat = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_classes, input_size)), requires_grad=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(.75)
        self.weighted_class_sum = nn.Linear(n_classes, 1, bias=False)
        # self.fc_out = nn.Linear


    def forward(self, hidden_states, masks, use_oppontent=False):
        '''
        Inputs:
            hidden_states: (batch x max_len, hidden_dim)
                Hidden states from encoder (probably a BERT-based LM)

            masks: (batch x max_len)
                Attention masks for which words are not padded.  Only use hidden states corresponding to nonzero values in mask
        Returns:


        '''
        # logger.debug(f"Hidden States: {hidden_states.size()}")
        # We use an MLP to get attention
        # Get attention.  Should have shape (batch_size x max_len x n_classes)
        z0 = self.dropout(torch.tanh(self.fc1(hidden_states)))
        attn_logits = self.fc2(z0)

        # Check shape is what we want

        # Calculate attention
        attention = masked_softmax(attn_logits, masks, dim=1)

        # Should have shape (batch_size x n_classes x input_size)
        class_reps = torch.sum(attention.unsqueeze(-1) * hidden_states.unsqueeze(-2), dim=1)

        # Get vector representing each document
        doc_vector = self.weighted_class_sum(class_reps.transpose(1, 2)).squeeze()
        

        # Logits should have shape(batch_size x n_classes)
        logits = torch.sum(class_reps * self.class_mat, dim=2)
        

        # Get output probabilities for each class
        probs = F.softmax(logits, dim=1)

        return logits, probs, attention, doc_vector


class AttentionModel(nn.Module):
    def __init__(self, input_size, max_rules, fc_size, nclass):
        super(AttentionModel, self).__init__()
        self.max_rules = max_rules
        self.nclass = nclass
        self.fc1 = nn.Linear(input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, max_rules)
        self.dropout = nn.Dropout(.75)
        

    def forward(self, x_lf, x_l, n_rules):
        '''
        Run inputs through network module

        Inputs:
            x_lf: 
                Output of labeling functions on x
            x_l: 
                Embedded representation of each sentence to use as input features
            n_rules:
                Number of rules that have currently been specified
        '''
        # logger.debug(x_l.size())
        # logger.debug(mask.size())

        # Mask of which rules are in use
        mask = torch.zeros(x_lf.size(0), self.max_rules, device=x_lf.device)
        mask[:, :n_rules] = (x_lf >= 0).float()

        # Features of document and which rules apply
        # We concatenate the applicable rules to the input representation
        # to help our model learn from their correlations
        x = x_l
        z0 = self.dropout(torch.tanh(self.fc1(x)))
        z = self.fc2(z0)
        
        # Get mask of which rules are currently in use
        rules_mask = torch.zeros_like(z)
        rules_mask[:, :n_rules] = 1
        score = masked_softmax(z, rules_mask, dim=1)
        coverage_score = score * mask #conditional labeling source score A_i

        score_matrix = torch.empty(x_lf.size(0), self.nclass, device=x_lf.device)
        for k in range(self.nclass):
            score_matrix[:, k] = (score[:, :n_rules] * (x_lf == k).float()).sum(dim=1)

        score_matrix[score_matrix == 0] = -10000
        softmax_new_y = F.softmax(score_matrix, dim=1) #weighted soft predictions
        return softmax_new_y, coverage_score



class AssembleModel(nn.Module):
    def __init__(self, basemodel, fc_size, max_rules, n_class, use_bert_attention=False, attention_layer=-2):
        '''
        Inputs:
        --------------------
            basemodel: transformers.BertModel
                Huggingface transformers model from which to get token-level embeddings
            input_size: int
        '''
        super(AssembleModel, self).__init__()
        self.n_class = n_class
        self.max_rules = max_rules
        self.basemodel = basemodel
        self.word_attention = WordAttention(basemodel.config.hidden_size, n_class, fc_size)
        self.rule_attention = AttentionModel(input_size=basemodel.config.hidden_size, max_rules=max_rules, fc_size=fc_size, nclass=n_class)
        self.use_bert_attention = use_bert_attention
        self.attention_layer = attention_layer

    # def forward(self, input_ids, attention_masks, doc_labels, n_rules, covered_mask, mlm_labels):
    def forward(self, input_ids, attention_masks, soft_labels, mlm_labels):
        '''
        Inputs:
            input_ids: 
                Sequence of token ids corresponding to document to be labeled

            attention_masks:
                Sequence of attention masks to avoid calculating on <PAD> tokens

            doc_labels:
                batch_size x n_rules array of noisy labels from labeling rules

            n_rules:
                Current number of active rules

            covered_mask:
                Mask of which examples are currently covered by at least one rule

            mlm_labels:
                Sequence labels for masked language modeling loss
        '''
        
        # last_states, cls_token = self.basemodel(input_ids=input_ids,
        #                                    attention_mask=attention_masks, 
        #                                    token_type_ids=None)
        out = self.basemodel(input_ids=input_ids,
                                attention_mask=attention_masks, 
                                token_type_ids=None,
                                labels=mlm_labels,
                                return_dict=True)
        mlm_loss = out.loss
        mlm_logits = out.logits
        hidden_states = out.hidden_states
        last_states = hidden_states[-1]
        # logger.debug(f"States: {last_states}") 
        logits, probs, attention, doc_embs = self.word_attention(last_states, attention_masks)

        # logger.debug(f"doc_labels: {doc_labels.size()}")
        if len(doc_embs.size()) == 1:
            doc_embs = doc_embs.unsqueeze(0)

        
        # soft_labels, reliability_scores = self.rule_attention(doc_labels, doc_embs, n_rules)
        # logger.debug(f"Soft labels: {soft_labels}")
        # logger.debug(f"Reliability scores: {reliability_scores}")

        # reliability_prior = torch.zeros_like(reliability_scores)
        # reliability_prior[:,:n_rules] = 1./n_rules
        # reliability_regularization_loss = ((reliability_scores - reliability_prior)**2).mean()
        reliability_regularization_loss = 0

        ### Hard Labels Experiment ###
        # Normalize rows to sum to 1
        # ordered_vals, _ = (soft_labels).topk(dim=1, k=2)
        # polarity_vals = ordered_vals[:,0] - ordered_vals[:,1]
        # polarity_mask = polarity_vals > self.polarity_thresh

        ### End Hard Labels Experiment ###

        # soft_label_loss, n_contributing =  masked_cross_entropy_with_probs(logits, soft_labels, covered_mask, reduction='mean')
        soft_label_loss = cross_entropy_with_probs(logits, soft_labels, reduction='mean')
        # soft_label_loss = F.cross_entropy(logits, soft_labels)
        n_contributing = logits.size(0)
        

        output_dict = {'soft_label_loss': soft_label_loss,
                       'mlm_loss': mlm_loss,
                       'reliability_regularization': reliability_regularization_loss,
                       'n': n_contributing,
                       'soft_labels': soft_labels,
                       'logits':logits,
                       'probs':probs,
                       'attention':attention,
                    #    'covered_mask':covered_mask
                       }

        return output_dict
