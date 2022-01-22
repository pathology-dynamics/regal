import torch
import numpy as np 

from collections import defaultdict as dd


def accumulate_phrase_scores(words, word_scores, batch_inds, k, precomputed_phrases=None):
    '''
    Accumulate total phrase scores for phrases of length k
    Tests passing.
    
    Inputs:
    -------
        words: list of str, size=(total_words,)
            Words in batch (in order)
            
        word_scores: torch.FloatTensor, size=(total_words,)
            Scores of each word
            
        batch_inds: torch.LongTensor, size=(total_words,)
            Batch index of each word.  Used to prevent us from 
            phrases from spans spanning documents
            
        k: int
            Length of phrases
            
    Returns:
    --------
        phrase_scores: dict
            Dict mapping phrases to score of each
            
        phrase_counts: dict
            Dict mapping phrases to count of each
    '''
    # Use cumulative sum to efficiently get scores of each chunk
    # Prepend a 0 so we still get the first span
    scores = torch.zeros((word_scores.size(0) + 1, word_scores.size(1)))
    scores[1:,:] = word_scores.cumsum(0)
    chunk_scores = scores[k:,:] - scores[:-k,:]
    
    phrases = [' '.join(words[i:i+k]) if batch_inds[i]==batch_inds[i+k-1] else None for i in range(len(words)-(k-1))]
    
    # Condense into dict and eliminate cross_s
    phrase_scores = dd(float)
    # phrase_counts = dd(float)
    for phrase, score in zip(phrases, chunk_scores):
        if phrase is None:
            continue

        # Check if phrase is in acceptable set
        if precomputed_phrases is not None:
            if phrase not in precomputed_phrases:
                continue
        # Otherwise, update score
        phrase_scores[phrase] += score
        # phrase_counts[phrase] += 1
        
    # return phrase_scores, phrase_counts
    return phrase_scores



def get_word_scores(attentions, batch_inds, word_inds, word_mask):
        '''
        Get score for each span of tokens
        
        Inputs:
        -------
            attentions: torch.FloatTensor, size=(batch, seq_len, n_classes)
                Attention weights of each token
                
            batch_inds: torch.LongTensor, size=(total_words, 1)
                Batch index of each word. Each row corresponds to one word
                
            word_inds: torch.LongTensor (total_words, max_word_len)
                Index in sequence of each token in a particular word
                Each row corresponds to a single word
                
            word_mask: torch.tensor(total_words, max_word_length)
                Mask to zero out attentions for wordw with less than 
                    max_length wordpiece tokens
                
        Returns:
        --------
            word_scores: torch.FloatTensor, size=(total_words,)
                Scores for each word in order they appear
        '''
        word_scores = (attentions[batch_inds, word_inds, :] * word_mask.unsqueeze(-1)).sum(1)
        return word_scores