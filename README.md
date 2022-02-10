# REGAL: Rule-Guided Active Learning for Semi-Automated Weak Supervision

REGAL is a framework for interactive, weakly supervised text classification. 
Starting with a small number of seed rules, REGAL uses transformers to extract 
high-quality labeling functions (LFs) directly from text.  
This transforms the problem of creating LFs to one of simply accepting or 
rejecting LFs created by the model.  This enables users to quickly create a
set of weak labelers for text without the need of manual LF development.  

For more details on how REGAL works, please [check out our paper](https://openreview.net/forum?id=FZDPu3JLEPg)!

<div>
  <img src="regal_diagram.png" width="800">
</div>


# Getting Started
* [Install REGAL's dependencies](#installation) 
* [Download the data](https://figshare.com/articles/dataset/regal_data_zip/19090253) used in the paper from figshare
* [Run REGAL](#usage) on one of the datasets in our paper or train a new model on your own dataset

# Installation


1. Clone a copy of REGAL's repository

```
git clone https://github.com/pathology-dynamics/regal.git
cd regal
```

2. Create a conda environment with REGAL's dependencies.
_NOTE:  You will also need to download stopwords from NLTK_
```
conda env create -f environment.yml
conda activate regal
python -m nltk.downloader stopwords
```

# Usage
To run the models described in the paper, first [download the preprocessed data](https://figshare.com/articles/dataset/regal_data_zip/19090253).  The following script will create the directories and download the data used in the demos:

```
bash download_data.sh
```

Assuming you have a [properly formatted dataset](#use-your-own-dataset) located at `data/yelp.pt`, you ca run REGAL from the command line as follows:
```
python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --output yelp_output
```

Examples of running REGAL with a variety of options can be found in the examples in `scripts/`

# Use your own dataset
To use your own dataset, simply format your data as a dictionary as described below and save using `torch.save`:
```
{
    'train': {
        'text': [
            'Text of first train example.',
            'Text of second train example.',
           ...
        ]
        'labels': torch.LongTensor([0,1,1,0,2,...])
    }

    'valid': {
        'text': [...]
        'labels': torch.LongTensor([...])
    }

    'test': {
        'text': [...]
        'labels': torch.LongTensor([...])
    }

    'class_names': {
        0: 'class_0_name',
        1: 'class_1_name',
        ...
    }

    'rule_keywords': {
        0: ['keyword_1', 'keyword_2', 'keyword_3'],
        1: ['keyword_4', 'keyword_5', 'keyword_6'],
    }
}
```
**Note: Due to the identifiability constraints of Snorkel, you will need to have at least 3 keywords for each class**

To train REGAL on your dataset, use `run_rpn.py`
```
usage: run_rpn.py [-h] --data_path DATA_PATH [--output_dataset OUTPUT_DATASET]
                  [--refresh_data] [--logdir LOGDIR]
                  [--checkpoint_dir CHECKPOINT_DIR]
                  [--tokenizer_path TOKENIZER_PATH]
                  [--num_classes NUM_CLASSES] [--batch_size BATCH_SIZE]
                  [--metric {precision,recall,f1,accuracy,avg_precision,avg_recall,avg_f1}]
                  [--true_labels TRUE_LABELS] [--min_lf MIN_LF]
                  [--update_size UPDATE_SIZE] [--freeze] [--max_norm MAX_NORM]
                  [--input_size INPUT_SIZE] [--max_len MAX_LEN]
                  [--fc_size FC_SIZE] [--max_rules MAX_RULES]
                  [--num_epochs NUM_EPOCHS] [--lr LR]
                  [--weight_decay WEIGHT_DECAY] [--max_iters MAX_ITERS]
                  [--max_rule_iters MAX_RULE_ITERS]
                  [--num_workers NUM_WORKERS] [--model_no MODEL_NO]
                  [--seed SEED] [--rules_per_epoch RULES_PER_EPOCH]
                  [--batches_per_epoch BATCHES_PER_EPOCH]
                  [--retokenize RETOKENIZE] [--debug]
                  [--warmup_epochs WARMUP_EPOCHS]
                  [--max_rule_length MAX_RULE_LENGTH]
                  [--min_count_cutoff MIN_COUNT_CUTOFF]
                  [--polarity_thresh POLARITY_THRESH] [--autoeval]
                  [--autoeval_thresh AUTOEVAL_THRESH] [--oracle]
                  [--num_autoeval_examples NUM_AUTOEVAL_EXAMPLES] [--refresh]
                  [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to data dictionary with train, test, and
                        validation data
  --output_dataset OUTPUT_DATASET
                        Name of dataset for saving data
  --refresh_data        Reprocess dataset from scratch, even if it has been
                        processed and cached
  --logdir LOGDIR       Path to directory to log performance after each epoch
  --checkpoint_dir CHECKPOINT_DIR
                        Path to model checkpoints
  --tokenizer_path TOKENIZER_PATH
                        Path to pretrained tokenizer
  --num_classes NUM_CLASSES
                        Number of classes in the data
  --batch_size BATCH_SIZE
                        Training batch size
  --metric {precision,recall,f1,accuracy,avg_precision,avg_recall,avg_f1}
  --true_labels TRUE_LABELS
                        Indicator of whether train dataset has true labels
                        included
  --min_lf MIN_LF       Minimum number of matched LFs required for a sample to
                        be included in labeled data subset
  --update_size UPDATE_SIZE
                        Number of matched samples over which to accumulate
                        gradient
  --freeze              Freeze layers of BERT encoer to speed training
  --max_norm MAX_NORM   Clipped gradient norm
  --input_size INPUT_SIZE
                        Size of input into RPN. Should be dimension of
                        embeddings returned for individual words.
  --max_len MAX_LEN     Maximum number of tokens allowed in example
  --fc_size FC_SIZE     Size of fully connected layer in rpn
  --max_rules MAX_RULES
                        Maximum number of rules
  --num_epochs NUM_EPOCHS
                        No of epochs
  --lr LR               Learning rate
  --weight_decay WEIGHT_DECAY
                        L2 regularization constant
  --max_iters MAX_ITERS
                        Maximum number of iters of training before adding new
                        rules
  --max_rule_iters MAX_RULE_ITERS
                        Number of batches used to generate rules
  --num_workers NUM_WORKERS
                        Number of dataloading cores
  --model_no MODEL_NO   Model ID: 0 - BERT 1 - ALBERT 2 - SciBERT 3 - BioBERT
  --seed SEED           Random seed for reproducibility
  --rules_per_epoch RULES_PER_EPOCH
                        Number of rules to propose for each class at every
                        epoch
  --batches_per_epoch BATCHES_PER_EPOCH
                        Number of minibatches to run before updating rules
                        (default: whole training set)
  --retokenize RETOKENIZE
                        Rerun tokenization at runtime
  --debug               Only use 1000 samples for debugging purposes
  --warmup_epochs WARMUP_EPOCHS
                        Number of epochs to train on small labeled data before
                        generating rules
  --max_rule_length MAX_RULE_LENGTH
                        Maximum words in a phrase-based rule
  --min_count_cutoff MIN_COUNT_CUTOFF
                        Minimum number of times a phrase must occur in
                        training set to be considered for a rule
  --polarity_thresh POLARITY_THRESH
                        Minimum difference between prob of #1 and #2 classes
                        for phrase to be considered for a rule
  --autoeval            Use automatic evaulation rather than having humans
                        accept/reject rules
  --autoeval_thresh AUTOEVAL_THRESH
                        Minimum threshold of matched instances having correct
                        label in order to accept rule
  --oracle              Use oracle rule quality evaluation instead of sampling
  --num_autoeval_examples NUM_AUTOEVAL_EXAMPLES
                        Number of examples to labels for autoeval
  --refresh             Reset model weights after each epoch
  --alpha ALPHA         Alpha value to balance coverage and precision for
                        rules. Higher favors more class specificity whereas
                        lower favors higher coverage.
```




