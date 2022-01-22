from rpn import main
from argparse import ArgumentParser
import logging
import torch
import numpy as np


logging.basicConfig(format='%(asctime)s [%(levelname)s] (%(filename)s): %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
logger = logging.getLogger('__file__')

# Set random seed for reproducibility
torch.manual_seed(641)
np.random.seed(641)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data dictionary with train, test, and validation data", required=True)
    parser.add_argument("--output_dataset", type=str, default='yelp', help='Name of dataset for saving data')
    parser.add_argument("--refresh_data", action='store_true', help='Reprocess dataset from scratch, even if it has been processed and cached')
    parser.add_argument("--logdir", type=str, default='logs', help='Path to directory to log performance after each epoch')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpts', help='Path to model checkpoints')
    parser.add_argument("--tokenizer_path", type=str, help="Path to pretrained tokenizer")
    parser.add_argument("--num_classes", type=int, default=19, help='Number of classes in the data')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument('--metric', 
                        choices=['precision',
                                'recall',
                                'f1',
                                'accuracy',
                                'avg_precision',
                                'avg_recall',
                                'avg_f1',
                                ], 
                        default='accuracy')

    parser.add_argument("--true_labels", type=bool, default=True, help="Indicator of whether train dataset has true labels included")
    parser.add_argument('--min_lf', type=int, default=2, help='Minimum number of matched LFs required for a sample to be included in labeled data subset')
    parser.add_argument("--update_size", type=int, default=64, help="Number of matched samples over which to accumulate gradient")
    parser.add_argument('--freeze', action='store_true', help='Freeze layers of BERT encoer to speed training')
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--input_size", type=int, default=768, help="Size of input into RPN.  Should be dimension of embeddings returned for individual words.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum number of tokens allowed in example")
    parser.add_argument('--fc_size', type=int, default=256, help="Size of fully connected layer in rpn")
    parser.add_argument("--max_rules", type=int, default=400, help="Maximum number of rules")
    parser.add_argument("--num_epochs", type=int, default=5, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="L2 regularization constant")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum number of iters of training before adding new rules")
    parser.add_argument("--max_rule_iters", type=int, default=1000, help="Number of batches used to generate rules")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloading cores")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
        1 - ALBERT\n
        2 - SciBERT\n
        3 - BioBERT''')
    # parser.add_argument("--do_re", type=bool, default=False, help="Flag to indicate that we are training a relation extraction model")
    # parser.add_argument("--auto", action='store_true', help="Whether to let the model autoselect rules without human input")
    parser.add_argument('--seed', type=int, default=619, help='Random seed for reproducibility')
    parser.add_argument('--rules_per_epoch', type=int, default=40, help='Number of rules to propose for each class at every epoch')
    parser.add_argument('--batches_per_epoch', type=int, help="Number of minibatches to run before updating rules (default: whole training set)")
    parser.add_argument('--retokenize', type=bool, default=False, help="Rerun tokenization at runtime")
    parser.add_argument('--debug', action='store_true', help='Only use 1000 samples for debugging purposes')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of epochs to train on small labeled data before generating rules')
    parser.add_argument('--max_rule_length', type=int, default=3, help='Maximum words in a phrase-based rule')
    parser.add_argument('--min_count_cutoff', type=int, default=20, help='Minimum number of times a phrase must occur in training set to be considered for a rule')
    parser.add_argument('--polarity_thresh', type=float, default=0.25, help='Minimum difference between prob of #1 and #2 classes for phrase to be considered for a rule')
    parser.add_argument('--autoeval', action='store_true', help='Use automatic evaulation rather than having humans accept/reject rules')
    parser.add_argument('--autoeval_thresh', type=float, default=.7, help="Minimum threshold of matched instances having correct label in order to accept rule")
    parser.add_argument('--oracle', action='store_true', help='Use oracle rule quality evaluation instead of sampling')
    parser.add_argument('--num_autoeval_examples', type=int, default=10, help='Number of examples to labels for autoeval')
    parser.add_argument('--refresh', action='store_true', help='Reset model weights after each epoch')
    parser.add_argument('--alpha', type=float, default=.7, help="Alpha value to balance coverage and precision for rules.  Higher favors more class specificity whereas lower favors higher coverage.")

    
    args = parser.parse_args()

    logger.info(f"Num Workers: {args.num_workers}")

    main(args)