#!/bin/bash/

# python run_rpn.py --data_path datasets/imdb/imdb_1.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --autoeval --refresh --output_dataset imdb_1

# python run_rpn.py --data_path datasets/imdb/imdb_3.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --autoeval --refresh --output_dataset imdb_3

# python run_rpn.py --data_path datasets/imdb/imdb_6.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --autoeval --refresh --output_dataset imdb_6

python run_rpn.py --data_path temp/imdb.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output imdb_new --autoeval --max_rule_length 1 --min_lf 2 --polarity_thresh .22 #--refresh_data 

# python run_rpn.py --data_path temp/imdb.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output imdb_new --autoeval --max_rule_length 2 --min_lf 2 #--refresh_data

# python run_rpn.py --data_path temp/imdb.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output imdb_new --autoeval --max_rule_length 3 --min_lf 2 #--refresh_data