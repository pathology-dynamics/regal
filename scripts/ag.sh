#!/bin/bash/

# python test_rpn_dataloader.py --data_path datasets/ag_data_rpn.pt --logdir logs --checkpoint_dir ckpts/yelp --num_classes 4 --batch_size 128 --warmup_epochs 0 

# python run_rpn.py --data_path datasets/ag/ag_1.pt  --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --polarity_thresh .1  --autoeval --refresh --output_dataset ag_1

# python run_rpn.py --data_path datasets/ag/ag_3.pt  --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --polarity_thresh .1  --autoeval --refresh --output_dataset ag_3

# python run_rpn.py --data_path datasets/ag/ag_6.pt  --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --polarity_thresh .1  --autoeval --refresh --output_dataset ag_6

python run_rpn.py --data_path temp/ag.pt --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --refresh --oracle --model_no 1 --output ag_new --autoeval --max_rule_length 1 --min_lf 1  --polarity_thresh .125 # --debug #--refresh_data

# python run_rpn.py --data_path temp/ag.pt --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --refresh --oracle --model_no 1 --output ag_new --autoeval --max_rule_length 2 --min_lf 2  --polarity_thresh .125 --debug #--refresh_data

# python run_rpn.py --data_path temp/ag.pt --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --refresh --oracle --model_no 1 --output ag_new --autoeval --max_rule_length 3 --min_lf 2  --polarity_thresh .125 --debug #--refresh_data
