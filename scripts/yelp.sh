#!/bin/bash/

# python qun_rpn.py --data_path datasets/yelp_data_rpn.pt --logdir logs --checkpoint_dir ckpts/yelp --num_classes 2 --batch_size 64 --warmup_epochs 0 --max_iters 500

# python run_rpn.py --data_path datasets/yelp/yelp_1.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 500 --max_rule_iters 500 --autoeval --refresh --output_dataset yelp_1

# python run_rpn.py --data_path datasets/yelp/yelp_6.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 500 --max_rule_iters 500 --autoeval --refresh --output_dataset yelp_6

# python run_rpn.py --data_path datasets/yelp/yelp_3.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 500 --max_rule_iters 500 --autoeval --refresh --output_dataset yelp_3

# python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output yelp  --max_rule_length 1 --min_lf 1 #--autoeval #--refresh_data #--debug

# python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output yelp --autoeval --max_rule_length 1 --min_lf 1 #--refresh_data #--debug

# python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --batch_size 128 --warmup_epochs 1 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output yelp_warmup_1 --autoeval --max_rule_length 1 --min_lf 1 #--refresh_data #--debug

python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output yelp --autoeval --max_rule_length 1 --min_lf 2  #--refresh_data #--debug

# python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output yelp --autoeval --max_rule_length 2 --min_lf 2  #--refresh_data #--debug

# python run_rpn.py --data_path temp/yelp.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output yelp --autoeval --max_rule_length 3 --min_lf 2  #--refresh_data #--debug


