#!/bin/bash/

python run_rpn.py --data_path temp/ag.pt --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --refresh --oracle --model_no 1 --output ag_new --autoeval --max_rule_length 1 --min_lf 1  --polarity_thresh .125 
