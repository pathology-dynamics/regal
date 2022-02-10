#!/bin/bash/

python run_rpn.py --data_path temp/professor_physician.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --oracle --model_no 1 --output professor_physician --autoeval --max_rule_length 1 --min_lf 2 