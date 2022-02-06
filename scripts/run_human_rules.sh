python run_rpn.py --data_path datasets/yelp/yelp_6.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 500 --max_rule_iters 500 --refresh --dataset_name yelp_human --num_epochs 3

python run_rpn.py --data_path datasets/imdb/imdb_6.pt --num_classes 2 --batch_size 128 --warmup_epochs 0 --max_iters 1000 --max_rule_iters 1000 --refresh --dataset_name imdb_human --num_epochs 3

python run_rpn.py --data_path datasets/ag/ag_6.pt  --num_classes 4 --batch_size 128 --warmup_epochs 0 --max_iters 5000 --max_rule_iters 5000 --polarity_thresh .1 --refresh --dataset_name ag_human --num_epochs 3