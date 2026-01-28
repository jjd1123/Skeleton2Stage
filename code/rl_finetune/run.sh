source ../../environment.sh
accelerate launch --multi_gpu --num_processes $2 --main_process_port 60008 train_rl.py --batch_size $4 --exp_name $1 --epochs $3