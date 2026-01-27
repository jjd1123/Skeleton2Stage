# cd /workspace/jja3wx/jidong/EDGE
# # source /workspace/xak1wx/.cu117rc
# source ../environment.sh
# git config --global --add safe.directory /workspace/jja3wx/jidong/EDGE
# accelerate launch train_rl.py --batch_size 32 --exp_name $1
accelerate launch --multi_gpu --num_processes $2 --main_process_port 60008 train_rl.py --batch_size 32 --exp_name $1 --epochs 3000