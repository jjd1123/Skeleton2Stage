from args import parse_train_opt
from model.imitation.embodied_pose.run_player import get_player
from EDGE import EDGE
import os;os.environ["WANDB_MODE"]="offline"

def train_rl(opt):
    model = EDGE(opt.feature_type,opt.checkpoint)
    player = get_player(device = model.accelerator.device)
    model.train_rl(opt,player)


if __name__ == "__main__":
    opt = parse_train_opt()
    train_rl(opt)