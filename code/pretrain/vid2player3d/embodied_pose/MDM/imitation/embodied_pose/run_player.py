# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# import multiprocessing as mp
# try:
#     mp.set_start_method("spawn")
# except:
#     pass
import sys
import os
# sys.path.append('./poselib')

sys.path.append("./model/imitation/")

from embodied_pose.utils.config import set_np_formatting, get_args, parse_sim_params, load_cfg
from embodied_pose.utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from embodied_pose.models.im_models import ImitatorModel
from embodied_pose.models.im_network_builder import ImitatorBuilder
from embodied_pose.agents.im_agent import ImitatorAgent
from embodied_pose.players.im_player import ImitatorPlayer
from embodied_pose.players.im_player_diff import diff_player
# import torch.multiprocessing as mp
# try:
#     mp.set_start_method("spawn")
# except:
#     pass
args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if not (args.tmp or args.no_log):
            if self.consecutive_successes.current_size > 0:
                mean_con_successes = self.consecutive_successes.get_mean()
                self.algo.log_dict.update({'successes/consecutive_successes/mean': mean_con_successes})
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    
    runner.algo_factory.register_builder('pose_im_rnn', lambda **kwargs : ImitatorAgent(**kwargs))        # training agent
    runner.player_factory.register_builder('pose_im_rnn', lambda **kwargs : diff_player(**kwargs))     # testing agent
    runner.model_builder.model_factory.register_builder('pose_im', lambda network, **kwargs : ImitatorModel(network))    # network wrapper
    runner.model_builder.network_factory.register_builder('pose_im_rnn', lambda **kwargs : ImitatorBuilder())     # actuall network definition class
    
    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train = load_cfg(args)

    vargs = vars(args)
    if cfg["headless"]:
        import warnings
        from loguru import logger
        logger.remove(handler_id=None)
        warnings.filterwarnings("ignore")
        sys.stdout = open(os.devnull,'w')
    
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return

def process_args(args):
    if args.resume_id is not None:
        args.resume = True
    args.sim_device = args.rl_device
    if args.use_gpu_pipeline:
        args.compute_device_id = int(args.rl_device[-1])
        args.graphics_device_id = int(args.rl_device[-1])
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args

def get_player(device=None):
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args(device=device)
    cfg, cfg_train = load_cfg(args)

    vargs = vars(args)
    if cfg["headless"]:
        import warnings
        from loguru import logger
        # logger.remove(handler_id=None)
        warnings.filterwarnings("ignore")
        # sys.stdout = open(os.devnull,'w')
    
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    if 'checkpoint' in vargs and vargs['checkpoint'] is not None:
        if len(vargs['checkpoint']) > 0:
            runner.load_path = vargs['checkpoint']
    if device!=None:
        runner.config["device_name"] = str(device)
    player = runner.create_player()
    player.restore(runner.load_path)
    return player

if __name__ == '__main__':
    main()
