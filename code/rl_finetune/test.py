import isaacgym
import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE,PopDance
from data.audio_extraction.baseline_features import extract as baseline_extract
# from data.audio_extraction.jukebox_features import extract as juke_extract
from dataset.dance_dataset import AISTPPDataset
import pickle
try:
    from model.imitation.embodied_pose.run_player import get_agent,get_player
except:
    pass

from accelerate.utils import set_seed

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    # feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    # print(sample_size)
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        # print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        if PopDance==True:
            num=2
        else:
            num=4
        for _ in range(num):
            for dir in tqdm(dir_list):
                file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
                juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
                assert len(file_list) == len(juke_file_list)
                # random chunk after sanity check
                # rand_idx = random.randint(0, len(file_list) - sample_size)
                rand_idx=0
                sample_size=len(file_list)
                file_list = file_list[rand_idx : rand_idx + sample_size]
                juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
                cond_list = [np.load(x) for x in juke_file_list]
                all_filenames.append(file_list)
                all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        if PopDance==True:
            num=2
        else:
            num=4
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = 0#random.randint(0, len(file_list) - sample_size)
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])
        all_cond = all_cond*num
        all_filenames = all_filenames*num

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    # print("Generating dances")
    # print(len(all_cond))
    # print(all_filenames[0])
    # all_cond[0] = all_cond[0].view(186,1,150,-1).repeat(1,5,1,1).reshape(930,150,-1)
    # all_filenames[0] = [i for i in all_filenames[0] for j in range(5)]
    # print(all_filenames[0])
    # print(all_cond[0].shape)
    # all_cond = all_cond[0]
    player = None#get_player(torch.device("cuda:0"))
    # all_cond = torch.concat(all_cond,dim=0)
    # all_cond = [all_cond]*4
    # all_file = []
    # for file in all_filenames:
    #     all_file.extend(file)
    # all_filenames = [all_file]*4
    test_tensor_dataset_path = os.path.join(
            "./data/dataset_backups", f"test_tensor_dataset.pkl"
        )
    test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
    motion_file = glob.glob("./data/test/motions/*.pkl")
    k = [i for i in range(len(all_cond))]
    # import random
    # random.shuffle(k)
    for i in tqdm(k):
        # i=k[8]
        # a = torch.randint(0,10000,(1,))
        # a = int(a.item())
        # set_seed(a,True)
        data_tuple = None, all_cond[i], all_filenames[i]
        # import ipdb;ipdb.set_trace()
        if PopDance==True:
            motion=None
        else:
            motion = None
            # for motion in motion_file:
            #     name = motion.split("/")[-1].split("_")[-2]
            #     if name in all_filenames[i][0]:
            #         motion_file.remove(motion)
            #         motion = np.load(motion,allow_pickle=True)
            #         break
            # l = 30
            # try:
            #     q = motion["smpl_poses"][::2][:l].reshape(1,l,72)
            #     trans = (motion["smpl_trans"][::2]/motion["smpl_scaling"])[:l].reshape(1,l,-1)
            # except:
            #     q = motion["q"][::2][:l].reshape(1,l,72)
            #     trans = (motion["pos"][::2]/motion["scale"])[:l].reshape(1,l,-1)
            # motion = test_dataset.process_dataset(trans,q)
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render,player=player,i=i,motion=motion
        )
        # if i==1:
        #     break
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
