import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton
import librosa

class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        
        self.data = {
            "pose": pose_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length
    
    def generate_idx(self):
        self.index = torch.arange(len(self.data['filenames']))
        # self.length = 512
        # self.index = torch.randperm(len(self.data['filenames']))[:self.length].tolist()
        # for i in self.data.keys():
        #     if type(self.data[i]) == list:
        #         tmp_list = []
        #         for j in self.index:
        #             tmp_list.append(self.data[i][j])
        #         self.data[i] = tmp_list
        #     else:
        #         self.data[i] = self.data[i][self.index]
        # self.index = torch.arange(self.length)
    
    def get_music_beat_fromwav(self, fpath, length):
        FPS = 60
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH
        # EPS = 1e-6
        # print(length)
        data, _ = librosa.load(fpath, sr=SR)
        # print("loaded music data shape", data.shape)
        _, audio_percussive = librosa.effects.hpss(y=data)
        envelope = librosa.onset.onset_strength(y=audio_percussive, aggregate=np.median,sr=SR)  # (seq_len,)
        # peak_idxs = librosa.onset.onset_detect(
        #     onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
        # )
        # start_bpm = librosa.beat.tempo(y=audio_percussive)[0]
        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=SR,
            hop_length=HOP_LENGTH,
            # start_bpm=start_bpm,
            tightness=100,
        )
        # print(envelope.shape)
        beats_one_hot = np.zeros(len(envelope))
        for idx in beat_idxs:
            beats_one_hot[idx] = 1
        beats_one_hot = beats_one_hot[:length]
        beats = beats_one_hot.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_idxs = beat_axis[beats]
        return beat_idxs

    def collate_fn(self,batch):
        pose,feature,filename_,wav,index,beats = zip(*batch)
        return (
            default_collate(pose),
            default_collate(feature),
            default_collate(filename_),
            default_collate(wav),
            default_collate(index),
            list(beats)   # 或自定义处理
        )

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        filename_beats = filename_.replace("jukebox_feats","beats_sliced")
        if os.path.exists(filename_beats):
            beats = np.load(filename_beats)
        else:
            beats = self.get_music_beat_fromwav(self.data["wavs"][idx],self.data["pose"][idx].shape[0]*2)
            np.save(filename_beats,beats)
        feature = torch.from_numpy(np.load(filename_))
        return self.data["pose"][idx], feature, filename_, self.data["wavs"][idx],self.index[idx]#, beats

    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- jukebox_features
        #   |    |- motions
        #   |    |- wavs

        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        assert len(motions) == len(features)
        for motion, feature, wav in zip(motions, features, wavs):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert m_name == f_name == w_name, str((motion, feature, wav))
            # load motion
            data = pickle.load(open(motion, "rb"))
            pos = data["pos"]
            q = data["q"]
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(all_pos.shape)
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "baseline",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4",
                "mKR2",
                "mBR0",
                "mLO2",
                "mJB5",
                "mWA0",
                "mJS3",
                "mMH3",
                "mHO5",
                "mPO1",
            ]
        )
        self.train = train

        # if not aist, then set train to true to ignore test split logic
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True

        self.data = self.load_music()  # Call this last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        # now we have a batch of filenames
        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        # get the features
        features = np.array([np.load(x) for x in filenames])

        return torch.Tensor(features), seq_slice

    def load_music(self):
        # open data path
        split_data_path = os.path.join(self.data_path)
        music_path = os.path.join(
            split_data_path,
            f"{self.data_name}_baseline_feats"
            if self.feature_type == "baseline"
            else f"{self.data_name}_juke_feats/juke_66",
        )
        self.music_path = music_path
        # get the music filenames strided, with each subsequent item 5 slices (2.5 seconds) apart
        all_names = []

        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
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

        for features in glob.glob(os.path.join(music_path, "*.npy")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict


class AISTPPDataset_RL(Dataset):
    def __init__(
        self,
        rewards,
        timesteps,
        latents,
        log_probs,
        conds,
        alives,
        x_starts,
        data_len: int=-1,
    ):
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps
        self.samples = {
            "rewards":rewards.cpu(),
            "latents":latents[:,:-1].cpu(),
            "next_latents":latents[:,1:].cpu(),
            "log_probs":log_probs.cpu(),
            "alives":alives.cpu()
        } 
        self.file_name = conds.cpu()
        self.time_step = timesteps 
        if data_len<0:
            self.data_len = len(self.file_name)
        else:
            self.data_len = data_len
        self.x_start = x_starts.cpu()

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        filename_ = self.file_name[idx]
        feature = filename_#torch.from_numpy(np.load(filename_))
        reward = self.samples["rewards"][idx]
        latent = self.samples["latents"][idx]
        next_latent = self.samples["next_latents"][idx]
        alive = self.samples["alives"][idx]
        logprob = self.samples["log_probs"][idx]
        return reward,latent,next_latent,logprob,feature,self.time_step,alive,self.x_start[idx]
    
    def process(self, samples):
        out = []
        for sample in samples:
            for i in range(sample["log_prob"].shape[0]):
                out.append({"time_step":sample["time_step"],#.cpu(),
                            "latent":sample["latent"][i,:-1],#.cpu(),
                            "next_latent":sample["latent"][i,1:],#.cpu(),
                            "log_prob":sample["log_prob"][i],#.cpu(),
                            "cond":sample["cond"][i],#.cpu(),
                            "reward":sample["reward"][i],})#.cpu()})
        return out
