import numpy as np
import pickle 
# from features.kinetic import extract_kinetic_features
# from features.manual_new import extract_manual_features
from scipy import linalg
import json, torch, sys
# kinetic, manual
import os, librosa
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
sys.path.append(os.getcwd())
from vis import SMPLSkeleton
import argparse
from tqdm import tqdm


def get_mb(key, length=None):
    music_root = "../../Bailando/data/aistpp_test_full_wav"
    # music_root = "/workspace/jja3wx/popdg_dataprocess/POPDG-main/data/test/wavs"
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
    
        return beat_axis
    
def get_music_beat_fromwav(fpath, length):
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


def get_music_beat_from_musicfea35(fpath, length):


    data = np.load(fpath)[:length]
    beat_idxs = data[-1]

    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]
  
    return beat_idxs



def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    # print(len(music_beats))
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def calc_ba_score(motionroot, musicroot):
    # gt_list = []
    ba_scores = []
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211",  "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]

    for pkl in tqdm(os.listdir(motionroot)):
        # print(pkl)
        if os.path.isdir(os.path.join(motionroot, pkl)):
            continue
        if pkl[-3:] == 'pkl':
            data = pickle.load(open(os.path.join(motionroot, pkl), "rb"))
            # print(data.keys())
            try:
                model_q = torch.from_numpy(data['smpl_poses']).reshape(-1,24,3).unsqueeze(0).to("cuda")   
                model_x = torch.from_numpy(data['smpl_trans']).squeeze().unsqueeze(0).to("cuda")
            except:
                model_q = torch.from_numpy(data['q']).reshape(-1,24,3).unsqueeze(0).to("cuda")   
                model_x = torch.from_numpy(data['pos']/data['scale'][0]).unsqueeze(0).to("cuda")
                # print(data['scale'])
            # print("model_q", model_q.shape)
            # print("model_x", model_x.shape)
            # model_q156 = torch.cat([model_q, torch.zeros([model_q.shape[0], 90])], dim=-1)
            with torch.no_grad():
                joint3d = smplx_model.forward(model_q, model_x)[0][:,:24,:]
                # print("joint3d", joint3d.shape)
        elif pkl[-3:]=="npy":
            joint3d = np.load(os.path.join(motionroot, pkl), allow_pickle=True).item()['pred_position'][:, :]
        else:
            continue
        # assert len(joint3d.shape) == 3      # T, J, 3
        try:
            joint3d = joint3d.reshape(joint3d.shape[0], 24*3).detach().cpu().numpy()
        except:
            joint3d = joint3d.reshape(joint3d.shape[0], 24*3)
        # joint3d[...,[0,1,2]]=joint3d[...,[0,2,1]]
        joint3d_60 = np.zeros((joint3d.shape[0]*2,joint3d.shape[1]))
        joint3d_60[:-1:2] = joint3d
        joint3d_60[1::2] = joint3d
        joint3d = joint3d_60
        joint3d = joint3d[:1200]
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 24)) 
        joint3d = joint3d.reshape(-1, 24, 3)

        # joint3d = np.load(os.path.join(motionroot, pkl), allow_pickle=True).item()['pred_position'][:, :]
        dance_beats, length = calc_db(joint3d, pkl)    
        pkl1 = pkl
        pkl = pkl.split('_')[-1].split('.')[0]
        # pkl = "_".join(pkl)
        # music_beats = get_mb(pkl.split('.')[0] + '.json', length)
        music_beats = get_music_beat_fromwav(os.path.join(musicroot, pkl.split('.')[0] + '.wav'), joint3d.shape[0])
        # music_beats_bailando = get_mb(pkl1.split('.')[0] + '.json',length)
        # print(music_beats-music_beats_bailando)
        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    music_root = "./music/test"
    # music_root = "./music/pop_music"
    smplx_model = SMPLSkeleton("cuda")
    pred_root = opt.motion_path

    print('Calculating pred metrics')
    print(pred_root)
    bascore = calc_ba_score(pred_root, music_root)
    with open(f"{pred_root}/metrics.txt","a") as f:
        f.write(f"BA score: {bascore}\n")
  