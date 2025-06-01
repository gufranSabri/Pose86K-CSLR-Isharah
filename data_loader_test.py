import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lips = sorted(set(lipsUpperOuter + lipsLowerOuter))  # Remove duplicates and sort
NUM_LIPS = 20  # after deduplication

class PoseDataset(Dataset):
    def __init__(self, dataset_name2, pkl_path, label_csv, split_type, target_enc_df, 
                 transform=None, augmentations=True, augmentations_prob=0.5, additional_joints=False):

        self.dataset_name = dataset_name2
        self.pkl_path = pkl_path
        self.split_type = split_type
        self.transform = transform
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.additional_joints = additional_joints

        with open(pkl_path, 'rb') as f:
            self.pose_dict = pickle.load(f)

        self.files = []
        self.labels = []

        self.all_data = pd.read_csv(label_csv, delimiter="|")

        for _, row in self.all_data.iterrows():
            sample_id = int(row["id"])
           
            enc_label = target_enc_df[target_enc_df["id"] == sample_id]["enc"]
            if  sample_id in self.pose_dict:
               
                self.files.append(sample_id)
                self.labels.append(enc_label.iloc[0])
            else:
                print("not in dict")

        print(f"Loaded {len(self.files)} samples for split: {split_type}")

    def __len__(self):
        return len(self.files)

    def readPose(self, sample_id):
        pose_data = self.pose_dict[sample_id]['keypoints']  # shape (T, J, 2)
        if pose_data is None or pose_data.shape[0] == 0:
            raise ValueError(f"Empty pose data for {sample_id}")

        T = pose_data.shape[0]
        aug = False

        if self.augmentations and np.random.rand() < self.augmentations_prob:
            aug = True
            angle = np.radians(np.random.uniform(-13, 13))
            pose_data = self.augment_time_warp(pose_data)
            pose_data = self.augment_frame_dropout(pose_data)
            
        right_hand = pose_data[:, 0:21, :]
        left_hand = pose_data[:, 21:42, :]
        lips = pose_data[:, 42:42+NUM_LIPS, :]
        body =  pose_data[:, 42+NUM_LIPS:, :]

        for ii in range(T):
            if aug:
                right_hand[ii] = self.augment_data(right_hand[ii], angle)
                left_hand[ii] = self.augment_data(left_hand[ii], angle)

            right_hand[ii] = self.normalize(right_hand[ii])
            left_hand[ii] = self.normalize(left_hand[ii])

            lips[ii] = self.normalize(lips[ii])
            body[ii] = self.normalize(body[ii])

        
        combined = np.concatenate([right_hand, left_hand], axis=1)
        if self.additional_joints:
            combined = np.concatenate([combined, lips, body], axis=1)
        return combined

    def normalize(self, pose):
        pose = pose - pose[0]
        pose = pose - np.min(pose, axis=0)

        max_vals = np.max(pose, axis=0)
        scale = max(max_vals)

        if scale == 0 or np.isnan(scale):
            return np.zeros_like(pose)

        pose = pose / scale
        pose = pose - np.mean(pose)
        max_abs = np.max(np.abs(pose))
        if max_abs == 0 or np.isnan(max_abs):
            return np.zeros_like(pose)

        return pose / max_abs * 0.5

    def augment_time_warp(self, pose_data, max_shift=2):
        T = pose_data.shape[0]
        new_data = np.zeros_like(pose_data)
        for i in range(T):
            shift = np.random.randint(-max_shift, max_shift + 1)
            new_idx = np.clip(i + shift, 0, T - 1)
            new_data[i] = pose_data[new_idx]
        return new_data

    def augment_frame_dropout(self, pose_data, drop_prob=0.1):
        T = pose_data.shape[0]
        mask = np.random.rand(T) > drop_prob
        return pose_data * mask[:, np.newaxis, np.newaxis]

    def augment_data(self, data, angle=None):
        if np.random.rand() < 0.5:
            data = np.array([self.rotate((0.5, 0.5), point, angle) for point in data])
        if np.random.rand() < 0.5:
            data = self.augment_jitter(data)
        if np.random.rand() < 0.5:
            data = self.augment_scale(data)
        if np.random.rand() < 0.5:
            data = self.augment_dropout(data)
        return data

    def augment_jitter(self, keypoints, std_dev=0.01):
        noise = np.random.normal(loc=0, scale=std_dev, size=keypoints.shape)
        return keypoints + noise

    def augment_dropout(self, keypoints, drop_prob=0.1):
        mask = np.random.rand(*keypoints.shape[:1]) > drop_prob
        keypoints *= mask[:, np.newaxis]
        return keypoints

    def augment_scale(self, keypoints, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(*scale_range)
        return keypoints * [scale, scale]

    def rotate(self, origin, point, angle):
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def pad_or_crop_sequence(self, sequence, min_len=32, max_len=1000):
        T, J, D = sequence.shape
        if T < min_len:
            pad_len = min_len - T
            pad = np.zeros((pad_len, J, D))
            sequence = np.concatenate((sequence, pad), axis=0)
        if sequence.shape[0] > max_len:
            sequence = sequence[:max_len]
        return sequence

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        pose = self.readPose(sample_id)
        pose = self.pad_or_crop_sequence(pose, min_len=32, max_len=1000)
        pose = torch.from_numpy(pose).float()
        if self.transform:
            pose = self.transform(pose)
        label = self.labels[idx]
        return sample_id, pose, torch.as_tensor(label)
