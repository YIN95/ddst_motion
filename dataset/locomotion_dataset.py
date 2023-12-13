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
import joblib as jl

class LOCODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        data_len: int = 150,
        force_reload: bool = False,
        data_sub: str = "all",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.data_sub = data_sub
        self.data_len = data_len
        self.force_reload = force_reload

        self.train = train
        self.name = "Train" if self.train else "Test"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        self.load_data()

    def load_data(self):
        data_path = self.data_path
        style_list = np.load(os.path.join(data_path, 'train_style_30fps.npz'))['clips']
        label_list = np.load(os.path.join(data_path, 'train_label_30fps.npz'))['clips']
        motion_list = np.load(os.path.join(data_path, 'train_output_30fps.npz'))['clips']
        ctrl_list = np.load(os.path.join(data_path, 'train_input_30fps.npz'))['clips']

        self.input_scaler = jl.load(os.path.join(data_path, 'input_scaler.sav'))
        self.output_scaler = jl.load(os.path.join(data_path, 'output_scaler.sav'))
        self.data_pipe = jl.load(os.path.join(data_path, 'data_pipe.sav'))

        if self.data_sub != 'all':
            sub_index = style_list.tolist().index(self.data_sub)
            sub_indices = np.where(label_list==sub_index)[0].tolist()
        else:
            sub_indices = np.where(label_list!=sub_index)[0].tolist()

        self.style_list = style_list
        self.label_list = torch.from_numpy(label_list[sub_indices])
        self.motion_list = torch.from_numpy(motion_list[sub_indices])
        self.ctrl_list = torch.from_numpy(ctrl_list[sub_indices])

        self.length = len(self.motion_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        motion = self.motion_list[idx]
        ctrl = self.ctrl_list[idx]
        label = self.label_list[idx]
        style = self.data_sub
        
        return motion, ctrl, label, style

