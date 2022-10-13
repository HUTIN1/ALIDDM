from operator import getitem
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torch import int64, float32, tensor
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from ALIDDM.py.Tools.ALIDDM_utils import FocusTeeth, MeanScale, DecomposeSurf
from ALIDDM.py.Tools.utils import(
    ReadSurf,
    ScaleSurf,
    RandomRotation,
    ComputeNormals,
    GetColorArray,
    GetTransform
)
from monai.transforms import (
    ToTensor
)
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

from vtk import vtkMatrix4x4, vtkMatrix3x3, vtkPoints

import ALIDDM.py.Tools.utils as utils
from random import choice
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from random import randint



class TeethDataModule(pl.LightningDataModule):
    def __init__(self,df,dataset_dr,batch_size,device,jaw, num_workers = 4,
     train_transform = None, val_transform = None,unique_ids=[],surf_property =None,lst_landmarks = []) -> None:
        super().__init__()
        df = pd.read_csv(df)
        df_train = df.loc[df['for'] == "train"]
        self.df_train = df_train.loc[df_train['jaw'] == jaw]
        df_val = df.loc[df['for'] == "val"]
        self.df_val = df_val.loc[df_val['jaw'] == jaw]
        # df_test = df.loc[df['for'] == "test"]
        # self.df_test = df_test.loc[df_val['jaw'] == jaw]
        self.dataset_dr = dataset_dr
        self.device = device

        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.num_workers = num_workers
        self.unique_ids = unique_ids
        self.surf_property = surf_property
        self.lst_landmarks = lst_landmarks

    def setup(self,stage =None):
        self.train_ds = DatasetValidation(self.df_val,surf_property = self.surf_property,unique_ids=self.unique_ids,dataset_dir=self.dataset_dr,lst_landmarks=self.lst_landmarks,random=True)
        self.val_ds = DatasetValidation(self.df_val,surf_property = self.surf_property,unique_ids=self.unique_ids,dataset_dir=self.dataset_dr,lst_landmarks=self.lst_landmarks)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.pad_verts_faces)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.ad_verts_faces)

    def pad_verts_faces(self, batch):
        surf = [s for s, v, f, ri, cn, lp, sc, ma  in batch]
        verts = [v for s, v, f, ri, cn, lp, sc, ma  in batch]
        faces = [f for s, v, f, ri, cn, lp, sc, ma  in batch]
        region_id = [ri for s, v, f, ri, cn, lp, sc, ma  in batch]
        color_normals = [cn for s, v, f, ri, cn, lp, sc, ma, in batch]
        landmark_position = [lp for s, v, f, ri, cn, lp, sc, ma in batch]
        scale_factor = [sc for s, v, f, ri, cn, lp , sc, ma  in batch]
        mean_arr = [ma for s, v, f, ri, cn,lp, sc, ma   in batch]

        return surf, pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1),region_id, pad_sequence(color_normals, batch_first=True, padding_value=0.), landmark_position, mean_arr, scale_factor









class DatasetValidation(Dataset):
    def __init__(self,df,surf_property,unique_ids =None ,dataset_dir='',lst_landmarks = [],random=False):
        self.df = df
        self.dataset_dir = dataset_dir

        self.surf_property = surf_property
        self.unique_ids = unique_ids
        self.lst_landmarks = lst_landmarks
        self.random = random


    def __len__(self):
        return len(self.df)*len(self.unique_ids)

    def __getitem__(self, index) :
        if self.random:
            index = randint(0, len(self)-1)
        idx=index//len(self.unique_ids)
        index_teeth = len(self.df)%len(self.unique_ids)
        id_teeth = self.unique_ids[index_teeth]
        surf = ReadSurf(os.path.join(self.dataset_dir,self.df.iloc[idx]["surf"])) 
        if self.random:
            surf = utils.RandomRotation(surf)
        mean, scale, surf = FocusTeeth(surf,self.surf_property,id_teeth)
        surf = ComputeNormals(surf) 

        return DecomposeSurf(surf,self.surf_property,idx,mean,scale,self.lst_landmarks)




class PrintCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("on_train_epoch_end")
        print(self.epoch)
        return super().on_train_epoch_end(trainer, pl_module)