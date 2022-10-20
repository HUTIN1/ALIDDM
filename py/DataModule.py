from operator import getitem
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import int64, float32, tensor
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from monai.transforms import (
    ToTensor
)
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence


from random import choice
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from random import randint
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'Tools')
sys.path.append(mymodule_dir)
from ALIDDM_utils import FocusTeeth, MeanScale, DecomposeSurf, get_landmarks_position, RandomRotation, pos_landmard2texture, numberTooth2Landmark
from utils import(
    ReadSurf,
    ComputeNormals,
    GetColorArray
)

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
            surf, angle ,vectorrotation = RandomRotation(surf)
        mean, scale, surf = FocusTeeth(surf,self.surf_property,id_teeth)
        surf = ComputeNormals(surf) 

        verts = ToTensor(dtype=torch.float32)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int64)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        list_landmark = numberTooth2Landmark(id_teeth)
        pos_landmark = get_landmarks_position(dataset_dir = self.dataset_dir,df = self.df, idex = idx, mean_arr= mean, scale_factor = 1/scale ,lst_landmarks= list_landmark)
        color_normals = ToTensor(dtype=torch.float32)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
        color_landmark = pos_landmard2texture(verts,pos_landmark)

        texture_normal = TexturesVertex(verts_features=color_normals[None,:,:])
        texture_landmarks = TexturesVertex(verts_features=color_landmark[None,:,:])
        mesh_normal = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_normal).to(self.device)
        mesh_landmark = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_landmarks).to(self.device)


        faces_pid0 = faces[:,0:1]
        surf_point_data = surf.GetPointData().GetScalars(self.surf_property)

        surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
        surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

        surf_point_data_faces[surf_point_data_faces==-1] = 33 
        YF = surf_point_data_faces.to(self.device)
        YF = YF.to(torch.int64)

        return mesh_normal, mesh_landmark, YF




class PrintCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("on_train_epoch_end")
        print(self.epoch)
        return super().on_train_epoch_end(trainer, pl_module)