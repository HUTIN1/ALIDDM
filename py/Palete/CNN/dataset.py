from operator import getitem
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import int64, float32, tensor
from torch.nn.utils.rnn import pad_sequence
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from monai.transforms import (
    ToTensor
)
import os
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
from icp import PrePreAso


from utils import(
    ReadSurf,
    ComputeNormals,
    GetColorArray,
    RandomRotation,get_landmarks_position, pos_landmard2texture, pos_landmard2seg, TransformSurf
)


class TeethDataModuleLm(pl.LightningDataModule):
    def __init__(self, df_train, df_val,df_test,num_workers = 4,surf_property =None ,mount_point='./',batch_size=1, drop_last=False,
    train_transform=None,val_transform=None,test_transform=None,landmark='') -> None:
        super().__init__()
        self.df_train = df_train
        self.df_val= df_val
        self.df_test = df_test
        self.batch_size = batch_size
        # df_test = df.loc[df['for'] == "test"]
        # self.df_test = df_test.loc[df_val['jaw'] == jaw]

        self.mount_point = mount_point
        self.drop_last = drop_last

        self.num_workers = num_workers

        self.surf_property = surf_property

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.landmark=landmark
        self.test_transform = test_transform


    def setup(self,stage =None):
        self.train_ds = TeethDatasetLm(mount_point = self.mount_point, df = self.df_train,surf_property = self.surf_property,transform=self.train_transform,landmark=self.landmark)
        self.val_ds = TeethDatasetLm(mount_point = self.mount_point, df = self.df_val,surf_property = self.surf_property, transform=self.val_transform,landmark=self.landmark)
        self.test_ds = TeethDatasetLm(mount_point = self.mount_point, df = self.df_test,surf_property = self.surf_property,transform=self.test_transform,landmark=self.landmark)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.pad_verts_faces)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.ad_verts_faces)

    def pad_verts_faces(self, batch):
        V = [V for V, F, CN, YF  in batch]
        F = [F for V, F, CN, YF in batch]
        CN = [CN for V, F, CN, YF in batch]
        LF = [LF for V, F, CN ,LF in batch]


        V = pad_sequence(V,batch_first=True, padding_value=0.0)
        F = pad_sequence(F,batch_first=True,padding_value=-1)
        CN = pad_sequence(CN,batch_first=True,padding_value=0.0)
        LF = torch.cat(LF)

        return V, F, CN, LF








class TeethDatasetLm(Dataset):
    def __init__(self,df,surf_property ,mount_point='',transform = False,landmark='',test=False,prediction=False):
        self.df = df
        self.mount_point = mount_point

        self.surf_property = surf_property

        self.transform = transform
        self.landmark = landmark
        self.test = test
        self.prediction= prediction


    def __len__(self):
            
        return len(self.df)

    def __getitem__(self, index) :
        if isinstance(self.df,list):
            surf = ReadSurf(self.df[index])

        else :

            surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[index]["surf"]))

        surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['4','9','10','15'])

        if self.transform:
            surf, matrix_transform = self.transform(surf)

        matrix = np.matmul(matrix_transform,matrix)

        scale = 3
        scale_matrix = np.array([[scale,0,0,0],
                                             [0, scale,0 ,0],
                                             [0, 0, scale ,0],
                                             [0, 0, 0, 1]])
        surf = TransformSurf(surf, scale_matrix)

        matrix = np.matmul(scale_matrix,matrix)

        
        

        surf = ComputeNormals(surf) 
     

        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
        CN = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 




        if not self.prediction :


            pos_landmark = get_landmarks_position(os.path.join(self.mount_point,self.df.iloc[index]["landmark"]),self.landmark,matrix)


            LF = pos_landmard2seg(V,pos_landmark)
            faces_pid0 = F[:,0:1]       
            LF = torch.take(LF, faces_pid0)            
            LF = LF.to(torch.int64)

            if self.test:
                CL = pos_landmard2texture(V,pos_landmark)
                # CL = pos_landmard2seg(V,pos_landmark)
                return V, F, CN, CL
            
            return V, F, CN, LF
            
            
        else :

            return V, F, CN 
        



        


    def getSurf(self,idx):
        if isinstance(self.df,list):
            surf = ReadSurf(self.df[idx])
        else :
            surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[idx]["surf"]))
        return surf
    
    def getName(self,idx):
        if isinstance(self.df,list):
            path = self.df[idx]
        else :
            path = os.path.join(self.mount_point,self.df.iloc[idx]["surf"])
        name = os.path.basename(path)
        name , _ = os.path.splitext(name)

        return name



