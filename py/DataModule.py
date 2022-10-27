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
from ALIDDM_utils import FocusTeeth, MeanScale, DecomposeSurf, get_landmarks_position, pos_landmard2texture, numberTooth2Landmark
from utils import(
    ReadSurf,
    ComputeNormals,
    GetColorArray,
    RandomRotation
)

class TeethDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val,df_test,num_workers = 4,surf_property =None ,mount_point='./',batch_size=1, drop_last=False) -> None:
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

    def setup(self,stage =None):
        self.train_ds = DatasetValidation(mount_point = self.mount_point, df = self.df_train,surf_property = self.surf_property,random=True)
        self.val_ds = DatasetValidation(mount_point = self.mount_point, df = self.df_val,surf_property = self.surf_property)
        self.test_ds = DatasetValidation(mount_point = self.mount_point, df = self.df_test,surf_property = self.surf_property)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.pad_verts_faces)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.ad_verts_faces)

    def pad_verts_faces(self, batch):
        V = [V for V, F, CN, CL, YF  in batch]
        F = [F for V, F, CN, CL, YF in batch]
        CN = [CN for V, F, CN, CL, YF in batch]
        CL = [CL for V, F, CN, CL ,YF in batch]
        YF = [YF for V, F, CN, CL ,YF in batch]

        V = pad_sequence(V,batch_first=True, padding_value=0.0)
        F = pad_sequence(F,batch_first=True,padding_value=-1)
        CN = pad_sequence(CN,batch_first=True,padding_value=0.0)
        CL = pad_sequence(CL,batch_first=True,padding_value=0.0)
        YF = torch.cat(YF)
        return V, F, CN, CL, YF









class DatasetValidation(Dataset):
    def __init__(self,df,surf_property ,mount_point='',random=False):
        self.df = df
        self.mount_point = mount_point

        self.surf_property = surf_property

        self.random = random
        self.number_teeth = 14


    def __len__(self):
        return len(self.df)*self.number_teeth

    def __getitem__(self, index) :


        if self.random:
            index = randint(0, len(self)-1)


        idx=index//self.number_teeth
        surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[idx]["surf"]))


        CLF = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)
        unique_ids = torch.unique(CLF).cpu().tolist()
        if 33 in unique_ids :
            unique_ids.remove(33)

        index_teeth = len(self.df)%len(unique_ids)
        id_teeth = unique_ids[index_teeth]
         

        
        

        vectorrotation=None
        angle=False
        if self.random:
            surf, angle ,vectorrotation = RandomRotation(surf)
        mean, scale, surf = FocusTeeth(surf,self.surf_property,id_teeth)
        surf = ComputeNormals(surf) 

        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
        CN = ToTensor(dtype=torch.float32)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
        CLF = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)

        # texture_normal = TexturesVertex(verts_features=color_normals[None,:,:])
        # texture_landmarks = TexturesVertex(verts_features=color_landmark)
        # mesh_normal = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_normal)
        # mesh_landmark = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_landmarks)


        faces_pid0 = F[:,0:1]
        surf_point_data = surf.GetPointData().GetScalars(self.surf_property)

        surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
        surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

        surf_point_data_faces[surf_point_data_faces==-1] = 33 
        YF = surf_point_data_faces
        YF = YF.to(torch.int64)

        return V, F, CN, CLF, YF


# class DatasetValidation(Dataset):
#     def __init__(self,df,surf_property ,mount_point='',random=False):
#         self.df = df
#         self.mount_point = mount_point

#         self.surf_property = surf_property

#         self.random = random
#         self.number_teeth = 14


#     def __len__(self):
#         return len(self.df)*self.number_teeth

#     def __getitem__(self, index) :


#         if self.random:
#             index = randint(0, len(self)-1)


#         idx=index//self.number_teeth

#         unique_ids=[]
#         for i in range(2,32):
#             unique_ids.append(int(self.df.iloc[idx][str(i)])*i)
        
#         while 0 in unique_ids:
#             unique_ids.remove(0)

#         index_teeth = len(self.df)%len(unique_ids)
#         id_teeth = unique_ids[index_teeth]
#         surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[idx]["surf"])) 

        
        

#         vectorrotation=None
#         angle=False
#         if self.random:
#             surf, angle ,vectorrotation = RandomRotation(surf)
#         mean, scale, surf = FocusTeeth(surf,self.surf_property,id_teeth)
#         surf = ComputeNormals(surf) 

#         V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
#         F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
#         list_landmark = numberTooth2Landmark(id_teeth)
#         pos_landmark = get_landmarks_position(mount_point = self.mount_point,df = self.df, idx = idx, mean_arr= mean, scale_factor = 1/scale ,lst_landmarks= list_landmark, angle= angle,vector= vectorrotation)
#         CN = ToTensor(dtype=torch.float32)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
#         CL = pos_landmard2texture(V,pos_landmark)
#         CLF = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)

#         # texture_normal = TexturesVertex(verts_features=color_normals[None,:,:])
#         # texture_landmarks = TexturesVertex(verts_features=color_landmark)
#         # mesh_normal = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_normal)
#         # mesh_landmark = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_landmarks)


#         faces_pid0 = F[:,0:1]
#         surf_point_data = surf.GetPointData().GetScalars(self.surf_property)

#         surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
#         surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

#         surf_point_data_faces[surf_point_data_faces==-1] = 33 
#         YF = surf_point_data_faces
#         YF = YF.to(torch.int64)

#         return V, F, CN, CLF, YF

