from operator import getitem
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import int64, float32, tensor
from torch.nn.utils.rnn import pad_sequence
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
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
    RandomRotation,get_landmarks_position, pos_landmard2texture, pos_landmard2seg, TransformSurf,TransformRotationMatrix,RandomRotationZ,
    pos_landmard2seg_special,pos_landmard2texture_special,pos_Tshape_texture)
from utils2 import rectangle_patch_texture


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
        # self.train_ds = TeethDatasetLm(mount_point = self.mount_point, df = self.df_train,surf_property = self.surf_property,transform=self.train_transform,landmark=self.landmark)
        # self.val_ds = TeethDatasetLm(mount_point = self.mount_point, df = self.df_val,surf_property = self.surf_property, transform=self.val_transform,landmark=self.landmark)
        # self.test_ds = TeethDatasetLm(mount_point = self.mount_point, df = self.df_test,surf_property = self.surf_property,transform=self.test_transform,landmark=self.landmark)
        # self.train_ds  = TeethDatasetLmCoss(mount_point = self.mount_point, df = self.df_train,surf_property = self.surf_property,transform=self.train_transform,landmark=self.landmark,random_rotation=True)
        # self.val_ds = TeethDatasetLmCoss(mount_point = self.mount_point, df = self.df_val,surf_property = self.surf_property, transform=self.val_transform,landmark=self.landmark)
        # self.test_ds = TeethDatasetLmCoss(mount_point = self.mount_point, df = self.df_test,surf_property = self.surf_property,transform=self.test_transform,landmark=self.landmark)
        self.train_ds = TeethDatasetPatch(mount_point = self.mount_point, df = self.df_train,surf_property = self.surf_property,transform=self.train_transform,landmark=self.landmark,random_rotation=True)
        self.val_ds = TeethDatasetPatch(mount_point = self.mount_point, df = self.df_val,surf_property = self.surf_property, transform=self.val_transform,landmark=self.landmark)
        self.test_ds = TeethDatasetPatch(mount_point = self.mount_point, df = self.df_test,surf_property = self.surf_property,transform=self.test_transform,landmark=self.landmark)
        
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


    # def pad_verts_faces(self, batch):
    #     V = [V for V, F, CN, vector, distance  in batch]
    #     F = [F for V, F, CN, vector, distance in batch]
    #     CN = [CN for V, F, CN, vector, distance in batch]
    #     distance = [distance for V, F, CN, vector, distance in batch]
    #     vector = [vector for V, F, CN, vector, distance in batch]




    #     V = pad_sequence(V,batch_first=True, padding_value=0.0)
    #     F = pad_sequence(F,batch_first=True,padding_value=-1)
    #     CN = pad_sequence(CN,batch_first=True,padding_value=0.0)
    #     vector = torch.cat(vector)
    #     distance = torch.cat(distance)


    #     return V, F, CN, vector, distance





class TeethDatasetLm(Dataset):
    def __init__(self,df,surf_property ,mount_point='',transform = False,landmark=[],test=False,prediction=False,random_rotation= False):
        self.df = df
        self.mount_point = mount_point

        self.surf_property = surf_property

        self.transform = transform
        self.landmark = landmark
        self.test = test
        self.prediction= prediction
        self.random_rotation = random_rotation


    def __len__(self):
            
        return len(self.df)

    def __getitem__(self, index) :
        if isinstance(self.df,list):
            surf = ReadSurf(self.df[index])
            # print(f'path scan {self.df[index]}')
            # surf = ReadSurf('/home/luciacev/Desktop/Data/IOSReg/files_not_organize/DeniseTest_json/P2_Upper.vtk')

        else :

            surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[index]["surf"]))

        surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['5','9','10','12'])

  

        if self.random_rotation:
            surf , angle, vector = RandomRotationZ(surf)
            angle = angle*np.pi / 180
            matrix_random_rotation = TransformRotationMatrix(vector, angle)
            matrix = np.matmul(matrix_random_rotation,matrix)


        if self.transform:
            surf, matrix_transform = self.transform(surf)

            matrix = np.matmul(matrix_transform,matrix)

        # scale = 3
        # scale_matrix = np.array([[scale,0,0,0],
        #                                      [0, scale,0 ,0],
        #                                      [0, 0, scale ,0],
        #                                      [0, 0, 0, 1]])
        # surf = TransformSurf(surf, scale_matrix)

        # matrix = np.matmul(scale_matrix,matrix)

        
        

        surf = ComputeNormals(surf) 
     

        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
        CN = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 




        if not self.prediction :


            pos_landmark = get_landmarks_position(os.path.join(self.mount_point,self.df.iloc[index]["landmark"]),self.landmark,matrix)


            print(f'pos landmark {pos_landmark}')


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

            return V, F, CN , torch.tensor(matrix)
        



        


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



class TeethDatasetLmCoss(Dataset):
    def __init__(self,df,surf_property ,mount_point='',transform = False,landmark='',test=False,prediction=False,random_rotation=False):
        self.df = df
        self.mount_point = mount_point

        self.surf_property = surf_property

        self.transform = transform
        self.landmark = landmark
        self.test = test
        self.prediction= prediction
        self.random_rotation = random_rotation


    def __len__(self):
            
        return len(self.df)

    def __getitem__(self, index) :
        if isinstance(self.df,list):
            surf = ReadSurf(self.df[index])

        else :

            surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[index]["surf"]))


        surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['5','9','10','14'])

        if self.random_rotation:
            surf , angle, vector = RandomRotationZ(surf)
            angle = angle*np.pi / 180
            matrix_random_rotation = TransformRotationMatrix(vector, angle)
            matrix = np.matmul(matrix_random_rotation,matrix)


        if self.transform:
            surf, matrix_transform = self.transform(surf)

        matrix = np.matmul(matrix_transform,matrix)
        # matrix= np.identity(4)

        # scale = 3
        # scale_matrix = np.array([[scale,0,0,0],
        #                                      [0, scale,0 ,0],
        #                                      [0, 0, scale ,0],
        #                                      [0, 0, 0, 1]])
        # surf = TransformSurf(surf, scale_matrix)

        # matrix = np.matmul(scale_matrix,matrix)

        
        

        surf = ComputeNormals(surf) 
     

        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
        CN = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 




        if not self.prediction :


            pos_landmark = get_landmarks_position(os.path.join(self.mount_point,self.df.iloc[index]["landmark"]),self.landmark,matrix)
            distance = np.linalg.norm(pos_landmark)
            vector = pos_landmark / distance


            # LF = pos_landmard2seg(V,pos_landmark)
            # faces_pid0 = F[:,0:1]       
            # LF = torch.take(LF, faces_pid0)            
            # LF = LF.to(torch.int64)

            if self.test:
                CL = pos_landmard2texture(V,pos_landmark)
                # CL = pos_landmard2seg(V,pos_landmark)
                return V, F, CN, CL, torch.tensor(vector),torch.tensor(distance).unsqueeze(0).unsqueeze(0)
            
            return V, F, CN, torch.tensor(vector), torch.tensor(distance).unsqueeze(0).unsqueeze(0)#, matrix
            
            
        else :

            return V, F, CN , torch.tensor(matrix)
        



        


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



class RegistrationDataset(Dataset):
    def __init__(self,df,surf_property ,mount_point='',transform = False,landmark=[],test=False,prediction=False,random_rotation= False):
        self.df = df
        self.mount_point = mount_point

        self.surf_property = surf_property

        self.transform = transform
        self.landmark = landmark
        self.test = test
        self.prediction= prediction
        self.random_rotation = random_rotation

    def __len__(self):
            
        return len(self.df)
    
    def __getitem__(self, index):
        surf_T1 = ReadSurf(self.df.iloc[index]['T1'])



        surf_T1, matrix = PrePreAso(surf_T1,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],['3','8','9','14'])

        if self.random_rotation:
            surf_T1 , angle, vector = RandomRotationZ(surf_T1)
            angle = angle*np.pi / 180
            matrix_random_rotation = TransformRotationMatrix(vector, angle)
            matrix = np.matmul(matrix_random_rotation,matrix)


        if self.transform:
            surf_T1, matrix_transform = self.transform(surf_T1)

            matrix = np.matmul(matrix_transform,matrix)


        surf_T2 = ReadSurf(self.df.iloc[index]['T2'])
        surf_T2 = TransformSurf(surf_T2,matrix)


    def decomposition(surf):
        quit()


        




class TeethDatasetPatch(Dataset):
    def __init__(self,df,surf_property ,mount_point='',transform = False,landmark=[],test=False,prediction=False,random_rotation= False):
        self.df = df
        self.mount_point = mount_point

        self.surf_property = surf_property

        self.transform = transform
        self.landmark = landmark
        self.test = test
        self.prediction= prediction
        self.random_rotation = random_rotation


    def __len__(self):
            
        return len(self.df)

    def __getitem__(self, index) :
        if isinstance(self.df,list):
            surf = ReadSurf(self.df[index])
            # print(f'path scan {self.df[index]}')
            # surf = ReadSurf('/home/luciacev/Desktop/Data/IOSReg/files_not_organize/DeniseTest_json/P2_Upper.vtk')

        else :

            surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[index]["surf"][1:]))

        surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],['3','8','9','14'])

  

        if self.random_rotation:
            surf , angle, vector = RandomRotationZ(surf)
            angle = angle*np.pi / 180
            matrix_random_rotation = TransformRotationMatrix(vector, angle)
            matrix = np.matmul(matrix_random_rotation,matrix)


        if self.transform:
            surf, matrix_transform = self.transform(surf)

            matrix = np.matmul(matrix_transform,matrix)

        # scale = 3
        # scale_matrix = np.array([[scale,0,0,0],
        #                                      [0, scale,0 ,0],
        #                                      [0, 0, scale ,0],
        #                                      [0, 0, 0, 1]])
        # surf = TransformSurf(surf, scale_matrix)

        # matrix = np.matmul(scale_matrix,matrix)

        
        

        surf = ComputeNormals(surf) 
     

        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
        CN = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 




        if not self.prediction :


            pos_landmark = get_landmarks_position(os.path.join(self.mount_point,self.df.iloc[index]["landmark"][1:]),self.landmark,matrix,dic=True)

            pos_landmark = (pos_landmark[self.landmark[0]] + pos_landmark[self.landmark[1]])/2




            # LF = pos_landmard2seg_special(V,pos_landmark)
            # faces_pid0 = F[:,0:1]       
            # LF = torch.take(LF, faces_pid0)            
            # LF = LF.to(torch.int64)

            LF = torch.tensor(0)

            if self.test:
                print('test')
                # CL = pos_landmard2texture_special(V,pos_landmark)
                # CL = pos_Tshape_texture(V,pos_landmark)
                # CL = pos_landmard2seg(V,pos_landmark)
                CL = rectangle_patch_texture(surf,matrix)
                return V, F, CN, CL, pos_landmark
            
            return V, F, CN, LF
            
            
        else :

            return V, F, CN 
        
    def getSurf(self,idx):
        if isinstance(self.df,list):
            surf = ReadSurf(self.df[idx])
        else :
            surf = ReadSurf(os.path.join(self.mount_point,self.df.iloc[idx]["surf"][1:]))
        return surf
    
    def getName(self,idx):
        if isinstance(self.df,list):
            path = self.df[idx]
        else :
            path = os.path.join(self.mount_point,self.df.iloc[idx]["surf"][1:])
        name = os.path.basename(path)
        name , _ = os.path.splitext(name)

        return name



