import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch import tensor, float32, int64
from pytorch3d.transforms import random_rotation
import torch

from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
import numpy as np
import os
import vtk
import json
import glob
from torch_geometric.nn import knn_graph
from utils_GCN import ComputeNormals, GetColorArray, MeanScale, ReadSurf, Downscale, get_landmarks_position, segmentationLandmarks, RemoveBase
from icp import PrePreAso
from segmented_from_point import Segmentation


class DataModuleGCN(pl.LightningDataModule):
    def __init__(self,train_csv,val_csv,test_csv,landmark, batch_size,radius, transform,surf_transform,num_worker = 4, drop_last = False,mouth_path='.') -> None:
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.landmark = landmark
        self.batch_size = batch_size
        self.num_worker = num_worker

        self.drop_last = drop_last
        self.prepare_data_per_node = None
        self._log_hyperparams = None
        self.transform = transform
        self.radius = radius
        self.surf_transform = surf_transform
        self.mouth_path = mouth_path



    def setup(self, stage = None) -> None:
        self.train_ds = DatasetGCN(self.train_csv, self.landmark,self.transform,self.radius,self.surf_transform,mouth_path = self.mouth_path,surf_transfrom=self.surf_transform)
        self.val_ds = DatasetGCN(self.val_csv, self.landmark,self.transform,self.radius,self.surf_transform,mouth_path = self.mouth_path,surf_transfrom=self.surf_transform)
        self.test_ds = DatasetGCN(self.test_csv , self.landmark,self.transform,self.radius,self.surf_transform,mouth_path = self.mouth_path,surf_transfrom=self.surf_transform)
        # self.train_ds = DatasetGCNSegTeeth(self.train_csv, self.landmark,self.transform,self.radius)
        # self.val_ds = DatasetGCNSegTeeth(self.val_csv, self.landmark,self.transform,self.radius)
        # self.test_ds = DatasetGCNSegTeeth(self.test_csv , self.landmark,self.transform,self.radius)

    def train_dataloader(self) :
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers =self.num_worker, pin_memory = True, persistent_workers = True, drop_last = self.drop_last , shuffle=True)
    
    def val_dataloader(self) :
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers =self.num_worker, pin_memory = True, persistent_workers = True, drop_last = self.drop_last, shuffle=True)
    
    def test_dataloader(self) :
        return DataLoader(self.test_ds, batch_size=1, num_workers =self.num_worker, pin_memory = True, persistent_workers = True, drop_last = self.drop_last)
    
    def prepare_data(self) -> None:
        pass



class DatasetGCN(Dataset):
    def __init__(self,path,landmark,radius, surf_transfrom) -> None:
        self.df = self.setup(path)
        self.landmark = landmark
        self.radius = radius
        self.surf_transform = surf_transfrom


    def setup(self,path):
        return pd.read_csv(path)
    

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) :
        mounth = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/'

        surf = ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))
       
        matrix = np.eye(4)
        surf, matrix_or = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['4','9','10','15'])
        matrix = np.matmul(matrix_or,matrix)
        if self.surf_transform :
            surf , matrix_transfrom = self.surf_transform(surf)
            matrix = np.matmul(matrix_transfrom,matrix)

        surf = ComputeNormals(surf)

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        CN = tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32)  


        

        

        landmark_pos = get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,matrix = matrix)
        
        V, index = Segmentation(landmark_pos, vertex = V2)
        CN  = CN[index , :]
        edge_index = knn_graph(V, k = 7)
        VCN = torch.cat((V,CN),dim=-1)
        data = Data(x= VCN , edge_index=edge_index )
        data.segmentation_labels = segmentationLandmarks(V,landmark_pos,self.radius)
        

        return data


    

    
    def getName(self,index):
        return self.df.iloc[index]['surf']
    
    def getLandmark(self,index):

        mounth = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/'
        surf = ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        # F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        
        matrix = np.eye(4)
        if self.surf_transform :
            surf , matrix_transform = self.surf_transform(surf)
            matrix = np.matmul(matrix,matrix_transform)

        landmark_pos = get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,matrix=matrix)

        return landmark_pos
    

class DatasetGCNSeg():
    def __init__(self,path,landmark,radius, surf_transfrom,mouth_path) -> None:
        self.df = self.setup(path)
        self.landmark = landmark
        self.radius = radius
        self.surf_transform = surf_transfrom
        self.mouth_path = mouth_path


    def setup(self,path):
        return pd.read_csv(path)
    

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) :


        surf = ReadSurf(os.path.join(self.mouth_path,self.df.iloc[index]['surf']))
       
        matrix = np.eye(4)
        surf, matrix_or = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['4','9','10','15'])
        matrix = np.matmul(matrix_or,matrix)
        if self.surf_transform :
            surf , matrix_transfrom = self.surf_transform(surf)
            matrix = np.matmul(matrix_transfrom,matrix)

        surf = ComputeNormals(surf)

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        CN = tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 
    


        

        

        landmark_pos = get_landmarks_position(os.path.join(self.mouth_path,self.df.iloc[index]['landmark']),self.landmark,matrix = matrix)


        V2, index_remove_base = RemoveBase(vertex=V)
        V_segmented, index = Segmentation(landmark_pos, vertex = V2)
        segmentation_labels = torch.zeros((V2.shape[0],1))

        segmentation_labels[index,:] = torch.ones((index.shape[0],1))

        edge_index = knn_graph(V2, k = 7)
        VCN = torch.cat((V2,CN[index_remove_base,:]),dim=-1)
        data = Data(x= VCN , edge_index=edge_index )
        data.segmentation_labels = segmentation_labels
        

        return data


    

    
    def getName(self,index):
        return self.df.iloc[index]['surf']  
    
    def getLandmark(self,index):
        surf = ReadSurf(os.path.join(self.mouth_path,self.df.iloc[index]['surf']))

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        # F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        
        matrix = np.eye(4)
        if self.surf_transform :
            surf , matrix_transform = self.surf_transform(surf)
            matrix = np.matmul(matrix,matrix_transform)

        landmark_pos = get_landmarks_position(os.path.join(self.mouth_path,self.df.iloc[index]['landmark']),self.landmark,matrix=matrix)

        return landmark_pos
    



class DatasetGCNSegPred():
    def __init__(self, path, surf_transfrom) -> None:
        self.df = self.search(path,'.vtk')['.vtk']
        self.surf_transform = surf_transfrom

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):

        surf = ReadSurf(self.df[index])
       
        matrix = np.eye(4)
        surf, matrix_or = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['4','9','10','15'])
        matrix = np.matmul(matrix_or,matrix)
        if self.surf_transform :
            surf , matrix_transfrom = self.surf_transform(surf)
            matrix = np.matmul(matrix_transfrom,matrix)

        surf = ComputeNormals(surf)

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        CN = tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 

        V2, index_remove_base = RemoveBase(vertex=V)
        edge_index = knn_graph(V2, k = 7)
        VCN = torch.cat((V2,CN[index_remove_base,:]),dim=-1)
        data = Data(x= VCN , edge_index=edge_index )
        return data ,V , index_remove_base, torch.tensor(matrix)
    
    def getName(self,index):
        return self.df[index] 
    
    def search(self,path,*args):
            """
            Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

            Example:
            args = ('json',['.nii.gz','.nrrd'])
            return:
                {
                    'json' : ['path/a.json', 'path/b.json','path/c.json'],
                    '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                    '.nrrd.gz' : ['path/c.nrrd']
                }
            """
            arguments=[]
            for arg in args:
                if type(arg) == list:
                    arguments.extend(arg)
                else:
                    arguments.append(arg)
            return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}
    

class DatasetGCNSegTeeth(DatasetGCN):
    def __init__(self, path, landmark, transfrom,radius) -> None:
        super().__init__(path, landmark, transfrom,radius)
        self.dic = {'UL7CL': 15, 'UL7CB': 15, 'UL7O': 15, 'UL7DB': 15, 'UL7MB': 15, 'UL7R': 15, 'UL7RIP': 15, 'UL7OIP': 15,
         'UL6CL': 14, 'UL6CB': 14, 'UL6O': 14, 'UL6DB': 14, 'UL6MB': 14, 'UL6R': 14, 'UL6RIP': 14, 'UL6OIP': 14,
         'UL5CL': 13, 'UL5CB': 13, 'UL5O': 13, 'UL5DB': 13, 'UL5MB': 13, 'UL5R': 13, 'UL5RIP': 13, 'UL5OIP': 13, 
         'UL4CL': 12, 'UL4CB': 12, 'UL4O': 12, 'UL4DB': 12, 'UL4MB': 12, 'UL4R': 12, 'UL4RIP': 12, 'UL4OIP': 12,
          'UL3CL': 11, 'UL3CB': 11, 'UL3O': 11, 'UL3DB': 11, 'UL3MB': 11, 'UL3R': 11, 'UL3RIP': 11, 'UL3OIP': 11, 
          'UL2CL': 10, 'UL2CB': 10, 'UL2O': 10, 'UL2DB': 10, 'UL2MB': 10, 'UL2R': 10, 'UL2RIP': 10, 'UL2OIP': 10, 
          'UL1CL': 9, 'UL1CB': 9, 'UL1O': 9, 'UL1DB': 9, 'UL1MB': 9, 'UL1R': 9, 'UL1RIP': 9, 'UL1OIP': 9, 'UR1CL': 8, 
          'UR1CB': 8, 'UR1O': 8, 'UR1DB': 8, 'UR1MB': 8, 'UR1R': 8, 'UR1RIP': 8, 'UR1OIP': 8, 'UR2CL': 7, 'UR2CB': 7, 
          'UR2O': 7, 'UR2DB': 7, 'UR2MB': 7, 'UR2R': 7, 'UR2RIP': 7, 'UR2OIP': 7, 'UR3CL': 6, 'UR3CB': 6, 'UR3O': 6, 
          'UR3DB': 6, 'UR3MB': 6, 'UR3R': 6, 'UR3RIP': 6, 'UR3OIP': 6, 'UR4CL': 5, 'UR4CB': 5, 'UR4O': 5, 'UR4DB': 5, 
          'UR4MB': 5, 'UR4R': 5, 'UR4RIP': 5, 'UR4OIP': 5, 'UR5CL': 4, 'UR5CB': 4, 'UR5O': 4, 'UR5DB': 4, 'UR5MB': 4, 
          'UR5R': 4, 'UR5RIP': 4, 'UR5OIP': 4, 'UR6CL': 3, 'UR6CB': 3, 'UR6O': 3, 'UR6DB': 3, 'UR6MB': 3, 'UR6R': 3, 
          'UR6RIP': 3, 'UR6OIP': 3, 'UR7CL': 1, 'UR7CB': 1, 'UR7O': 1, 'UR7DB': 1, 'UR7MB': 1, 'UR7R': 1, 'UR7RIP': 1, 
          'UR7OIP': 1, 'LL7CL': 18, 'LL7CB': 18, 'LL7O': 18, 'LL7DB': 18, 'LL7MB': 18, 'LL7R': 18, 'LL7RIP': 18, 'LL7OIP': 18, 
          'LL6CL': 19, 'LL6CB': 19, 'LL6O': 19, 'LL6DB': 19, 'LL6MB': 19, 'LL6R': 19, 'LL6RIP': 19, 'LL6OIP': 19, 'LL5CL': 20, 
          'LL5CB': 20, 'LL5O': 20, 'LL5DB': 20, 'LL5MB': 20, 'LL5R': 20, 'LL5RIP': 20, 'LL5OIP': 20, 'LL4CL': 21, 'LL4CB': 21, 
          'LL4O': 21, 'LL4DB': 21, 'LL4MB': 21, 'LL4R': 21, 'LL4RIP': 21, 'LL4OIP': 21, 'LL3CL': 22, 'LL3CB': 22, 'LL3O': 22, 
          'LL3DB': 22, 'LL3MB': 22, 'LL3R': 22, 'LL3RIP': 22, 'LL3OIP': 22, 'LL2CL': 23, 'LL2CB': 23, 'LL2O': 23, 'LL2DB': 23, 
          'LL2MB': 23, 'LL2R': 23, 'LL2RIP': 23, 'LL2OIP': 23, 'LL1CL': 24, 'LL1CB': 24, 'LL1O': 24, 'LL1DB': 24, 'LL1MB': 24, 
          'LL1R': 24, 'LL1RIP': 24, 'LL1OIP': 24, 'LR1CL': 25, 'LR1CB': 25, 'LR1O': 25, 'LR1DB': 25, 'LR1MB': 25, 'LR1R': 25, 
          'LR1RIP': 25, 'LR1OIP': 25, 'LR2CL': 26, 'LR2CB': 26, 'LR2O': 26, 'LR2DB': 26, 'LR2MB': 26, 'LR2R': 26, 'LR2RIP': 26, 
          'LR2OIP': 26, 'LR3CL': 27, 'LR3CB': 27, 'LR3O': 27, 'LR3DB': 27, 'LR3MB': 27, 'LR3R': 27, 'LR3RIP': 27, 'LR3OIP': 27, 
          'LR4CL': 28, 'LR4CB': 28, 'LR4O': 28, 'LR4DB': 28, 'LR4MB': 28, 'LR4R': 28, 'LR4RIP': 28, 'LR4OIP': 28, 'LR5CL': 29, 
          'LR5CB': 29, 'LR5O': 29, 'LR5DB': 29, 'LR5MB': 29, 'LR5R': 29, 'LR5RIP': 29, 'LR5OIP': 29, 'LR6CL': 30, 'LR6CB': 30, 
          'LR6O': 30, 'LR6DB': 30, 'LR6MB': 30, 'LR6R': 30, 'LR6RIP': 30, 'LR6OIP': 30, 'LR7CL': 31, 'LR7CB': 31, 'LR7O': 31, 
          'LR7DB': 31, 'LR7MB': 31, 'LR7R': 31, 'LR7RIP': 31, 'LR7OIP': 31,'Palete':33}


    def __getitem__(self, index):
        mounth = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/data_base/'
        surf = ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))
        surf = ComputeNormals(surf)

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        CN = tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32)  
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
        crown_ids = torch.argwhere(region_id == self.dic[self.landmark]).reshape(-1)


        verts_crown = V[crown_ids]
        CN = CN[crown_ids]


        mean , scale = MeanScale(verts = verts_crown)

        verts_crown = Downscale(verts_crown,mean, scale)
        
        # verts_crown = (verts_crown - tensor(mean))*tensor(scale)

        edge_index = knn_graph(verts_crown, k = 7)


        x = torch.cat((verts_crown,CN),dim=-1)
        data = Data(x= x , edge_index=edge_index)

        landmark_pos = get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,mean,scale)
        data.segmentation_labels = segmentationLandmarks(verts_crown,landmark_pos,self.radius)

        # data = self.transform(data)

        return data

    def getLandmark(self,index):

        mounth = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/data_base/'
        surf = ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
        crown_ids = torch.argwhere(region_id == self.dic[self.landmark]).reshape(-1)


        verts_crown = V[crown_ids]

        # F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        

        mean , scale = MeanScale(verts = verts_crown)
        
        V = Downscale(V,mean,scale)


        landmark_pos = get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,mean,scale)

        return landmark_pos

    





class DatasetGCNSegTeethPrediction(DatasetGCNSegTeeth):
    def __init__(self, path, landmark, transfrom) -> None:
        self.landmark = landmark
        self.list_files = self.search(path,'.vtk')['.vtk']
        self.dic = {'UL7CL': 15, 'UL7CB': 15, 'UL7O': 15, 'UL7DB': 15, 'UL7MB': 15, 'UL7R': 15, 'UL7RIP': 15, 'UL7OIP': 15,
         'UL6CL': 14, 'UL6CB': 14, 'UL6O': 14, 'UL6DB': 14, 'UL6MB': 14, 'UL6R': 14, 'UL6RIP': 14, 'UL6OIP': 14,
         'UL5CL': 13, 'UL5CB': 13, 'UL5O': 13, 'UL5DB': 13, 'UL5MB': 13, 'UL5R': 13, 'UL5RIP': 13, 'UL5OIP': 13, 
         'UL4CL': 12, 'UL4CB': 12, 'UL4O': 12, 'UL4DB': 12, 'UL4MB': 12, 'UL4R': 12, 'UL4RIP': 12, 'UL4OIP': 12,
          'UL3CL': 11, 'UL3CB': 11, 'UL3O': 11, 'UL3DB': 11, 'UL3MB': 11, 'UL3R': 11, 'UL3RIP': 11, 'UL3OIP': 11, 
          'UL2CL': 10, 'UL2CB': 10, 'UL2O': 10, 'UL2DB': 10, 'UL2MB': 10, 'UL2R': 10, 'UL2RIP': 10, 'UL2OIP': 10, 
          'UL1CL': 9, 'UL1CB': 9, 'UL1O': 9, 'UL1DB': 9, 'UL1MB': 9, 'UL1R': 9, 'UL1RIP': 9, 'UL1OIP': 9, 'UR1CL': 8, 
          'UR1CB': 8, 'UR1O': 8, 'UR1DB': 8, 'UR1MB': 8, 'UR1R': 8, 'UR1RIP': 8, 'UR1OIP': 8, 'UR2CL': 7, 'UR2CB': 7, 
          'UR2O': 7, 'UR2DB': 7, 'UR2MB': 7, 'UR2R': 7, 'UR2RIP': 7, 'UR2OIP': 7, 'UR3CL': 6, 'UR3CB': 6, 'UR3O': 6, 
          'UR3DB': 6, 'UR3MB': 6, 'UR3R': 6, 'UR3RIP': 6, 'UR3OIP': 6, 'UR4CL': 5, 'UR4CB': 5, 'UR4O': 5, 'UR4DB': 5, 
          'UR4MB': 5, 'UR4R': 5, 'UR4RIP': 5, 'UR4OIP': 5, 'UR5CL': 4, 'UR5CB': 4, 'UR5O': 4, 'UR5DB': 4, 'UR5MB': 4, 
          'UR5R': 4, 'UR5RIP': 4, 'UR5OIP': 4, 'UR6CL': 3, 'UR6CB': 3, 'UR6O': 3, 'UR6DB': 3, 'UR6MB': 3, 'UR6R': 3, 
          'UR6RIP': 3, 'UR6OIP': 3, 'UR7CL': 1, 'UR7CB': 1, 'UR7O': 1, 'UR7DB': 1, 'UR7MB': 1, 'UR7R': 1, 'UR7RIP': 1, 
          'UR7OIP': 1, 'LL7CL': 18, 'LL7CB': 18, 'LL7O': 18, 'LL7DB': 18, 'LL7MB': 18, 'LL7R': 18, 'LL7RIP': 18, 'LL7OIP': 18, 
          'LL6CL': 19, 'LL6CB': 19, 'LL6O': 19, 'LL6DB': 19, 'LL6MB': 19, 'LL6R': 19, 'LL6RIP': 19, 'LL6OIP': 19, 'LL5CL': 20, 
          'LL5CB': 20, 'LL5O': 20, 'LL5DB': 20, 'LL5MB': 20, 'LL5R': 20, 'LL5RIP': 20, 'LL5OIP': 20, 'LL4CL': 21, 'LL4CB': 21, 
          'LL4O': 21, 'LL4DB': 21, 'LL4MB': 21, 'LL4R': 21, 'LL4RIP': 21, 'LL4OIP': 21, 'LL3CL': 22, 'LL3CB': 22, 'LL3O': 22, 
          'LL3DB': 22, 'LL3MB': 22, 'LL3R': 22, 'LL3RIP': 22, 'LL3OIP': 22, 'LL2CL': 23, 'LL2CB': 23, 'LL2O': 23, 'LL2DB': 23, 
          'LL2MB': 23, 'LL2R': 23, 'LL2RIP': 23, 'LL2OIP': 23, 'LL1CL': 24, 'LL1CB': 24, 'LL1O': 24, 'LL1DB': 24, 'LL1MB': 24, 
          'LL1R': 24, 'LL1RIP': 24, 'LL1OIP': 24, 'LR1CL': 25, 'LR1CB': 25, 'LR1O': 25, 'LR1DB': 25, 'LR1MB': 25, 'LR1R': 25, 
          'LR1RIP': 25, 'LR1OIP': 25, 'LR2CL': 26, 'LR2CB': 26, 'LR2O': 26, 'LR2DB': 26, 'LR2MB': 26, 'LR2R': 26, 'LR2RIP': 26, 
          'LR2OIP': 26, 'LR3CL': 27, 'LR3CB': 27, 'LR3O': 27, 'LR3DB': 27, 'LR3MB': 27, 'LR3R': 27, 'LR3RIP': 27, 'LR3OIP': 27, 
          'LR4CL': 28, 'LR4CB': 28, 'LR4O': 28, 'LR4DB': 28, 'LR4MB': 28, 'LR4R': 28, 'LR4RIP': 28, 'LR4OIP': 28, 'LR5CL': 29, 
          'LR5CB': 29, 'LR5O': 29, 'LR5DB': 29, 'LR5MB': 29, 'LR5R': 29, 'LR5RIP': 29, 'LR5OIP': 29, 'LR6CL': 30, 'LR6CB': 30, 
          'LR6O': 30, 'LR6DB': 30, 'LR6MB': 30, 'LR6R': 30, 'LR6RIP': 30, 'LR6OIP': 30, 'LR7CL': 31, 'LR7CB': 31, 'LR7O': 31, 
          'LR7DB': 31, 'LR7MB': 31, 'LR7R': 31, 'LR7RIP': 31, 'LR7OIP': 31}


    def __len__(self):
        return len(self.list_files)


    def search(self,path,*args):
            """
            Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

            Example:
            args = ('json',['.nii.gz','.nrrd'])
            return:
                {
                    'json' : ['path/a.json', 'path/b.json','path/c.json'],
                    '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                    '.nrrd.gz' : ['path/c.nrrd']
                }
            """
            arguments=[]
            for arg in args:
                if type(arg) == list:
                    arguments.extend(arg)
                else:
                    arguments.append(arg)
            return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}
    
    def __getitem__(self, index):



        surf = ReadSurf(self.list_files[index])
        # print(f'surf {surf}')
        surf = ComputeNormals(surf)

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        CN = tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32)
        region_id = tensor(vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID")),dtype=torch.int64)
        crown_ids = torch.argwhere(region_id == self.dic[self.landmark]).reshape(-1)


        verts_crown = V[crown_ids]
        CN = CN[crown_ids]


        mean , scale = MeanScale(verts = verts_crown)

        verts_crown = Downscale(verts_crown,mean,scale)
        
        # verts_crown = (verts_crown - tensor(mean))/tensor(scale)

        edge_index = knn_graph(verts_crown, k = 7)


        data = Data(x= torch.cat((verts_crown,CN),dim=-1) , edge_index=edge_index)

        return data ,tensor(mean), tensor(scale)
    
    def getName(self,index):
        file = self.list_files[index]
        name , _ = os.path.splitext(os.path.basename(file))
        return name
   







class DatasetGCNPrecdition(DatasetGCN):
    def __init__(self,path,transform) -> None:
        self.list_files = self.search(path,'.vtk')['.vtk']
        self.transform = transform


    def search(self,path,*args):
        """
        Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

        Example:
        args = ('json',['.nii.gz','.nrrd'])
        return:
            {
                'json' : ['path/a.json', 'path/b.json','path/c.json'],
                '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                '.nrrd.gz' : ['path/c.nrrd']
            }
        """
        arguments=[]
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
            else:
                arguments.append(arg)
        return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index) :
        surf = ReadSurf(self.list_files[index])
        surf = ComputeNormals(surf)

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        CN = tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32)

        mean , scale = MeanScale(verts = V)
        
        # V = (V - tensor(mean))/tensor(scale)

        V = Downscale(V,mean,scale)


        data = Data(x= torch.cat((V,CN),dim=-1) , face = F.t())

        data = self.transform(data)

        return data , tensor(mean), tensor(scale)
    

    def getName(self,index):
        file = self.list_files[index]
        name , _ = os.path.splitext(os.path.basename(file))
        return name
   
    