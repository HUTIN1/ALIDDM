import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch import tensor, float32, int64
import torch

from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
import numpy as np
import os
import vtk
import json



class DataModuleGCN(pl.LightningDataModule):
    def __init__(self,train_csv,val_csv,test_csv,landmark, batch_size, transform,num_worker = 4, drop_last = False) -> None:
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



    def setup(self, stage = None) -> None:
        self.train_ds = DatasetGCN(self.train_csv, self.landmark,self.transform)
        self.val_ds = DatasetGCN(self.val_csv, self.landmark,self.transform)
        self.test_ds = DatasetGCN(self.test_csv , self.landmark,self.transform)


    def train_dataloader(self) :
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers =self.num_worker, pin_memory = True, persistent_workers = True, drop_last = self.drop_last )
    
    def val_dataloader(self) :
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers =self.num_worker, pin_memory = True, persistent_workers = True, drop_last = self.drop_last)
    
    def test_dataloader(self) :
        return DataLoader(self.test_ds, batch_size=1, num_workers =self.num_worker, pin_memory = True, persistent_workers = True, drop_last = self.drop_last)
    
    def prepare_data(self) -> None:
        pass



class DatasetGCN(Dataset):
    def __init__(self,path,landmark,transfrom) -> None:
        self.df = self.setup(path)
        self.landmark = landmark
        self.transform = transfrom


    def setup(self,path):
        return pd.read_csv(path)
    

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) :
        mounth = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/data_base/'
        surf = self.ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)

        mean , scale = self.MeanScale(verts = V)
        
        V = (V - tensor(mean))/tensor(scale)


        data = Data(x= V , face = F.t())

        landmark_pos = self.get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,mean,scale)
        data.segmentation_labels = self.segmentation(V,landmark_pos)

        data = self.transform(data)

        return data


    def segmentation(self,vertex , landmark_pos):
        texture = torch.zeros(size=(vertex.shape[0],1),dtype=int64)
        vertex = vertex.to(torch.float64)
        radius = 0.04


        landmark_pos = tensor(np.array(landmark_pos)).unsqueeze(0)
        distance = torch.cdist(landmark_pos,vertex,p=2)
        minvalue = torch.min(distance)
        distance = distance - minvalue
        _, index_pos_land = torch.nonzero((distance<radius),as_tuple=True)
        for i in index_pos_land:

            texture[i]=1
        return texture
    
    def ReadSurf(self,path):


        fname, extension = os.path.splitext(path)
        extension = extension.lower()
        if extension == ".vtk":
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            surf = reader.GetOutput()

        return surf
    

    def get_landmarks_position(self,path,landmark, mean_arr, scale_factor):

        data = json.load(open(os.path.join(path)))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']

        landmarks_pos = None
        # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
        for lm in landmarks_lst:
            label = lm["label"]
            if label == landmark:
                landmarks_pos = np.array(self.Downscale(lm["position"],mean_arr,scale_factor),)
                continue
        
        return landmarks_pos
    

    def Downscale(self,pos_center,mean_arr,scale_factor):
        landmarks_position = (pos_center - mean_arr) * scale_factor
        return landmarks_position
    

    def MeanScale(self,verts = None):

        min_coord = torch.min(verts,0)[0]
        max_coord= torch.max(verts,0)[0]
        mean = (max_coord + min_coord)/2.0
        mean= mean.numpy()
        scale = np.linalg.norm(max_coord.numpy() - mean)

        return mean, scale
            