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
import glob
from torch_geometric.nn import knn_graph



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
        # self.train_ds = DatasetGCN(self.train_csv, self.landmark,self.transform)
        # self.val_ds = DatasetGCN(self.val_csv, self.landmark,self.transform)
        # self.test_ds = DatasetGCN(self.test_csv , self.landmark,self.transform)
        self.train_ds = DatasetGCNSegTeeth(self.train_csv, self.landmark,self.transform)
        self.val_ds = DatasetGCNSegTeeth(self.val_csv, self.landmark,self.transform)
        self.test_ds = DatasetGCNSegTeeth(self.test_csv , self.landmark,self.transform)

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
        radius = 0.01


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
        landmarks_position = (pos_center - mean_arr) / scale_factor
        return landmarks_position
    

    def MeanScale(self,verts = None):

        min_coord = torch.min(verts,0)[0]
        max_coord= torch.max(verts,0)[0]
        mean = (max_coord + min_coord)/2.0
        mean= mean.numpy()
        scale = np.linalg.norm(max_coord.numpy() - mean)

        return mean, scale
    
    def getName(self,index):
        return self.df.iloc[index]['surf']
    
    def getLandmark(self,index):

        mounth = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/data_base/'
        surf = self.ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        # F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        

        mean , scale = self.MeanScale(verts = V)
        
        V = (V - tensor(mean))/tensor(scale)




        landmark_pos = self.get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,mean,scale)

        return landmark_pos
    


class DatasetGCNSegTeeth(DatasetGCN):
    def __init__(self, path, landmark, transfrom) -> None:
        super().__init__(path, landmark, transfrom)
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


    def __getitem__(self, index):
        mounth = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/data_base/'
        surf = self.ReadSurf(os.path.join(mounth,self.df.iloc[index]['surf']))

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
        crown_ids = torch.argwhere(region_id == self.dic[self.landmark]).reshape(-1)


        verts_crown = V[crown_ids]


        mean , scale = self.MeanScale(verts = verts_crown)
        
        verts_crown = (verts_crown - tensor(mean))/tensor(scale)

        edge_index = knn_graph(verts_crown, k = 7)


        data = Data(x= verts_crown , edge_index=edge_index)

        landmark_pos = self.get_landmarks_position(os.path.join(mounth,self.df.iloc[index]['landmark']),self.landmark,mean,scale)
        data.segmentation_labels = self.segmentation(verts_crown,landmark_pos)

        # data = self.transform(data)

        return data
    
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



        surf = self.ReadSurf(self.list_files[index])

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
        crown_ids = torch.argwhere(region_id == self.dic[self.landmark]).reshape(-1)


        verts_crown = V[crown_ids]


        mean , scale = self.MeanScale(verts = verts_crown)
        
        verts_crown = (verts_crown - tensor(mean))/tensor(scale)

        edge_index = knn_graph(verts_crown, k = 7)


        data = Data(x= verts_crown , edge_index=edge_index)

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
        surf = self.ReadSurf(self.list_files[index])

        V = tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(float32)
        F = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(int64)

        mean , scale = self.MeanScale(verts = V)
        
        V = (V - tensor(mean))/tensor(scale)


        data = Data(x= V , face = F.t())

        data = self.transform(data)

        return data , tensor(mean), tensor(scale)
    

    def getName(self,index):
        file = self.list_files[index]
        name , _ = os.path.splitext(os.path.basename(file))
        return name
   
    