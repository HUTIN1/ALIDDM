from torch import tensor
import torch
from random import choice
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from monai.transforms import ToTensor

from ALIDDM_utils import MeanScale, TransformVTK
import ALIDDM_utils
import utils








class UnitSurfTransform:

    def __init__(self, random_rotation=False):
        
        self.random_rotation = random_rotation

    def __call__(self, surf):

        surf = utils.GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = utils.RandomRotation(surf)
        return surf


class RandomRotation:
    def __call__(self,surf):
        kwargs={}
        surf , angle, vector = utils.RandomRotation(surf)
        kwargs['angle']=angle
        kwargs['vector'] = vector
        return  surf , kwargs



class PickTeethTransform:
    def __init__(self,surf_property):
         self.surf_property = surf_property


    def __call__(self, surf,tooth):
        kwargs ={}
        try :
            region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)
        except:
            region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars('Universal_ID'))),dtype=torch.int64)

        crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
        verts = vtk_to_numpy(surf.GetPoints().GetData())

        verts_crown = tensor(verts[crown_ids])

        if len(verts_crown)==0:
            return None           
        # print(verts_crown)
        mean,scale ,_ = MeanScale(verts = verts_crown)
        kwargs['mean']=mean
        kwargs['scale']=scale

        surf = TransformVTK(surf,mean,scale)

        return surf , kwargs





class PickLandmarkTransform(PickTeethTransform):
    def __init__(self,landmark,surf_property):
        super().__init__(surf_property)
        dic = {'UL7CL': 15, 'UL7CB': 15, 'UL7O': 15, 'UL7DB': 15, 'UL7MB': 15, 'UL7R': 15, 'UL7RIP': 15, 'UL7OIP': 15,
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
        self.tooth=dic[landmark]


    def __call__(self,surf) :
        return super().__call__(surf,self.tooth)
       



class RandomPickTeethTransform(PickTeethTransform):

    def __init__(self, surf_property):
        super().__init__(surf_property)


    def __call__(self, surf):
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)
        unique_ids = torch.unique(region_id)[1:-1]


        tooth = torch.randint(low=torch.min(unique_ids),high=torch.max(unique_ids),size=(1,))
        surf = super().__call__(surf,tooth)
       
        while surf is None:
            surf, _ = super().__call__(surf,tooth)
        
        return surf



class IterTeeth:
    def __init__(self,surf_property) -> None:
        self.surf_property = surf_property
        self.surf=None
        self.list_tooth=None
        self.iter=0
        self.PickTeethTransform = PickTeethTransform(surf_property)

    def __getitem__(self,surf):
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)
        unique_ids = torch.unique(region_id)[1:-1]
        self.list_tooth=unique_ids

        self.surf=surf


    def __iter__(self):
        self.iter=0
        return self

    def __next__(self):
        
        if len(self.list_tooth)<=self.iter:
            raise StopIteration
        out , _ = self.PickTeethTransform(self.surf,self.list_tooth[self.iter])
        while out is None :
            self.iter+=1
            if len(self.list_tooth)<=self.iter:
                raise StopIteration
            out, _ = self.PickTeethTransform(self.surf,self.list_tooth[self.iter])
        self.iter+=1
        return out
    



class MyCompose:
    def __init__(self,transform):
        self.transform = transform

    def __call__(self,data):
        dic ={}
        for transform in self.transform:
            data, kwargs = transform(data)
            if isinstance(kwargs,dict):
                for key, value in kwargs.items():
                    dic[key]=value
        return data, dic