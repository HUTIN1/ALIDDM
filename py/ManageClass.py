from torch import tensor
import torch
from random import choice
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from monai.transforms import ToTensor

from ALIDDM_utils import MeanScale, TransformVTK
import utils



class RandomPickTeethTransform:

    def __init__(self, surf_property):
        self.surf_property = surf_property


    def __call__(self, surf):
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)
        unique_ids = torch.unique(region_id)[1:-1]


        tooth = torch.randint(low=torch.min(unique_ids),high=torch.max(unique_ids),size=(1,))
        crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
        verts = vtk_to_numpy(surf.GetPoints().GetData())

        verts_crown = tensor(verts[crown_ids])

        while len(verts_crown) ==0 :
                tooth = torch.randint(low=torch.min(unique_ids),high=torch.max(unique_ids),size=(1,))
                crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
                verts = vtk_to_numpy(surf.GetPoints().GetData())

                verts_crown = tensor(verts[crown_ids])

        # print(verts_crown)
        mean,scale ,_ = MeanScale(verts = verts_crown)

        surf = TransformVTK(surf,mean,scale)

        
        return surf




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
        return utils.RandomRotation(surf)