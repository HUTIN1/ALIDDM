from torch import tensor
import torch
from random import choice
import numpy as np
import ALIDDM.py.Tools.utils as utils
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from monai.transforms import ToTensor
from ALIDDM.py.Tools.ALIDDM_utils import FocusTeeth



class RandomPickTeethTransform:

    def __init__(self, surf_property, unique_ids=None):
        self.surf_property = surf_property, 
        self.unique_ids = unique_ids

    def __call__(self, surf):

        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)

        if self.unique_ids is None:
            unique_ids= torch.unique(region_id).numpy().tolist()  #create list with all number segmentation
        else:
            unique_ids = self.unique_ids

        tooth = choice(unique_ids)
        _ , _ , surf = FocusTeeth(surf,self.surf_property,tooth)


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