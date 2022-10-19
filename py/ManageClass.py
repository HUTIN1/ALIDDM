from torch import tensor
import torch
from random import choice
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from monai.transforms import ToTensor

import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'Tools')
sys.path.append(mymodule_dir)
from ALIDDM_utils import FocusTeeth
import utils



class RandomPickTeethTransform:

    def __init__(self, surf_property, unique_ids=None):
        self.surf_property = surf_property
        self.unique_ids = unique_ids

    def __call__(self, surf,MeanScale =False):
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)

        if self.unique_ids is None:
            unique_ids= torch.unique(region_id).numpy().tolist()  #create list with all number segmentation
        else:
            unique_ids = self.unique_ids

        tooth = choice(unique_ids)
        mean , scale , surf = FocusTeeth(surf,self.surf_property,tooth)

        if MeanScale:
            return mean,scale ,surf

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