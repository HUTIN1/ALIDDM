from torch import tensor
import torch
from random import choice
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from monai.transforms import ToTensor

from ALIDDM_utils import MeanScale
import utils



class RandomPickTeethTransform:

    def __init__(self, surf_property):
        self.surf_property = surf_property


    def __call__(self, surf):
        print('start randompickteethtransfrom')
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(self.surf_property))),dtype=torch.int64)
        unique_ids = torch.unique(region_id).cpu().tolist()
        if 33 in unique_ids:
            unique_ids.remove(33)

        tooth = choice(unique_ids)
        crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        verts_crown = verts[crown_ids]
        print('randompickteethtransfrom before meanscale')
        mean,scale ,surf = MeanScale(verts = verts_crown)
        print('scale',scale)
        print('randompickteethtransfrom after meanscale')
        print('randompickteethtransfrom before GenUnitsurf')
        surf = utils.GetUnitSurf(surf, mean_arr= mean, scale_factor = 1/scale)
        print('randompickteethtransfrom after GenUnitsurf')
        print('end randompickteehttransform')
        

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
        print('in random rotation')
        return utils.RandomRotation(surf)