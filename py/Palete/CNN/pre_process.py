import os
import sys
import torch
from vtk.util.numpy_support import vtk_to_numpy
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
TexturesVertex, FoVPerspectiveCameras, look_at_rotation,look_at_view_transform
)
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import vtk


from icp import vtkMeanTeeth, PrePreAso, ToothNoExist
from utils import ReadSurf, RandomRotation, WriteSurf, search
from tqdm import tqdm

from segmented_from_point import Segmentation





path = '/home/luciacev/Desktop/Data/IOSReg/ARON_GOLD/organize/Oriented/T1_seg/'
path_out = '/home/luciacev/Desktop/Data/IOSReg/ARON_GOLD/organize/Oriented/T1_testseg/'
files = search(path,'.vtk')['.vtk']
centroidf = vtkMeanTeeth([5,6,11,12,3,14],property='Universal_ID')
ratio_rect = 0.3
for file in tqdm(files) :

    surf = ReadSurf(file)
    surf_out = vtk.vtkPolyData()
    surf_out.DeepCopy(surf)

    try :
        surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],['3','8','9','14'])
        centroid = centroidf(surf)
    except ToothNoExist as error :
        print(f'Error {error}, file : {file}')


    
    V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
    
    haut_gauche1 = (centroid['5']+centroid['6'])/2
    haut_droite1 = (centroid['11']+centroid['12'])/2

    haut_droite = (1-ratio_rect) * haut_gauche1 + ratio_rect * haut_droite1
    haut_gauche = (1-ratio_rect) * haut_droite1 + ratio_rect * haut_gauche1



    bas_gauche1 = centroid['3']
    bas_droite1 = centroid['14']

    bas_droite = (1-ratio_rect) * bas_gauche1 + ratio_rect * bas_droite1
    bas_gauche = (1- ratio_rect) * bas_droite1 + ratio_rect * bas_gauche1







    _ , arg= Segmentation([haut_droite,haut_gauche,bas_gauche,bas_droite],vertex = V)


    V_label = torch.zeros((V.shape[0]))
    V_label[arg] = 1


    V_labels_prediction = numpy_to_vtk(V_label.cpu().numpy())
    V_labels_prediction.SetName('Pre_seg')


    surf_out.GetPointData().AddArray(V_labels_prediction)


    basename = os.path.basename(file)
    WriteSurf(surf_out,os.path.join(path_out,basename))