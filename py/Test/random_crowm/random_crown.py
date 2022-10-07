#import torch
# from pytorch3d.io import load_obj

import os
from requests import delete
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch import int64, float32, clamp, tensor, cat, min, max, arccos, arctan, sqrt, pow, unique
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from monai.transforms import (
    ToTensor
)
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.structures import Meshes, Pointclouds
#from pytorch3d.render import look_at_view_transform,FoVPerspectiveCameras
from pytorch3d.utils import ico_sphere
import matplotlib.pyplot as plt
import numpy as np
from utils_crown import ReadSurf, ComputeNormals, ScaleSurf, Boundingbox, CreateIcosahedron , GetActor, GetColoredActor
from vtk import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkPolyData, vtkPolyDataMapper, vtkPoints, vtkPolyDataWriter, VTK_FLOAT, vtkCellArray, vtkIdList
from random import choice

# import dash_html_components as html
# import dash_vtk
# from dash import Dash, Input, Output
# from dash_vtk.utils import to_mesh_state




path = '/Users/luciacev-admin/Desktop/ALIDDM/ALIDDM/random_crowm/T3_17_L_segmented.vtk'

surf = ReadSurf(path)

surf, mean_arr, scale_factor= ScaleSurf(surf) 
surf = ComputeNormals(surf) 




verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()),dtype= float32)
faces = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
try :
    region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
except AttributeError :
    try :
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("predictedId"))),dtype=torch.int64)
    except AttributeError:
        region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID"))),dtype=torch.int64)
# region_id = clamp(region_id, min=0)


region_id_list= unique(region_id).numpy().tolist()
if 33 in region_id_list:
    region_id_list.remove(33)
print("region_id",region_id_list)
tooth = choice(region_id_list)
print(tooth)

print(max(region_id) - min(region_id),"-"*100)

print(type(region_id),region_id.size())
print(type(verts),verts.size())
print(type(faces),faces.size())

liste=[]
nb_pop = 0
nb = region_id.size(0) 
pourcentage = 0.05
display = int(nb * pourcentage)


print(region_id.shape)
crown_ids = torch.argwhere(region_id == tooth).reshape(-1)
print(crown_ids.shape)
verts_crown = verts[crown_ids]

min_coord = torch.min(verts_crown, dim=0).values
max_coord= torch.max(verts_crown, dim=0).values
center = (max_coord + min_coord)/2.0
scale = torch.linalg.norm(max_coord - center)

print(center, scale)
#
# mesh.offset_verts_(-center)
# mesh.scale_verts_((1.0 / float(scale)));
verts = (verts - center)/scale
print(verts.size(),faces.size())
mesh = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0))



mesh_sphere = ico_sphere(4)

# print(mesh_sphere.verts_packed().shape)
verts_sphere= mesh_sphere.verts_packed() * float(1.1)
point_sphere = Pointclouds(points=  [verts_sphere])





#point_mouth = Pointclouds(points=  [scale]) #, textures = region_id)
# #mesh = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0))
# # mesh.offset_verts_(-center)
# # mesh.scale_verts_((1.0 / float(scale)))

# # R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
# # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)


fig = plot_scene({
    "subplot1": {
        "sphere": point_sphere,
        "mouth" : mesh
    }
})
fig.show()

