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
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas, blending
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

import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'..','..','Tools')
sys.path.append(mymodule_dir)
from utils import RandomRotation, GetColorArray
from ALIDDM_utils import FocusTeeth

# import dash_html_components as html
# import dash_vtk
# from dash import Dash, Input, Output
# from dash_vtk.utils import to_mesh_state




path = '/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/challenge_teeth_all_vtk/train/01FURZ62_lower.vtk'


device1 = torch.device('cuda:1')
surf_property = "PredictedID"
surf = ReadSurf(path)


surf, angle ,vectorrotation = RandomRotation(surf)





# try :
#     region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
# except AttributeError :
#     try :
#         region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("predictedId"))),dtype=torch.int64)
#     except AttributeError:
#         try:
#             region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID"))),dtype=torch.int64)
#         except AttributeError:
#             region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars("UniversalID"))),dtype=torch.int64)
# region_id = clamp(region_id, min=0)



#
# mesh.offset_verts_(-center)
# mesh.scale_verts_((1.0 / float(scale)));
CLF = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(surf_property))),dtype=torch.int64)
unique_ids = torch.unique(CLF).cpu().tolist()
if 33 in unique_ids :
    unique_ids.remove(33)

id_tooth = choice(unique_ids)


mean, scale, surf = FocusTeeth(surf,"PredictedID",id_tooth)
surf = ComputeNormals(surf) 

verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()),dtype= float32)
faces = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
CLF= ToTensor(dtype=torch.float32)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
CLF = CLF.unsqueeze(0)

textures_normal = TexturesVertex(verts_features=CLF)

mesh = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=textures_normal).to(device1)


cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90,device=device1) # Initialize a perspective camera.

raster_settings = RasterizationSettings(        
    image_size=320, 
    blur_radius=0, 
    faces_per_pixel=1, 
    max_faces_per_bin=200000
)

lights = PointLights(device=device1) # light in front of the object. 

rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )

b = blending.BlendParams(background_color=(0,0,0))
renderer = MeshRenderer(
    rasterizer=rasterizer,
    shader=HardPhongShader(cameras=cameras, lights=lights,blend_params=b,device=device1)
)

ico_verts = torch.tensor([[1,0,0]]).to(dtype=torch.float32).to(device1)
PF = []
X = []

for camera_position in ico_verts:

    camera_position = camera_position.unsqueeze(0)

    R = look_at_rotation(camera_position).to(device1)  # (1, 3, 3)
    T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0].to(device1)   # (1, 3)

    images = renderer(meshes_world=mesh.clone(), R=R, T=T)
    fragments = renderer.rasterizer(mesh.clone())
    zbuf = fragments.zbuf
    pf = fragments.pix_to_face


    PF.append(pf.unsqueeze(1))
    X.append(torch.cat((images, zbuf), dim=3).unsqueeze(1))

X = torch.cat(X, dim=1)
PF = torch.cat(PF, dim=1)      


cam = FoVPerspectiveCameras(R=R, T=T)



#point_mouth = Pointclouds(points=  [scale]) #, textures = region_id)
# #mesh = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0))
# # mesh.offset_verts_(-center)
# # mesh.scale_verts_((1.0 / float(scale)))

# # R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
# # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)


fig = plot_scene({
    "subplot1": {
        "mouth" : mesh,
        "cam" : cam
    }
})
fig.show()

