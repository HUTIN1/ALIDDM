
from re import M
from ssl import VERIFY_X509_TRUSTED_FIRST
from utils_crown import ReadSurf
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (look_at_rotation, FoVPerspectiveCameras, RasterizationSettings,
 MeshRenderer, MeshRasterizer, SoftPhongShader,PointLights,TexturesVertex,AmbientLights,HardPhongShader)
import torch
from torch import device, tensor, float32
from pytorch3d.structures import Meshes, Pointclouds
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import ToTensor
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
import plotly.graph_objects as go
import plotly.express as px 
import pandas as pd
import json
from vtk import vtkMatrix4x4, vtkMatrix3x3
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'..','..')
sys.path.append(mymodule_dir)
import ManageClass  as MC
from ALIDDM_utils import  MeanScale, isRotationMatrix, image_grid, ALIIOSRendering, numberTooth2Landmark
from utils import GetColorArray, ComputeNormals, RandomRotation, GetTransform



#TODO : ask juan about size of image torch.Size([32, 1, 224, 224, 1]),. why is not rgb?

dico = {0:[0,0,0],15:[10,0,0],16:[50,0,0],17:[100,0,0],18:[150,0,0],19:[200,0,0],20:[250,0,0],21:[0,50,0],22:[0,100,0],23:[0,150,0],24:[0,200,0],25:[0,250,0],
26:[0,0,50],27:[0,0,100],28:[0,0,150],29:[0,0,200],30:[0,0,250],31:[50,100,200],32:[200,100,50],33:[50,50,50]}
path = '/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Test/random_crowm/T3_17_L_segmented.vtk'
path_ld ='/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Test/random_crowm/T3_17_L.json'
image_size =224
blur_radius = 0
faces_per_pixel = 1
device1 = torch.device('cuda:1')

surf = ReadSurf(path)
surf_property = "PredictedID"
randonteeth = MC.RandomPickTeethTransform(surf_property=surf_property,unique_ids=[22])

mean, scale ,surf = randonteeth(surf,MeanScale=True)
surf, angle ,vectorrotation = RandomRotation(surf)

surf = ComputeNormals(surf) 

position_camera = ico_sphere(1).verts_packed() * float(1.1)
# sphere = Pointclouds(points=[T])

verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()),dtype= float32)
faces = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
region_id = ToTensor(dtype=torch.int64)(vtk_to_numpy(surf.GetPointData().GetScalars(surf_property)))

color_normals = ToTensor(dtype=torch.float32, device = device1)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
textures_normal = TexturesVertex(verts_features=color_normals[None,:,:])

print("color_normals[None,:,:].size()",color_normals[None,:,:].size())
print("region_id.size()",region_id.size())
segmentation = torch.empty((0))
for i in range(region_id.size()[0]):

    segmentation = torch.cat((segmentation,tensor([dico[int(region_id[i].numpy())]])),dim=0)
segmentation = segmentation.unsqueeze(0)
print(segmentation.size())
texture_segmentation = TexturesVertex(verts_features=segmentation.to(device1))

mesh_normal = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=textures_normal).to(device1)
mesh_segmentation = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=texture_segmentation).to(device1)



img_lst = torch.empty((0)).to(device1)
# for i in range(42):
#     image = surf2image(mesh,T,i)
#     if not image is None:
#             inputs = torch.cat((inputs,image.to(device1)),dim=0) #[num_im*batch,channels,size,size]
image_size =224
blur_radius = 0
faces_per_pixel = 1
a = ALIIOSRendering(image_size,blur_radius,faces_per_pixel,position_camera = position_camera, device=device1)

faces_pid0 = faces[:,0:1]
surf_point_data = surf.GetPointData().GetScalars(surf_property)

surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

surf_point_data_faces[surf_point_data_faces==-1] = 33 
YF = surf_point_data_faces.to(device1)



# for i in range(42):
#     image = surf2image(mesh,T,1)
#     if not image is None:
        # img_lst = torch.cat((img_lst,image.unsqueeze(0)),dim=0)
index_image =[]

X = torch.empty((0)).to(device1)
PF = torch.empty((0)).to(device1)
T2 = torch.empty((0)).to(device1)
R2 = torch.empty((0)).to(device1)
for ps in position_camera:

            ps = ps.unsqueeze(0)
            ps= ps.to(device1)

            R = look_at_rotation(ps)  # (1, 3, 3)
            if not isRotationMatrix(R[0]): #Some of matrix rotation isnot matrix rotation
                    continue
            R = R.to(device1)
            T = -torch.bmm(R.transpose(1, 2), ps[:,:,None])[:, :, 0].to(device1)  # (1, 3)

            images = renderer(meshes_world=mesh_normal.clone().to(device1), R=R, T=T)
        
            fragments = renderer.rasterizer(mesh_normal.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf

            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)

            
            X = torch.cat((X,images.unsqueeze(0)),dim=0) # 4 channel = 3 rgb (normal) + deptmap
            PF= torch.cat((PF,pix_to_face.unsqueeze(0)),dim=0)
            T2 = torch.cat((T2,T),dim=0)
            R2 = torch.cat((R2,R),dim=0)
PF = PF.to(torch.int64)
YF = YF.to(torch.int64)
print("YF.size()",YF.size())
# PF = PF.permute(0,4,1,2,3)
# PF = PF.squeeze(1)
# print("YF.size()",YF.size())
y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
print("X.size()", X.size())
print("PF.size()",PF.size())
print("y.size()",y.size())

cameras = FoVPerspectiveCameras(device = device1,R=R2,T=T2)

# for i in range(42):
#     print(i)
#     image = surf2image(surf)

    # print(image.size())
    # plt.figure(figsize=(10,10))
    # plt.imshow(image[0,...,:3].cpu().numpy())
    # plt.show()


fig = plot_scene({
    "subplot1": {
        "mounth": mesh_segmentation,
        "camera": cameras
    }}
)


fig.show()


# # print(image[0,...,:3] == image[1,...,:3])
# # print(image.size())
# image_grid(output_display[...,0].cpu().numpy(),rows=6,cols=7)
# image_grid(output_display[...,1].cpu().numpy(),rows=6,cols=7)
# image_grid(output_display[...,2].cpu().numpy(),rows=6,cols=7)
# # # plt.title('images with normal vector')
# image_grid(X[...,:3].cpu().numpy(),rows=6,cols=7)
# image_grid(y[...].cpu().numpy(),rows=6,cols=7)
# # # # plt.title("images with depth map")
# plt.show()

# # print(images[0,:,:,:3].size())
# # plt.imshow(images[0,...,:3].cpu().numpy())
# # plt.show()

for i in range(y.size()[0]) :
    print("y.unique()",y[i,0,...].unique())
    plt.figure()
    plt.imshow(X[i,0,...,:3].cpu().numpy())
    plt.figure()
    plt.imshow(y[i,0,...].cpu().numpy(),"flag")
    plt.show()

# # for i, image in enumerate(img_lst) :
# #     print("image ",index_image[i])
# #     plt.imshow(img_lst[i,0,...,4].cpu().numpy())
# #     plt.show()