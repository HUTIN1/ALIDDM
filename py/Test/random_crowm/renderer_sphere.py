
from re import M
from ssl import VERIFY_X509_TRUSTED_FIRST
from utils_crown import ReadSurf
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (look_at_rotation, FoVPerspectiveCameras, RasterizationSettings,
 MeshRenderer, MeshRasterizer, SoftPhongShader,PointLights,TexturesVertex)
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

import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'..','..')
sys.path.append(mymodule_dir)
import ManageClass  as MC
from ALIDDM_utils import  MeanScale, isRotationMatrix, image_grid, ALIIOSRendering
from utils import GetColorArray, ComputeNormals

#TODO : use linspace to create sphere not ico_sphere


def Downscale(pos_center,mean_arr,scale_factor):
    landmarks_position = (pos_center - mean_arr) * scale_factor
    return landmarks_position

def get_landmarks_position(path, mean_arr, scale_factor, angle=False, vector=None):
       
        data = json.load(open(os.path.join(path)))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']

        lst_lm =  ['LL7O', 'LL7MB', 'LL7DB', 'LL6O', 'LL6MB', 'LL6DB', 'LL5O', 'LL5MB', 'LL5DB']
        landmarks_position = np.zeros([len(lst_lm), 3])
        # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
        for landmark in landmarks_lst:
            label = landmark["label"]
            if label in lst_lm:
                landmarks_position[lst_lm.index(label)] = Downscale(landmark["position"],mean_arr,scale_factor)

        landmarks_pos = np.array([np.append(pos,1) for pos in landmarks_position])
        # if angle:
        #     transform = GetTransform(angle, vector)
        #     transform_matrix = arrayFromVTKMatrix(transform.GetMatrix())
        #     landmarks_pos = np.matmul(transform_matrix,landmarks_pos.T).T
        return landmarks_pos[:, 0:3]

def redpath(vertex,pos_landmark):
    print(pos_landmark)
    radius = 0.02
    texture = torch.zeros_like(vertex.unsqueeze(0))
    print(texture.size())
    distance = torch.dist(pos_landmark,vertex,p=2)
    print(distance.size())
    index_pos_land = torch.nonzero((distance<radius),as_tuple=True)
    for index in index_pos_land:
        texture[0,index,0]=255
    return texture


path = '/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Test/random_crowm/T3_17_L_segmented.vtk'
path_ld ='/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Test/random_crowm/T3_17_L.json'
image_size =224
blur_radius = 0
faces_per_pixel = 1
device1 = torch.device('cuda:1')

surf = ReadSurf(path)
surf_property = "PredictedID"
randonteeth = MC.RandomPickTeethTransform(surf_property=surf_property,unique_ids=[22,23,24,25,26])

mean, scale ,surf = randonteeth(surf,MeanScale=True)

surf = ComputeNormals(surf) 
landmark_pos = get_landmarks_position(path_ld, mean, 1/scale, angle=False, vector=None)

img_lst = torch.empty((0)).to(device1)

cloud_landmark = Pointclouds([tensor(landmark_pos)])
position_camera = ico_sphere(1).verts_packed() * float(1.1)
# sphere = Pointclouds(points=[T])

verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()),dtype= float32)
faces = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
texture = redpath(verts,tensor(landmark_pos[0]).unsqueeze(0).unsqueeze(0))

# color_normals = ToTensor(dtype=torch.float32, device = device1)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
# textures = TexturesVertex(verts_features=color_normals[None,:,:])
textures=TexturesVertex(verts_features=texture)
mesh = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=textures).to(device1)


# img_lst = torch.empty((0)).to(device1)
# # for i in range(42):
# #     image = surf2image(mesh,T,i)
# #     if not image is None:
# #             inputs = torch.cat((inputs,image.to(device1)),dim=0) #[num_im*batch,channels,size,size]
# image_size =224
# blur_radius = 0
# faces_per_pixel = 1
# phong_renderer,mask_renderer = GenPhongRenderer(image_size,blur_radius,faces_per_pixel,device=device1)
# seuil =1
# renderer = mask_renderer

# # for i in range(42):
# #     image = surf2image(mesh,T,1)
# #     if not image is None:
#         # img_lst = torch.cat((img_lst,image.unsqueeze(0)),dim=0)
# index_image =[]

# img_lst = torch.empty((0)).to(device1)

# T2 = torch.empty((0)).to(device1)
# R2 = torch.empty((0)).to(device1)
# for i , pc in enumerate(position_camera):
#             pc = pc.to(device)
#             pc = pc.unsqueeze(0)
#             # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
#             R = look_at_rotation(pc)  # (1, 3, 3)
#             if not isRotationMatrix(R[0]): #Some of matrix rotation isnot matrix rotation
#                 continue
#             R = R.to(device1)
#             T = -torch.bmm(R.transpose(1, 2).to(device1), pc[:,:,None].to(device))[:, :, 0].to(device)  # (1, 3)

#             images = renderer(meshes_world=mesh.clone().to(device), R=R, T=T)
#             y = images[:,:,:,:-1]

#             # yd = torch.where(y[:,:,:,:]<=seuil,0.,0.)
#             yr = torch.where(y[:,:,:,0]>seuil,1.,0.).unsqueeze(-1)
#             yg = torch.where(y[:,:,:,1]>seuil,2.,0.).unsqueeze(-1)
#             yb = torch.where(y[:,:,:,2]>seuil,3.,0.).unsqueeze(-1)

#             y = ( yr + yg + yb).to(torch.float32)

            
            
#             index_image.append(i)



#             img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
#             T2 = torch.cat((T2,pc),dim=0)
#             R2 = torch.cat((R2,R),dim=0)



# cameras = FoVPerspectiveCameras(device = device1,R=R2,T=T2)
# print('input.size()', img_lst.size())
# sphere = Pointclouds(points=[T2])
# print(verts.size())
# # for i in range(42):
# #     print(i)
# #     image = surf2image(surf)

#     # print(image.size())
#     # plt.figure(figsize=(10,10))
#     # plt.imshow(image[0,...,:3].cpu().numpy())
#     # plt.show()


fig = plot_scene({
    "subplot1": {
        "mounth": mesh,
        'landmark': cloud_landmark
    }}
)


fig.show()


# # print(image[0,...,:3] == image[1,...,:3])
# # print(image.size())
# image_grid(img_lst[...,:3].cpu().numpy(),rows=6,cols=7,rgb=True)
# # # plt.title('images with normal vector')
# # image_grid(img_lst.cpu().numpy(),rows=3,cols=3,rgb=False)
# # # plt.title("images with depth map")
# plt.show()

# # print(images[0,:,:,:3].size())
# # plt.imshow(images[0,...,:3].cpu().numpy())
# # plt.show()

# # for i, image in enumerate(img_lst) :
# #     print("image ",index_image[i])
# #     plt.imshow(img_lst[i,0,...,:3].cpu().numpy())
# #     plt.show()

# # for i, image in enumerate(img_lst) :
# #     print("image ",index_image[i])
# #     plt.imshow(img_lst[i,0,...,4].cpu().numpy())
# #     plt.show()