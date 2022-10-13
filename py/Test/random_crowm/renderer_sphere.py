
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
from pytorch3d.vis.plotly_vis import plot_scene


import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'..','..')
sys.path.append(mymodule_dir)
import ManageClass  as MC
from ALIDDM_utils import GenPhongRenderer, MeanScale
from utils import GetColorArray, ComputeNormals

#TODO : use linspace to create sphere not ico_sphere



def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()





def isRotationMatrix(M):
    M = M.numpy()
    tag = False
    I= np.identity(M.shape[0])
    if np.all(np.round_(np.matmul(M,M.T),decimals=5) == I) and (abs(np.linalg.det(M)) == 1 ): 
        tag = True
    return tag


def removerNotRotationMatrix(R,T):
    list_remove=[]
    for index in range(R.size()[0]):
        if not isRotationMatrix(R[index]):
            list_remove.append(index)

    for nb_remove , index_remove in enumerate(list_remove):
        R = torch.cat((R[:index_remove-nb_remove],R[index_remove-nb_remove+1:]),0)
        T = torch.cat((T[:index_remove-nb_remove],T[index_remove-nb_remove+1:]),0)
    return R,T

def surf2image(surf,index):

    image_size =224
    blur_radius = 0
    faces_per_pixel = 1
    device1 = torch.device('cuda:1')
    T = ico_sphere(1).verts_packed() * float(1.1)
    # T = torch.nn.functional.normalize(T)
    T.to(device1)
    T = T[index]


    R = look_at_rotation(T)
 
    # print(isRotationMatrix(R[index]))
    R.to(device1)
    if isRotationMatrix(R):
        return 1
    R,T = removerNotRotationMatrix(R,T)
    sphere = Pointclouds(points=[T])
    # a =R.transpose(1, 2)
    # T = -torch.bmm(a, T[:,:,None])
    # T= torch.squeeze(T).to(device1)
    number_view = R.size()[0]

    # cameras = FoVPerspectiveCameras(T=torch.unsqueeze(T[index],dim=0), R=torch.unsqueeze(R[index],dim=0),device=device1)
    cameras = FoVPerspectiveCameras(T=T, R=R,device=device1)

    phong_renderer,mask_renderer = GenPhongRenderer(image_size,blur_radius,faces_per_pixel,device=device1)

    verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()),dtype= float32)
    faces = tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    color_normals = ToTensor(dtype=torch.float32, device = device1)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    print(color_normals.size())

    textures = TexturesVertex(verts_features=color_normals[None,:,:])
    mesh = Meshes(verts=torch.unsqueeze(verts, dim=0), faces=torch.unsqueeze(faces, dim=0),textures=textures).to(device1)
    # meshes = mesh.extend(number_view).to(device1)



    # # target_image = mask_renderer(meshes_world = meshes, cameras = cameras)
    lights = PointLights(device=device1 , location=[[0.0,0.0,2.0]])
    camera = FoVPerspectiveCameras(device = device1, R=R[None,1,...],T=T[None,1,...])
    raster_setting = RasterizationSettings(image_size=image_size,blur_radius=blur_radius,faces_per_pixel=faces_per_pixel)
    rasterizer =MeshRasterizer(cameras=camera,raster_settings=raster_setting)
    shader=SoftPhongShader(device=device1,cameras=cameras,lights=lights)


    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )

    # image = renderer(meshes)
    image = mask_renderer(meshes_world = mesh)
    return image,sphere, mesh

path = '/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Test/random_crowm/T3_17_L_segmented.vtk'
image_size =224
blur_radius = 0
faces_per_pixel = 1
device1 = torch.device('cuda:1')

surf = ReadSurf(path)
surf_property = "PredictedID"
randonteeth = MC.RandomPickTeethTransform(surf_property=surf_property,unique_ids=[22,23,24,25,26])

surf = randonteeth(surf)
surf = ComputeNormals(surf) 

img_lst = torch.empty((0)).to(device1)

image, sphere, mesh = surf2image(surf)




for i in range(42):
    print(i)
    image = surf2image(surf)

    # print(image.size())
    # plt.figure(figsize=(10,10))
    # plt.imshow(image[0,...,:3].cpu().numpy())
    # plt.show()


fig = plot_scene({
    "subplot1": {
        "sphere": sphere,
        "mounth": mesh
    }
})
fig.show()

print(image[0,...,:3] == image[1,...,:3])
print(image.size())
image_grid(image.cpu().numpy(),rows=6,cols=7,rgb=True)
plt.show()

