import torch
from vtk.util.numpy_support import vtk_to_numpy
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (TexturesVertex, look_at_rotation, FoVPerspectiveCameras, RasterizationSettings,
MeshRasterizer, AmbientLights,MeshRenderer,HardPhongShader)
from monai.transforms import ToTensor


import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'..','..')
sys.path.append(mymodule_dir)
import ManageClass  as MC
from ALIDDM_utils import GenPhongRenderer, MeanScale,Frender
import utils


def Frender(V,F,CN):
        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(
            image_size=320, 
            blur_radius=0, 
            faces_per_pixel=1, 
            bin_size=0,
        )        
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        lights = AmbientLights()
        renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=HardPhongShader(cameras=cameras, lights=lights)
        )
        device1 = torch.device('cuda:1')
        X = []
        PF = []

        batch_size = V.shape[0]

        sphere_centers = torch.zeros([batch_size, 3]).to(torch.float32).to(device1)
        
        renderer = renderer.to(device1)

        V = V.to(device1)
        F = F.to(device1)
        CN = CN.to(device1).to(torch.float32)
        print(CN[None,:,:].size())
        print(F.size())
        print(V.size())
        textures = TexturesVertex(verts_features=CN[None,:,:])
        meshes = Meshes(verts=torch.unsqueeze(V,dim=0), faces=torch.unsqueeze(F,dim=0), textures=textures)



        ico_verts = torch.tensor([[0., 0., 1.]])

        for camera_position in ico_verts:
            
            current_cam_pos = sphere_centers + camera_position.to(device1)            

            R = look_at_rotation(current_cam_pos, device=device1)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:,:,None])[:, :, 0]   # (1, 3)

            images = renderer(meshes_world=meshes.clone(), R=R, T=T)            
            pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes.clone())

            pix_to_face = pix_to_face.permute(0, 3, 1, 2)
            images = images.permute(0, 3, 1, 2)            
            zbuf = zbuf.permute(0, 3, 1, 2)

            images = images[:,:-1,:,:] #grab RGB components only
            images = torch.cat([images, zbuf], dim=1) #append the zbuf as a channel

            X.append(images.unsqueeze(dim=1))
            PF.append(pix_to_face.unsqueeze(dim=1))
            
        return torch.cat(X, dim=1), torch.cat(PF, dim=1)


path = '/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Test/random_crowm/T3_17_L_segmented.vtk'
device1 = torch.device('cuda:1')
surf = utils.ReadSurf(path)
surf_property = "PredictedID"
surf = utils.ComputeNormals(surf)
color_normals = ToTensor(dtype=torch.float32, device = device1)(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/255.0)
verts = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
faces = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)        



faces_pid0 = faces[:,0:1]
surf_point_data = surf.GetPointData().GetScalars(surf_property)

surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

surf_point_data_faces[surf_point_data_faces==-1] = 33   
X, PF = Frender(verts,faces,color_normals)

y = torch.take(surf_point_data_faces,PF)*(PF >=0)

