import torch
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'Tools')
sys.path.append(mymodule_dir)

from Rendering import ALIIOSRendering


class RenderTraining :
    def __init__(self,args,device):
        self.device = device
        position_camera = ico_sphere(1).verts_packed() * float(1.1)
        self.Rendering = ALIIOSRendering(args.image_size,args.blur_radius,args.faces_per_pixel,position_camera = position_camera,device=device)

    def Landmarks(self,batch):
        V, F, CN, CL, YF = batch
        # print("V.size()",V.size(),"F.size()",F.size())
        texture_normal = TexturesVertex(verts_features=CN.squeeze(0))
        texture_landmarks = TexturesVertex(verts_features=CL.squeeze(1))
        mesh_normal = Meshes(verts=V, faces=F,textures=texture_normal)
        mesh_landmark = Meshes(verts=V, faces=F,textures=texture_landmarks)
        print("mesh_normal",mesh_normal)
        inputs = self.Rendering.renderingNormalDeptmap(mesh_normal)
        target = self.Rendering.renderingLandmark(mesh_landmark)

        return inputs, target

    def Segmentation(self,batch):
        V, F, CN, CL, YF = batch
        texture_normal = TexturesVertex(verts_features=CN[None,:,:])
        mesh_normal = Meshes(verts=V, faces=F,textures=texture_normal)
        YF = YF.to(self.device)
        inputs, PF = self.Rendering.renderingforSegmentation(mesh_normal)
        target = torch.take(YF, PF).to(torch.int64)*(PF >= 0)  #size = (number of image , 1, W,H,1)
        target = target.squeeze(1).permute(0,3,1,2) #size = (number of image , 1,W,H)

        return inputs, target