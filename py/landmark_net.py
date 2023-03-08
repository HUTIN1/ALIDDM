import math
import numpy as np 

import torch
from torch import Tensor, nn

import torchvision
from torchvision import models
from torchvision import transforms
import torchmetrics
# from shader import MaskRenderer
import utils

import monai
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas, blending
)
from pytorch3d.structures import Meshes

import pytorch_lightning as pl
from monai.losses import DiceCELoss

class TimeDistributed(nn.Module):
    def __init__(self, module,prediction=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.prediction = prediction
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output









class MonaiUNetHRes(pl.LightningModule):
    def __init__(self, args = None, out_channels=2, in_channels = 4,class_weights=None, image_size=320, radius=1.2, subdivision_level=1, train_sphere_samples=4,prediction=False):

        super(MonaiUNetHRes, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args
        self.image_size = image_size
        
        self.out_channels = out_channels
        self.class_weights = None
        if(class_weights is not None):
            self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy()
        
        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=in_channels,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
            out_channels=out_channels, 
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.model = TimeDistributed(unet)

        ico_verts, ico_faces = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            # if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))

        
        # self.register_buffer("ico_verts", ico_verts)
        self.ico_verts = ico_verts

        self.renderer = self.setup_render()

    def setup_render(self):
        # cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90, device= self.device) # Initialize a perspective camera.
        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(        
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=200000
        )

        # lights = PointLights(device = self.device) # light in front of the object. 
        lights = AmbientLights()

        rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            )


        phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=cameras, lights=lights)
        )
    
        return phong_renderer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):

        V, F, CN = x

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        
        X, PF = self.render(V, F, CN)
        x = self.model(X)
        
        return x, X, PF

    def render(self, V, F, CN):

        textures_normal = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures_normal)

        PF = []
        X = []

        for camera_position in self.ico_verts:

            camera_position = camera_position.unsqueeze(0).to(self.device)

            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)
            fragments = self.renderer.rasterizer(meshes.clone())
            zbuf = fragments.zbuf
            pf = fragments.pix_to_face


            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)

            pf = pf.permute(0,3,1,2)

            PF.append(pf.unsqueeze(1))
            X.append(images.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)      


        return X, PF

    def training_step(self, train_batch, batch_idx):

        V, F, CN, LF = train_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        LF = LF.to(self.device, non_blocking=True)

        x, X, PF = self((V, F, CN))

        y = torch.take(LF, PF)*(PF>=0)

        x = x.permute(0,2,1,3,4)
        y = y.permute(0,2,1,3,4)
        print('x.shape',x.shape)
        print('y.shape',y.shape)

        loss = self.loss(x, y)

        batch_size = V.shape[0]

        self.log('train_loss', loss, batch_size=batch_size)
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss



    def validation_step(self, val_batch, batch_idx):
        V, F, CN, LF = val_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        LF = LF.to(self.device, non_blocking=True)

        x, X, PF = self((V, F, CN))


        y = torch.take(LF, PF)*(PF>=0)


        x = x.permute(0,2,1,3,4)
        y = y.permute(0,2,1,3,4)      
        loss = self.loss(x, y)
        
        batch_size = V.shape[0]
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)

    def test_step(self, test_batch, batch_idx):

        V, F, CN, LF = test_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        LF = LF.to(self.device, non_blocking=True)

        x, X, PF = self((V, F, CN))

        y = torch.take(LF, PF)*(PF>=0)

        x = x.permute(0,2,1,3,4)
        y = y.permute(0,2,1,3,4) 
        loss = self.loss(x, y)

        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))        

        return {'test_loss': loss, 'test_correct': self.accuracy}



# class MonaiUNet(pl.LightningModule):
#     def __init__(self, args = None, out_channels=3, in_channels = 5,class_weights=None, image_size=320, radius=1.35, subdivision_level=1, train_sphere_samples=4):

#         super(MonaiUNet, self).__init__()        
        
#         self.save_hyperparameters()        
#         self.args = args
        
#         self.out_channels = out_channels
#         self.class_weights = None
#         if(class_weights is not None):
#             self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
#         self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
#         self.accuracy = torchmetrics.Accuracy()
        
#         unet = monai.networks.nets.UNet(
#             spatial_dims=2,
#             in_channels=in_channels,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
#             out_channels=out_channels, 
#             channels=(16, 32, 64, 128, 256),
#             strides=(2, 2, 2, 2),
#             num_res_units=2,
#         )
#         self.model = TimeDistributed(unet)

#         ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
#         ico_verts = ico_verts.to(torch.float32)

#         for idx, v in enumerate(ico_verts):
#             if (torch.abs(torch.sum(v)) == radius):
#                 ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))

        
#         self.register_buffer("ico_verts", ico_verts)

#         self.phong_renderer, self.mask_renderer = self.setuprender()
#     def setuprender(self):
#         cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90) # Initialize a perspective camera.

#         raster_settings = RasterizationSettings(        
#             image_size=self.image_size, 
#             blur_radius=self.blur_radius, 
#             faces_per_pixel=self.faces_per_pixel, 
#         )

#         lights = PointLights() # light in front of the object. 

#         rasterizer = MeshRasterizer(
#                 cameras=cameras, 
#                 raster_settings=raster_settings
#             )

#         b = blending.BlendParams(background_color=(0,0,0))
#         phong_renderer = MeshRenderer(
#             rasterizer=rasterizer,
#             shader=HardPhongShader( cameras=cameras, lights=lights,blend_params=b)
#         )
#         mask_renderer = MeshRenderer(
#             rasterizer=rasterizer,
#             shader=MaskRenderer(cameras=cameras, lights=lights,blend_params=b)
#         )
#         return phong_renderer,mask_renderer

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
#         return optimizer

#     def to(self, device=None):
#         self.renderer = self.renderer.to(device)
#         return super().to(device)

#     def forward(self, x):

#         V, F, CN, CL = x
        
#         X, Y = self.render(V, F, CN,CL)
#         x = self.model(X)
        
#         return x, X, Y

#     def render(self, V, F, CN,CL):

#         textures_normal = TexturesVertex(verts_features=CN)
#         meshes = Meshes(verts=V, faces=F, textures=textures_normal)        

#         textures_landnark = TexturesVertex(verts_features=CL)
#         meshes = Meshes(verts=V, faces=F, textures=textures_landnark) 

#         X = torch.empty((0)).to(self.device)
#         Y = torch.empty((0)).to(self.device)
#         for camera_position in self.ico_verts:

#             camera_position = camera_position.unsqueeze(0)

#             R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
#             T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

#             images_phong_renderer = self.phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
#             fragments = self.renderer.rasterizer(meshes.clone())
#             zbuf = fragments.zbuf
#             x = torch.cat((images_phong_renderer, zbuf), dim=3)
#             X = torch.cat((X,x.unsqueeze(0)),dim=0)


#             image_mask_renderer = self.mask_renderer(meshes_world=meshes.clone(), R=R, T=T)  
        

#             y = image_mask_renderer[:,:,:,:-1]

#             # yd = torch.where(y[:,:,:,:]<=seuil,0.,0.)
#             yr = torch.where(y[:,:,:,0]>1,1.,0.).unsqueeze(-1)
#             yg = torch.where(y[:,:,:,1]>1,2.,0.).unsqueeze(-1)
#             yb = torch.where(y[:,:,:,2]>1,3.,0.).unsqueeze(-1)

#             y = ( yr + yg + yb).to(torch.float32)





#             Y = torch.cat((Y,y.unsqueeze(0)),dim=0)
        
#         X = X.permute(1,0,4,2,3) # size of img_lst (batch size , number image, channel, image_size, image_size)
#         Y = Y.permute(1,0,4,2,3)       

#         return X, Y

#     def training_step(self, train_batch, batch_idx):

#         V, F, CN, CL, YF= train_batch

#         V = V.to(self.device, non_blocking=True)
#         F = F.to(self.device, non_blocking=True)
#         YF = YF.to(self.device, non_blocking=True)
#         CN = CN.to(self.device, non_blocking=True).to(torch.float32)
#         CL = CL.to(self.device, non_blocking=True).to(torch.float32)

#         x, X, y = self((V, F, CN,CL))

       

#         x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, H, W -> batch, channels, time, H, W
#         y = y.permute(0, 2, 1, 3, 4)
            
#         loss = self.loss(x, y)

#         batch_size = V.shape[0]
#         self.log('train_loss', loss, batch_size=batch_size)
#         self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
#         self.log("train_acc", self.accuracy, batch_size=batch_size)

#         return loss



#     def validation_step(self, val_batch, batch_idx):
#         V, F, YF, CN = val_batch

#         V = V.to(self.device, non_blocking=True)
#         F = F.to(self.device, non_blocking=True)
#         YF = YF.to(self.device, non_blocking=True)
#         CN = CN.to(self.device, non_blocking=True).to(torch.float32)

#         x, X, PF = self((V, F, CN))

#         y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)

#         x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, H, W -> batch, channels, time, H, W
#         y = y.permute(0, 2, 1, 3, 4)
            
#         loss = self.loss(x, y)
        
#         batch_size = V.shape[0]
#         self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
#         self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)
#         self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)

#     def test_step(self, batch, batch_idx):

#         V, F, YF, CN = val_batch

#         x, X, PF = self(V, F, CN)
#         y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        
#         x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, h, w -> batch, channels, time, h, w
#         y = y.permute(0, 2, 1, 3, 4) 

#         loss = self.loss(x, y)

#         self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))        

#         return {'test_loss': loss, 'test_correct': self.accuracy}