
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,look_at_rotation,AmbientLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex,blending
from utils import isRotationMatrix


class ALIIOSRendering:
    def __init__(self,image_size,blur_radius,faces_per_pixel,device,position_camera):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel
        self.device = device
        self.position_camera = position_camera
        #phong renderer is to get image with rgb (normal) + segmentation + deptmap 
        self.phong_renderer , self.mask_renderer = self.setup()
        self.phong_renderer2 = self.setupRenderSegmentation()

    def setup(self):

        # cameras = FoVOrthographicCameras(znear=0.1,zfar = 10,device=device) # Initialize a ortho camera.

        cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90, device= self.device) # Initialize a perspective camera.

        raster_settings = RasterizationSettings(        
            image_size=self.image_size, 
            blur_radius=self.blur_radius, 
            faces_per_pixel=self.faces_per_pixel, 
        )

        lights = PointLights(device = self.device) # light in front of the object. 

        rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            )

        b = blending.BlendParams(background_color=(0,0,0))
        phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader( device = self.device ,cameras=cameras, lights=lights,blend_params=b)
        )
        mask_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=MaskRenderer(device = self.device ,cameras=cameras, lights=lights,blend_params=b)
        )
        return phong_renderer,mask_renderer

    def setupRenderSegmentation(self):
        cameras = FoVPerspectiveCameras(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1,
            max_faces_per_bin=200000
        )        
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        lights = AmbientLights(device=self.device)
        renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=HardPhongShader(cameras=cameras, lights=lights,device=self.device)
)
        return renderer



    def renderingNormalDeptmap(self,mesh,cam_display=False):
        """ create a few images from each point of view of self.position_camera
            return multi_image 2D with 5 channels (3 for rgb (normal) + segmentation (mesh or avoid) + deptmap)
            cam_display allow to chose if we want output image for training or display

        Args:
            mesh (_type_): mesh with normal texture
            cam_display (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        img_lst = torch.empty((0)).to(self.device)
        if cam_display :
            T2 = torch.empty((0)).to(self.device)
            R2 = torch.empty((0)).to(self.device)
        for  pc in self.position_camera:
                    pc = pc.to(self.device)
                    pc = pc.unsqueeze(0)
                    # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
                    R = look_at_rotation(pc)  # (1, 3, 3)
                    if not isRotationMatrix(R[0]): #Some of matrix rotation isnot matrix rotation
                        continue
                    R = R.to(self.device)
                    T = -torch.bmm(R.transpose(1, 2).to(self.device), pc[:,:,None].to(self.device))[:, :, 0].to(self.device)  # (1, 3)

                    images = self.phong_renderer(meshes_world=mesh.clone().to(self.device), R=R, T=T)
                    # images = images[:,:-1,:,:]

                    fragments = self.phong_renderer.rasterizer(mesh.clone().to(self.device))
                    zbuf = fragments.zbuf
                    # zbuf = zbuf.permute(0, 3, 1, 2)
                    y = torch.cat((images, zbuf), dim=3)



                    img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
                    if cam_display :
                        T2 = torch.cat((T2,pc),dim=0)
                        R2 = torch.cat((R2,R),dim=0)
        if cam_display:
            return img_lst, T2, R2          # size of img_lst (number image, 1 , image_size, image_size, 5)
        img_lst = img_lst.squeeze().permute(0,3,1,2)
        return img_lst      #size (number image , 5 , image_size, image_size)   #5 = rgb (normal) + segmentation + deptmap 


    def renderingLandmark(self,mesh,cam_display=False):
        seuil =1 
        if cam_display:
            T2 = torch.empty((0)).to(self.device)
            R2 = torch.empty((0)).to(self.device)
        for pc in self.position_camera:
                    pc = pc.to(self.device)
                    pc = pc.unsqueeze(0)
                    # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
                    R = look_at_rotation(pc)  # (1, 3, 3)
                    if not isRotationMatrix(R[0]): #Some of matrix rotation isnot matrix rotation
                        continue
                    R = R.to(self.device)
                    T = -torch.bmm(R.transpose(1, 2).to(self.device), pc[:,:,None].to(self.device))[:, :, 0].to(self.device)  # (1, 3)

                    images = self.mask_renderer(meshes_world=mesh.clone().to(self.device), R=R, T=T)

                    y = images[:,:,:,:-1]

                    # yd = torch.where(y[:,:,:,:]<=seuil,0.,0.)
                    yr = torch.where(y[:,:,:,0]>seuil,1.,0.).unsqueeze(-1)
                    yg = torch.where(y[:,:,:,1]>seuil,2.,0.).unsqueeze(-1)
                    yb = torch.where(y[:,:,:,2]>seuil,3.,0.).unsqueeze(-1)

                    y = ( yr + yg + yb).to(torch.float32)
                    
        



                    img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
                    # img_lst = torch.cat((img_lst,images),dim=0)
                    if cam_display:
                        T2 = torch.cat((T2,T),dim=0)
                        R2 = torch.cat((R2,R),dim=0)
        if cam_display :
            return img_lst, T2, R2  # size of img_lst (number image, 1 , image_size, image_size, 1)

        img_lst = img_lst.squeeze(1).permute(0,3,1,2) #remove de dimension 1, after permute  dimension 0->0, 1->2, 2->3, 3->1
        return img_lst   # size of img_lst (number image, 1 , image_size, image_size)




    def renderingforSegmentation(self,mesh,cam_display=False):
        if cam_display:
            T2 = torch.empty((0)).to(self.device)
            R2 = torch.empty((0)).to(self.device)
        X = torch.empty((0)).to(self.device)
        PF = torch.empty((0)).to(self.device)

        for ps in self.position_camera:

            ps = ps.unsqueeze(0)
            ps= ps.to(self.device)

            R = look_at_rotation(ps)  # (1, 3, 3)
            if not isRotationMatrix(R[0]): #Some of matrix rotation isnot matrix rotation
                    continue
            R = R.to(self.device)
            T = -torch.bmm(R.transpose(1, 2), ps[:,:,None])[:, :, 0].to(self.device)  # (1, 3)

            images = self.phong_renderer2(meshes_world=mesh.clone().to(self.device), R=R, T=T)
        
            fragments = self.phong_renderer2.rasterizer(mesh.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf

            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)

            if cam_display:
                X = torch.cat((X,images.unsqueeze(0)),dim=0) # 4 channel = 3 rgb (normal) + deptmap
                PF= torch.cat((PF,pix_to_face.unsqueeze(0)),dim=0)


            T2 = torch.cat((T2,T),dim=0)
            R2 = torch.cat((R2,R),dim=0)


        PF = PF.to(torch.int64)
        if cam_display:
            return X, PF , T2, R2

        X = X.squeeze(1).permute(0,3,1,2) 
        return X, PF        #X size = (number image , 4 =rgb(normal) + deptmap , W,H)
                            #PF size = (number imag ,1 ,W,H,1)
    