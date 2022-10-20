
import os 
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'Tools')
sys.path.append(mymodule_dir)

from Rendering import ALIIOSRendring


class RenderTraining :
    def __init__(self,args):
       self.Rendering = ALIIOSRendering(args.image_size,args.blur_radius,args.faces_per_pixel,position_camera)

    def Landmarks(self,batch):
        mesh_normal, mesh_landmark, YF = batch
        inputs = self.Rendering.renderingNormalDeptmap(mesh_normal)
        target = self.Rendering.renderingLandmark(mesh_landmark)

        return inputs, target

    def Segmentation(self,batch):
        mesh_normal, mesh_landmark, YF = batch
        inputs, PF = self.Rendering.renderingforSegmentation(mesh_normal)
        target = torch.take(YF, PF).to(torch.int64)*(PF >= 0)  #size = (number of image , 1, W,H,1)
        target = target_segmentation.squeeze(1).permute(0,3,1,2) #size = (number of image , 1,W,H)

        return inputs, target