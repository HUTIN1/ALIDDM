
from monai.networks.nets import UNet

import torch
from Agent_class import *
from ALIDDM.py.Tools.ALIDDM_utils import *
from monai.data import decollate_batch
import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from torch.optim import Adam
from pytorch3d.utils import ico_sphere
from torch.utils.tensorboard import SummaryWriter


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
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


class Model(pl.LightningModule):
    def __init__(self,lst_label,agent,args):
        super().__init__()
        self.lr = args.learning_rate
        self.dir_model = args.dir_models
        self.num_classess = args.num_classes
        self.batch_size = args.batch_size
        self.csv_file = args.csv_file
        self.dir_patients = args.dir_patients
        self.surf_property = self.surf_property
        


        self.model_landmark =  UNet(
        spatial_dims=2, #depthmap 2 dimension
        in_channels=5,  #normal in rgb = 3 + segmentation +depthmap
        out_channels=4, #number label by tooth + 1 
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4
            ).to(self.device)

        unet = UNet(
        spatial_dims=2,
        in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
        out_channels=out_channels, 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

        self.model_segmentation = TimeDistributed(unet)

        self.__lst_label = lst_label
        self.loss_landmark = DiceCELoss(to_onehot_y=True,softmax=True)
        self.loss_segmentation = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.epoch_loss = 0
        self.post_true = AsDiscrete(to_onehot=self.num_classess)
        self.post_pred = AsDiscrete(argmax=True,to_onehot=self.num_classess)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.final_metric = -1
        self.best_metric = -1
        self.writer = SummaryWriter()
        self.nb_val = 0
        self.writer_image_interval =1
        self.unique_ids = []
        postion_camera =ico_sphere(1).verts_packed() * float(1.1)
        self.renderer = ALIIOSRendering(args.image_size,args.blur_radius,args.faces_per_pixel,self.device,position_camera=postion_camera)
        if args.jaw == 'L':
            for teeth in args.lst_label_l:
                self.unique_ids.append(int(teeth))
        else :
            for teeth in args.lst_label_u:
                self.unique_ids.append(int(teeth))




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.__model.parameters(), self.lr)

        return optimizer

    def reset(self):
        self.epoch_loss = 0


    def forward(self,batch):
        (mesh_normal, mesh_landmark, YF) = batch
        
        inputs =  self.renderer.renderingNormalDeptmap(mesh_normal)
    
        target =  self.renderer.renderingLandmark(mesh_landmark) #[batch,num_ima,channels,size,size]
        # PlotAgentViews(land_images.detach().unsqueeze(0).cpu())

        input_segmentation, PF = self.renderer.renderingforSegmentation(mesh_normal)

        target_segmentation = torch.take(YF, PF).to(torch.int64)*(PF >= 0)  #size = (number of image , 1, W,H,1)
        target_segmentation = target_segmentation.squeeze(1).permute(0,3,1,2) #size = (number of image , 1,W,H)

        inputs = inputs.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        input_segmentation = input_segmentation.to(dtype=torch.float32)

        return inputs , target, input_segmentation,target_segmentation




    def training_step(self,batch, batch_idx):
    
    # label = random.choice(list_label)
        # optimizer.zero_grad()

        inputs, target, input_segmentation, target_segmentation = self(batch)
    
        # optimizer.zero_grad()
        outputs_landmark = self.model_landmark(inputs)
        outputs_segmentation = self.model_segmentation(input_segmentation)
        # PlotAgentViews(outputs.detach().unsqueeze(0).cpu())
        loss = self.loss_landmark(outputs_landmark,target)
        loss = self.loss_segmentation(outputs_segmentation,target_segmentation)


        self.log('train_loss',loss)

        return jaw_loss



    def validation_step(self,batch,batch_idx):

            inputs, target = self(batch)


            outputs_pred = self.__model(inputs)
            
            val_pred_outputs_list = decollate_batch(outputs_pred)                
            val_pred_outputs_convert = [
                self.post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
            ]
            
            val_true_outputs_list = decollate_batch(target)
            val_true_outputs_convert = [
                self.post_true(val_true_outputs_tensor) for val_true_outputs_tensor in val_true_outputs_list
            ]
        
            self.dice_metric(y_pred=val_pred_outputs_convert, y=val_true_outputs_convert)
            self.final_metric += self.dice_metric.aggregate().item()
            return {"input":inputs,"y_true":target,"output_pred":outputs_pred}


    def validation_epoch_end(self,validation_step_outputs):
        metric = self.final_metric
        self.dice_metric.reset()
        if metric > self.best_metric :
            self.best_metric = metric
            torch.save(self.model.state_dict(), os.path.join(self.dir_models,f"best_metric_model.pth"))
        inputs = validation_step_outputs[0]["input"][:,:-1,:,:]
        
        val_pred = torch.empty((0)).to(self.device)
        for image in  validation_step_outputs[0]["output_pred"]:
            val_pred = torch.cat((val_pred,self.post_pred(image).unsqueeze(0).to(self.device)),dim=0)

        if self.nb_val %  self.writer_image_interval == 0:       
            self.writer.add_images("input",inputs,self.current_epoch)
            self.writer.add_images("true",validation_step_outputs[0]["y_true"], self.current_epoch)
            self.writer.add_images("output",val_pred[:,1:,:,:],self.current_epoch)
            
        self.writer.close()
        return self.best_metric


