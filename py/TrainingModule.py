
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
from torch.utils.tensorboard import SummaryWriter





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
        


        self.__model =  UNet(
        spatial_dims=2,
        in_channels=5,
        out_channels=args.num_classes,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4
            ).to(self.device)


        self.__lst_label = lst_label
        self.__agent = agent
        self.loss_function = DiceCELoss(to_onehot_y=True,softmax=True)
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


    def forward(self,batch,label):
        (S, V, F, RI, CN, LP, MR, SF) = batch
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
                    verts=V,   
                    faces=F, 
                    textures=textures
                    ).to(self.device)
        
        self.__agent.position_agent(RI,V,label)

        images =  self.__agent.GetView(meshes) #[batch,num_ima,channels,size,size]
        
        meshes_2 = Gen_mesh_patch(S,V,F,CN,LP,label,self.device)
    
        land_images =  self.__agent.GetView(meshes_2,rend=True) #[batch,num_ima,channels,size,size]
        # PlotAgentViews(land_images.detach().unsqueeze(0).cpu())

        inputs = torch.empty((0)).to(self.device)
        y_true = torch.empty((0)).to(self.device)
        for i,batch in enumerate(images):
            inputs = torch.cat((inputs,batch.to(self.device)),dim=0) #[num_im*batch,channels,size,size]
            y_true = torch.cat((y_true,land_images[i].to(self.device)),dim=0) #[num_im*batch,channels,size,size] channels=1

        inputs = inputs.to(dtype=torch.float32)
        y_true = y_true.to(dtype=torch.float32)

        return inputs , y_true




    def training_step(self,batch, batch_idx):
        jaw_loss = 0
    
    # label = random.choice(list_label)
        # optimizer.zero_grad()

        inputs, y_true = self(batch,label)
    
        # optimizer.zero_grad()
        outputs = self.__model(inputs)
        # PlotAgentViews(outputs.detach().unsqueeze(0).cpu())
        loss = self.loss_function(outputs,y_true)
        jaw_loss += loss

        self.log('train_loss',loss)

        return jaw_loss


    def training_step_end(self,batch_parts):
        self.epoch_loss += batch_parts
        return self.epoch_loss




    def training_epoch_end(self,training_step_outputs):
        print('training_step_outputs :',training_step_outputs)
        r = training_step_outputs[0]['loss'] / len(self.__lst_label)



    def validation_step(self,batch,batch_idx):

            inputs, y_true = self(batch,label)


            outputs_pred = self.__model(inputs)
            
            val_pred_outputs_list = decollate_batch(outputs_pred)                
            val_pred_outputs_convert = [
                self.post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
            ]
            
            val_true_outputs_list = decollate_batch(y_true)
            val_true_outputs_convert = [
                self.post_true(val_true_outputs_tensor) for val_true_outputs_tensor in val_true_outputs_list
            ]
        
            self.dice_metric(y_pred=val_pred_outputs_convert, y=val_true_outputs_convert)
            self.final_metric += self.dice_metric.aggregate().item()
        return {"input":inputs,"y_true":y_true,"output_pred":outputs_pred}


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


