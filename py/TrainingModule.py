
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
    def __init__(self,args,model,render):
        super().__init__()
        self.lr = args.learning_rate

        
        self.model=model
        self.render = render


        self.loss_function = DiceCELoss(to_onehot_y=True,softmax=True)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.__model.parameters(), self.lr)

        return optimizer



    def forward(self,inputs):

        
        outputs = self.model(inputs)

        return outputs



    def training_step(self,batch, batch_idx):
        inputs , target = self.render(batch)

    
        outputs = self(inputs)
        # PlotAgentViews(outputs.detach().unsqueeze(0).cpu())
        loss = self.loss_function(outputs,target)


        self.log('train_loss',loss)

        return loss



    def validation_step(self,batch,batch_idx):

            inputs, target = self.render()

            outputs = self(inputs)

            loss = self.loss_function(outputs,target)


 


