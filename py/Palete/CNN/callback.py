from pytorch_lightning.callbacks import Callback
import torchvision
import torch


class TeethNetImageLoggerLm(Callback):
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

                V, F, CN, LF = batch

                batch_size = V.shape[0]
                num_images = min(batch_size, self.num_images)

                V = V.to(pl_module.device, non_blocking=True)
                F = F.to(pl_module.device, non_blocking=True)
                LF = LF.to(pl_module.device, non_blocking=True)
                CN = CN.to(pl_module.device, non_blocking=True).to(torch.float32)



                with torch.no_grad():

                    # x, X, PF = pl_module((V[0:1], F[0:1], CN[0:1]))
                    batch = pl_module((V[0:1], F[0:1], CN[0:1]))

                    x = batch[0]
                    X = batch[1]
                    PF = batch[2]


                    Y = torch.take(LF, PF)*(PF>=0)

                    x = torch.argmax(x, dim=2, keepdim=True)

                    Y2 = torch.cat((Y,Y,Y),dim=2)
                    Yread = torch.tensor(X[:, :, 0:3, :, :].clone().detach().requires_grad_(True))
                    Yread[Y2==1] = 0

                    
                    grid_X = torchvision.utils.make_grid(X[0, :, 0:3, :, :])#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_normals', grid_X, pl_module.global_step)

                    grid_X = torchvision.utils.make_grid(X[0, :, 3, :, :].unsqueeze(1))#Grab the depth map. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_depth', grid_X, pl_module.global_step)
                    
                    grid_x = torchvision.utils.make_grid(x[0, :, :, :, :]/pl_module.out_channels)# The time dimension is on dim 1 grab only the first one
                    trainer.logger.experiment.add_image('x', grid_x, pl_module.global_step)

                    grid_y = torchvision.utils.make_grid(Y[0, :, :, :, :]/pl_module.out_channels)# The time dimension here is swapped after the permute and is on dim=2. It will grab the first image
                    trainer.logger.experiment.add_image('Y', grid_y, pl_module.global_step)

                    grid_y = torchvision.utils.make_grid(Yread[0, :, :, :, :]/pl_module.out_channels)# The time dimension here is swapped after the permute and is on dim=2. It will grab the first image
                    trainer.logger.experiment.add_image('Y + X_normal', grid_y, pl_module.global_step)