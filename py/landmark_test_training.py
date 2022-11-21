import argparse

import math
import os
import pandas as pd
import numpy as np 
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
TexturesVertex
)
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.transforms import Compose


from landmark_dataset import TeethDatasetLm
from landmark_net import MonaiUNetHRes
from ManageClass import RandomPickTeethTransform,RandomRotation, PickLandmarkTransform, MyCompose

from ALIDDM_utils import image_grid,removeversionfolder
import matplotlib.pyplot as plt
# from azureml.core.run import Run
# run = Run.get_context()



def main(args):


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
    removeversionfolder(os.path.join(args.tb_dir,args.tb_name))
    mount_point = args.mount_point

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))

    # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
    class_weights = None

    train_transfrom = MyCompose([PickLandmarkTransform(args.landmark,args.property),RandomRotation()])
    radius = 1.2
    model = MonaiUNetHRes(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples,radius=radius)
    train_ds  = TeethDatasetLm(mount_point = args.mount_point, df = df_train,surf_property = args.property,random=True,transform =train_transfrom,landmark=args.landmark )
    dataloader = DataLoader(train_ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    
    iterdata = iter(dataloader)
    device = torch.device('cuda')

    print("after get item")
    model.to(device)

    data =next(iterdata)

    
    V, F, CN, LF = data

    V = V.to(device, non_blocking=True)
    F = F.to(device, non_blocking=True)
    CN = CN.to(device, non_blocking=True).to(torch.float32)
    LF = LF.to(device, non_blocking=True)

    x, X, PF = model((V, F, CN))

    y = torch.take(LF, PF)*(PF>=0)

    x = x.permute(0,2,1,3,4)
    y = y.permute(0,2,1,3,4) 
    print('x.size()',x.size())
    print('y.size()',y.size())

    loss = model.loss(x, y)


    # sphere_verts = ico_sphere(1).verts_packed() * float(radius)
    # sphere =  Pointclouds(points=[sphere_verts])

    # texture = TexturesVertex(CL)

    # mesh = Meshes(verts=V,faces=F,textures=texture)
    # fig = plot_scene({
    # "subplot1": {
    #     "mouth" : mesh,
    #     'sphere':sphere
    # }
    # })
    # fig.show()
    # print("fait")


    # image_grid(y[...,:3].cpu().numpy(),rows=4,cols=3)
    # image_grid(X[...,:3].cpu().numpy(),rows=4,cols=3)
    # plt.show()



    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/csv/train_LL1CB.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/csv/val_LL1CB.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/csv/test_LL1CB.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/model_out")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=4)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    parser.add_argument('--property', help='label of segmentation', type=str, default="PredictedID")
    parser.add_argument('--landmark',help='name of landmark to found',default='LL1CB')
    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/tensorboard')
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")





    args = parser.parse_args()
    removeversionfolder(os.path.join(args.tb_dir,args.tb_name))
    main(args)

