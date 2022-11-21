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


from seg_dataset import TeethDatasetSeg
from seg_trainning import MonaiUNetHRes
from ManageClass import RandomPickTeethTransform,RandomRotation

# from azureml.core.run import Run
# run = Run.get_context()


def pad_verts_faces(self, batch):
    V = [V for V, F, CN, CLF, YF  in batch]
    F = [F for V, F, CN, CLF, YF in batch]
    CN = [CN for V, F, CN, CLF, YF in batch]
    CLF = [CL for V, F, CN, CL ,YF in batch]
    YF = [YF for V, F, CN, CLF ,YF in batch]

    V = pad_sequence(V,batch_first=True, padding_value=0.0)
    F = pad_sequence(F,batch_first=True,padding_value=-1)
    CN = pad_sequence(CN,batch_first=True,padding_value=0.0)
    CLF = torch.cat(YF)
    YF = torch.cat(YF)
    return V, F, CN, CLF, YF

def main(args):


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
    
    mount_point = args.mount_point

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))

    # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
    class_weights = None

    train_transfrom = Compose([RandomPickTeethTransform(args.property),RandomRotation()])

    model = MonaiUNetHRes(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples)
    train_ds  = TeethDatasetSeg(mount_point = args.mount_point, df = df_train,surf_property = args.property,random=True,transform =train_transfrom )

    data = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=False, collate_fn=pad_verts_faces)

    batch1= train_ds[0]
    print("after get item")

    V, F, CN, CLF, YF =batch1

    V = V.unsqueeze(0)
    F = F.unsqueeze(0)

    # X, PF = model.render(V, F, CN, CLF)

    # X = X.permute(0,1,3,4,2)
    # X = X[0]

    print('V.size()',V.size())

    sphere_verts = ico_sphere(1).verts_packed() * float(1.01)
    sphere =  Pointclouds(points=[sphere_verts])

    mesh = Meshes(verts=V,faces=F)
    fig = plot_scene({
    "subplot1": {
        "mouth" : mesh,
        'sphere':sphere
    }
    })
    fig.show()
    print("fait")



    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/train_test.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/val_test.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/test_test.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/model_output_train")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=4)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    parser.add_argument('--property', help='label of segmentation', type=str, default="PredictedID")
    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/tensorboard')
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")




    args = parser.parse_args()

    main(args)

