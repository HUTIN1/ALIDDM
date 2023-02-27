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
TexturesVertex, FoVPerspectiveCameras, look_at_rotation
)

from pytorch_lightning.callbacks import ModelCheckpoint
from monai.transforms import Compose


from landmark_dataset import TeethDatasetLm
from landmark_net import MonaiUNetHRes
from ManageClass import RandomPickTeethTransform, PickLandmarkTransform, MyCompose, RandomRotation

from ALIDDM_utils import image_grid,removeversionfolder
import matplotlib.pyplot as plt
# from azureml.core.run import Run
# run = Run.get_context()
from utils import WriteSurf
from torch import tensor
from PIL import Image
import cv2
import utils


def ListToMesh(list,radius=0.05):
    list_verts =[]
    list_faces = []
    for point in list:
        sphere = ico_sphere(2)
        list_verts.append(sphere.verts_packed()*radius+tensor(point).unsqueeze(0).unsqueeze(0))
        list_faces.append(sphere.faces_list()[0].unsqueeze(0))


    list_verts = torch.cat(list_verts,dim=0)
    list_faces = torch.cat(list_faces,dim=0)
    mesh = Meshes(verts=list_verts,faces=list_faces)

    return mesh


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
    # train_transfrom = MyCompose([PickLandmarkTransform(args.landmark,args.property)])
    radius = 1.6
    model = MonaiUNetHRes(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples,radius=radius)
    train_ds  = TeethDatasetLm(mount_point = args.mount_point, df = df_train,surf_property = args.property,transform =train_transfrom,landmark=args.landmark ,test=True)
    dataloader = DataLoader(train_ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')

    print("after get item")
    model.to(device)


    

    iterdata= iter(dataloader)
    data = next(iterdata)

    V, F, CN, CL = data
    # V = V.squeeze(0)
    # F = F.squeeze(0) 
    CL = CL.squeeze(0)

    print('V.size()',V.size())
    print('F.size()',F.size())
    print('CL.shape',CL.shape)


    ico_verts, ico_faces = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=1))
    for idx, v in enumerate(ico_verts):
        if (torch.abs(torch.sum(v)) == radius):
            ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))
    # sphere =  Pointclouds(points=[sphere_verts])


    texture = TexturesVertex(CL)

    mesh = Meshes(verts=V,faces=F,textures=texture)
    V = V.to(device)
    F = F.to(device)
    CL = CL.to(device)
    CN = CN.to(device)
    X, PF = model.render(V, F, CL)
    R=[]
    T = []

    for camera_position in ico_verts:

        camera_position = camera_position.unsqueeze(0).to(device)

        r = look_at_rotation(camera_position).to(device)  # (1, 3, 3)
        t = -torch.bmm(r.transpose(1, 2), camera_position[:,:,None])[:, :, 0].to(device)   # (1, 3)
        R.append(r)
        T.append(t)

    R = torch.cat(R,dim=0)
    T = torch.cat(T,dim=0)
    cam = FoVPerspectiveCameras(R=R, T=T)
    fig = plot_scene({
    "subplot1": {
        "mouth" : mesh,
        'sphere':ListToMesh(ico_verts.tolist()),
        'cam' : cam
    }
    })
    fig.show()
    print("fait")


    # image_grid(y[...,:3].cpu().numpy(),rows=4,cols=3)
    X = X.permute(1,0,3,4,2)
    print(X.size())
    image_grid(X[...,:3].cpu().numpy(),rows=3,cols=4)
    plt.show()
    # print('unique',torch.unique(X))
    # for i in range(X.size()[0]):
    #     imname = f'/home/luciacev/Desktop/Project/ALIDDM/figure/test{i}.jpeg'
    #     image = X[i,0,...,:3]
    #     mini = torch.min(image)
    #     maxi = torch.max(image)
    #     new_image = torch.tensor((image-mini)/(maxi-mini)*255,dtype=torch.int)
    #     print(torch.unique(new_image))
    #     cv2.imwrite(imname,new_image.cpu().numpy())
    #     # plt.plot(X[i,0,...,:3].cpu().numpy())
    #     # plt.savefig(imname)





    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/train_LL1CB.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/val_LL1CB.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/test_LL1CB.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Test/random_rotation")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/data_base")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)    
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

