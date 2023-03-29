import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from net import MonaiUNetHRes, MonaiUnetCosine
from dataset import TeethDatasetLm
from ManageClass import IterTeeth, PickLandmarkTransform, UnitSurfTransform
from utils import WriteLandmark
import utils
# import cv2
# from ALIDDM_utils import image_grid
from vtk.util.numpy_support import  numpy_to_vtk
import matplotlib.pyplot as plt
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.structures import Meshes, Pointclouds

def main(args):
    
        
    mount_point = args.mount_point


    class_weights = None
    out_channels = 2

    model = MonaiUnetCosine(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=4, subdivision_level=2,radius=1.6)

    model.load_state_dict(torch.load(args.model)['state_dict'])

    df =utils.search(args.input,'.vtk')['.vtk']


    ds = TeethDatasetLm(df,args.array_name, mount_point = args.mount_point,landmark=args.landmark,transform=UnitSurfTransform(),prediction=True)

    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            V, F, CN, matrix = batch

            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True).to(torch.float32)
            matrix = matrix.cuda(non_blocking = True)

            P_faces = torch.zeros(out_channels, F.shape[1]).to(device)
            V_labels_prediction = torch.zeros(V.shape[1]).to(device).to(torch.int64)



            x, X, PF = model((V, F, CN))
            


            x = softmax(x*(PF>=0))
        


            PF = PF.squeeze()
            x = x.squeeze()

            for pf, pred in zip(PF, x):

                P_faces[:,pf]+=pred


            P_faces = torch.argmax(P_faces, dim=0)

            faces_pid0 = F[0,:,0]
            V_labels_prediction[faces_pid0] = P_faces

            V_landmark_ids = torch.argwhere(V_labels_prediction).squeeze(-1)


            landmark_pos = torch.mean(V[:,V_landmark_ids,:],1)
            name = ds.getName(idx)
            ready_pos = torch.cat((landmark_pos,torch.tensor([[1]],device=device)),dim=1).T.to(torch.float64)
            

            apply_matrix = torch.matmul(torch.linalg.inv(matrix),ready_pos).T
            dic={args.landmark:apply_matrix.squeeze()[:3]}

            WriteLandmark(dic,os.path.join(args.out,f'{name}_{args.landmark}.json'))


            fig = plot_scene({
            "subplot1": {
                "mouth" : Pointclouds(V),
                'path' : Pointclouds(V[:,V_landmark_ids,:])
            }
            })
            fig.show()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/Aron/scan/')      
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/CNN/model/['L2RM']epoch=69-val_loss=0.19_unet_densenet_depth_map_cosine.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/Aron/json/')
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/jaw_upper")
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--landmark',default='L2RM')


    args = parser.parse_args()

    main(args)

