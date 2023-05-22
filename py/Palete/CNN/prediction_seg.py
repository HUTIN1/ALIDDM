import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from net import MonaiUNetHRes, MonaiUnetCosine
from dataset import TeethDatasetLm, TeethDatasetLmCoss, TeethDatasetPatch
from ManageClass import IterTeeth, PickLandmarkTransform, UnitSurfTransform
from utils import WriteLandmark, WriteSurf
import utils
# import cv2
# from ALIDDM_utils import image_grid
from vtk.util.numpy_support import  numpy_to_vtk
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import  plot_scene
from pytorch3d.utils import ico_sphere
from post_process import RemoveIslands, ErodeLabel, DilateLabel
from torchvision.transforms import GaussianBlur

import matplotlib.pyplot as plt
def ListToMesh(list,radius=0.01):
    print(f' list listtomesh {list}')
    list_verts =[]
    list_faces = []
    for point in list:
        sphere = ico_sphere(2)
        list_verts.append(sphere.verts_packed()*radius+torch.tensor(point).unsqueeze(0).unsqueeze(0))
        list_faces.append(sphere.faces_list()[0].unsqueeze(0))


    list_verts = torch.cat(list_verts,dim=0)
    list_faces = torch.cat(list_faces,dim=0)
    mesh = Meshes(verts=list_verts,faces=list_faces)

    return mesh



def main(args):
    
        
    mount_point = args.mount_point


    class_weights = None
    out_channels = 2

    model = MonaiUNetHRes(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=4, subdivision_level=2,radius=1.6)

    model.load_state_dict(torch.load(args.model)['state_dict'])

    df =utils.search(args.input,'.vtk')['.vtk']


    ds = TeethDatasetPatch(df,args.array_name, mount_point = args.mount_point,landmark=args.landmark,transform=UnitSurfTransform(),prediction=True)

    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    gauss_filter = GaussianBlur(15)

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Segment Palete')
        for idx, batch in pbar:
            name = ds.getName(idx)
            pbar.set_description(name)

            V, F, CN = batch


            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True).to(torch.float32)

            x, X, PF = model((V, F, CN))
            x = softmax(x*(PF>=0))

            # x = torch.argmax(x,dim=2)

            # x = torch.where(gauss_filter(x) > 0.7,1,0) 


            # print(f'x shape {x.shape} ,unique {torch.unique(x)}')

            # quit()

            P_faces = torch.zeros(out_channels, F.shape[1]).to(device)
            # P_faces = torch.zeros(F.shape[1]).to(device).to(torch.int64)
            V_labels_prediction = torch.zeros(V.shape[1]).to(device).to(torch.int64)

            PF = PF.squeeze()
            x = x.squeeze(0)

            # print(f'PF {PF.shape}, x {x.shape}, P_faces {P_faces.shape}')

            # print(f'PF : {PF.shape}, x : {x.shape}, P_faces : {P_faces.shape}, V_labels_prediction { V_labels_prediction.shape}')

            for pf, pred in zip(PF, x):
                P_faces[:, pf] += pred
                # P_faces[pf] += pred
            # P_faces[PF] = x

            P_faces = torch.argmax(P_faces, dim=0)

            faces_pid0 = F[0,:,0]
            # print(f'shape face pid0 {faces_pid0.shape}, P_faces : {P_faces.shape}')
            V_labels_prediction[faces_pid0] = P_faces
            

            surf = ds.getSurf(idx)

            V_labels_prediction = torch.where(V_labels_prediction >= 1, 1, 0)

            V_labels_prediction = numpy_to_vtk(V_labels_prediction.cpu().numpy())
            V_labels_prediction.SetName('Palete')
            surf.GetPointData().AddArray(V_labels_prediction)

            # WriteSurf(surf,os.path.join(args.out,f'{name}_palete.vtk'))

            #Post Process
            RemoveIslands(surf,V_labels_prediction,33,500, ignore_neg1=True)
            for label in range(2):
                RemoveIslands(surf,V_labels_prediction, label, 200, ignore_neg1=True)




            for label in range(1,2):
                DilateLabel(surf,V_labels_prediction, label, iterations=2, dilateOverTarget=False, target = None)
                ErodeLabel(surf,V_labels_prediction, label, iterations=2, target=None)

            WriteSurf(surf,os.path.join(args.out,f'{name}.vtk'))






if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/IOSReg/Aron_Meg/Original/')      
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Downloads/['L3RM', 'R3RM']epoch=951-val_loss=0.471_unetseg_butterfly_Aron_create_patch_manually.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default='/home/luciacev/Desktop/Data/IOSReg/Aron_Meg/Patched/NN_createPatch_from_patchcreatemanually/')
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/Aron/json2/")
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="Universal_ID")
    parser.add_argument('--landmark',default='L2RM')


    args = parser.parse_args()

    main(args)

