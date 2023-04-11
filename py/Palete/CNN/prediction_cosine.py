import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from net import MonaiUNetHRes, MonaiUnetCosine
from dataset import TeethDatasetLm, TeethDatasetLmCoss
from ManageClass import IterTeeth, PickLandmarkTransform, UnitSurfTransform
from utils import WriteLandmark
import utils
# import cv2
# from ALIDDM_utils import image_grid
from vtk.util.numpy_support import  numpy_to_vtk
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import  plot_scene
from pytorch3d.utils import ico_sphere

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

    model = MonaiUnetCosine(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=4, subdivision_level=2,radius=1.6)

    model.load_state_dict(torch.load(args.model)['state_dict'])

    df =utils.search(args.input,'.vtk')['.vtk']


    ds = TeethDatasetLmCoss(df,args.array_name, mount_point = args.mount_point,landmark=args.landmark,transform=UnitSurfTransform(),prediction=True)

    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            name = ds.getName(idx)

            V, F, CN, matrix= batch

            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True).to(torch.float32)
            matrix = matrix.cuda(non_blocking = True).to(torch.float32).squeeze()


            vector, distance = model((V, F, CN))
            # print(f'shape vector {vector.shape}, distance {distance.shape}')

            landmark_vector = torch.mul(vector,distance)
            # print(f'landmark vector {landmark_vector}')


            landmark_vector = torch.cat((landmark_vector,torch.ones((landmark_vector.shape[0],1),device=device)),dim=1).to(torch.float32)

            # print(f'cat landmark pos {landmark_vector}')
            # print(f'matrix {matrix}')

            landmark_vector = torch.matmul(torch.linalg.inv(matrix),landmark_vector.T).T
            landmark_vector = landmark_vector[...,:3]
            # print(f'after matrix landmark vector {landmark_vector}')
  

            dic ={}
            for idx , landmark in enumerate(landmark_vector):
                dic[f'{args.landmark}_{idx}'] = landmark

            # dic={args.landmark:apply_matrix.squeeze()[:3]}
            print(f'patient {name}, pos landmark {landmark_vector}')

            WriteLandmark(dic,os.path.join(args.out,f'{name}_{args.landmark}.json'))

            # print(f' vertex { V.shape}')

            # mesh = Pointclouds(V)

            # fig = plot_scene({'subplot 1':{
            #     'mesh':mesh,
            #     'landmark':ListToMesh(landmark_vector.cpu().tolist()),
            #     # 'landmark':ListToMesh([pos.squeeze(0).cpu().tolist()])
            # }})

            # fig.show()
            # quit()






if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/denise/scan/')      
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/CNN/model/['L2RM']epoch=113-val_loss=0.09_monairesnet_cosine.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/denise/json/')
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/denise/json2/")
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--landmark',default='L2RM')


    args = parser.parse_args()

    main(args)

