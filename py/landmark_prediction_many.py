import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from landmark_net import MonaiUNetHRes
from landmark_dataset import TeethDatasetLm
from ManageClass import IterTeeth, PickLandmarkTransform
from ALIDDM_utils import WriteLandmark
import utils

from vtk.util.numpy_support import  numpy_to_vtk



def main(args):
    
        
    mount_point = args.mount_point
    all_model = utils.search(args.model,'.ckpt')['.ckpt']
    dic_model ={}
    for mod in all_model:
        name, _ = os.path.splitext(os.path.basename(mod))
        dic_model[name] = mod


    class_weights = None
    out_channels = 2


    df =utils.search(args.input,'.vtk')['.vtk']



    device = torch.device('cuda')


    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():
        for landmark in tqdm(args.landmarks,total=len(args.landmarks)):
            if landmark in  dic_model:
                print(f'Landmark : {landmark}')
                model = MonaiUNetHRes(args, out_channels = 2, class_weights=class_weights, image_size=320, train_sphere_samples=4)

                model.load_state_dict(torch.load(dic_model[landmark])['state_dict'])

                ds = TeethDatasetLm(df,args.array_name, mount_point = args.mount_point,landmark=landmark,transform=PickLandmarkTransform(landmark,args.array_name),prediction=True)

                dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
                model.to(device)
                model.eval()



                for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

                    V, F, CN, mean, scale = batch

                    V = V.cuda(non_blocking=True)
                    F = F.cuda(non_blocking=True)
                    CN = CN.cuda(non_blocking=True).to(torch.float32)
                    mean = mean.cuda(non_blocking=True)
                    scale = scale.cuda(non_blocking= True)

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

                    dic={landmark:landmark_pos*scale+mean}

                    WriteLandmark(dic,os.path.join(args.out,f'{name}_{landmark}.json'))




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/many_prediction/scan2_Or')   
    parser.add_argument('--model', help='path folder with all model', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/many_prediction/model_organize")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/many_prediction/scan2_Or")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/many_prediction")
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="UniversalID")
    parser.add_argument('--landmarks',type=list,default=['UR1O', 'UR1MB', 'UR1DB', 'UR2O', 'UR2MB', 'UR2DB', 'UR3O', 'UR3MB', 'UR3DB', 'UR4O', 'UR4MB', 
    'UR4DB', 'UR5O', 'UR5MB', 'UR5DB', 'UR6O', 'UR6MB', 'UR6DB', 'UL1O', 'UL1MB', 'UL1DB', 'UL2O', 'UL2MB', 'UL2DB', 'UL3O', 'UL3MB', 'UL3DB', 'UL4O', 
    'UL4MB', 'UL4DB', 'UL5O', 'UL5MB', 'UL5DB', 'UL6O', 'UL6MB', 'UL6DB'])


    args = parser.parse_args()

    main(args)

