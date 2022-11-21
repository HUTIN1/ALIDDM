import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from seg_net import MonaiUNetHRes
from seg_dataset import TeethDatasetSeg
from ManageClass import IterTeeth

import utils

from vtk.util.numpy_support import  numpy_to_vtk



def main(args):
    
        
    mount_point = args.mount_point


    class_weights = None
    out_channels = 34

    model = MonaiUNetHRes(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=4,prediction=True)

    model.load_state_dict(torch.load(args.model)['state_dict'])

    df = pd.read_csv(os.path.join(mount_point, args.csv))

    ds = TeethDatasetSeg(df,"PredictedID", mount_point = args.mount_point, prediction=IterTeeth("PredictedID"))

    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            V, F, CN, CLF, YF = batch

            V = V.cuda(non_blocking=True).squeeze(0)
            F = F.cuda(non_blocking=True).squeeze(0)
            CN = CN.cuda(non_blocking=True).to(torch.float32).squeeze(0)
            CLF = CLF.cuda( non_blocking=True).squeeze(0)
            YF = YF.cuda( non_blocking=True).squeeze(0)

            P_faces = torch.zeros(out_channels, F.shape[1]).to(device)
            V_labels_prediction = torch.zeros(V.shape[1]).to(device).to(torch.int64)

            for v, f ,cn , clf in tqdm(zip(V,F,CN,CLF),total=V.shape[0]):
                v = v.unsqueeze(0)
                f = f.unsqueeze(0)
                cn = cn.unsqueeze(0)
                clf = clf.unsqueeze(0)
                x, X, PF = model((v, f, cn, clf))

                x = softmax(x*(PF>=0))
                

                PF = PF.squeeze()
                x = x.squeeze()

                for pf, pred in zip(PF, x):

                    P_faces[:,pf]+=pred


            P_faces = torch.argmax(P_faces, dim=0)

            faces_pid0 = F[0,:,0]
            V_labels_prediction[faces_pid0] = P_faces


            surf = ds.getSurf(idx)

            V_labels_prediction = numpy_to_vtk(V_labels_prediction.cpu().numpy())
            V_labels_prediction.SetName(args.array_name)
            surf.GetPointData().AddArray(V_labels_prediction)

            output_fn = os.path.join(args.out, df["surf"][idx])

            output_dir = os.path.dirname(output_fn)

            if(not os.path.exists(output_dir)):
                os.makedirs(output_dir)

            utils.Write(surf , output_fn)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--csv', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/prediction_test.csv')    
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/best_model.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/test_prediction_Seg")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth")
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="TestID")


    args = parser.parse_args()

    main(args)

