import argparse

import os


import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils_GCN import WriteLandmark
from module_net_GCN import GCNNet
from dataset import DatasetGCNPrecdition,  DatasetGCNSegTeethPrediction, DatasetGCNSegPred
from torch_geometric.transforms import FaceToEdge
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import  plot_scene
from pytorch3d.utils import ico_sphere
import numpy as np
import polyscope as ps
from ManageClass import UnitSurfTransform

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
    

    model = GCNNet(num_classes=2,in_features=6)

    model.load_state_dict(torch.load(args.model)['state_dict'])




    # ds = DatasetGCNPrecdition(args.input,FaceToEdge(remove_faces=False))
    # ds = DatasetGCNSegTeethPrediction(args.input,landmark=args.landmark,transfrom=None)
    ds = DatasetGCNSegPred(args.input,UnitSurfTransform())

    dataloader = DataLoader(ds, batch_size=1, num_workers =args.num_workers, pin_memory = True, persistent_workers = True )
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            data , vertex, index, matrix = batch



            data = data.to(device)
            index = index.to(device)
            vertex = vertex.to(device).squeeze()
            matrix = matrix.to(device)

            pred_segmentation = model(data)
            


            pred = pred_segmentation.argmax(dim = -1, keepdim = True).squeeze()
            # print(f'x ; {x}, size {x.shape}')

            # print(f' unique x :{torch.unique(x)}')
            where = torch.argwhere(pred)
            # print(f'where : {where}, shape {where.shape}')

            if where.shape[0] == 0:
                print(f'dont found landmark {args.landmark}, for this patient: {ds.getName(idx)}')
                continue
            
            # pos  = torch.take(data.x,where)

            index_pos = torch.take(index,where)
            print(f'index pos {index_pos.shape}')

            pos = []
            for p in index_pos:
                pos.append(vertex[p])

            pos = torch.cat(pos,dim=0)
            print(f'pos {pos.shape}')

            if len(pos.shape) == 1 :
                pos = pos.unsqueeze(0)





            # print(f'pos : {pos}, size : {pos.shape}')
            # landmark_pos = torch.mean(pos,0).unsqueeze(0)
            # print(f'landmakr pos : {landmark_pos}, data : {data}')

            
            # distance = torch.cdist(landmark_pos,vertex,p=2)
            # minvarg = torch.argmin(distance)
            # print(f'minvarg : {minvarg}, distance {distance}')
            # landmark_pos = vertex[minvarg]
            

            # landmark_pos_scale = torch.matmul(matrix,landmark_pos)
            # print(f'landmakr pos : {landmark_pos}')
            # quit()

            name = ds.getName(idx)

            # dic={args.landmark:landmark_pos_scale.cpu().tolist()[0]}

            # WriteLandmark(dic,os.path.join(args.out,f'{name}_{args.landmark}.json'))

            # texture = torch.zeros((vertex.shape[0],vertex.shape[1]),device=device)


            # texture[...,0] = torch.where(x == 0 ,0,255)
            # texture[...,1] = torch.where(x == 0 ,0,255)
            # texture[...,2] = torch.where(x == 0 ,0,255)
            # # texture = TexturesVertex(texture.unsqueeze(0))

            # # mesh = Meshes(verts=data.x.unsqueeze(0),faces=data.face.t().unsqueeze(0),textures=texture)
            # # print(f'pos : {pos}, pos shape : {pos.shape}, pos true : {pos.shape[0] == 0}')

            # # fig = plot_scene({'subplot 1':{
            # #     'mesh':mesh,
            # #     # 'landmark':ListToMesh([pos.cpu().tolist()])
            # # }})


            points = Pointclouds(pos.unsqueeze(0))
            mesh = Pointclouds(vertex.unsqueeze(0))
            fig = plot_scene({'subplot 1':{
                'mesh':mesh,
                'point':points,
                # 'landmark':ListToMesh([pos.squeeze(0).cpu().tolist()])
            }})

            fig.show()
            quit()






if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/random_scan/scan/')      
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Model/model_test/['R3RL', 'R2RM', 'L2RM', 'L3RL', 'L3RM', 'LPR', 'RPR', 'R3RM']_radius=0.1_epoch=490-val_loss=0.065_segpalete.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Palete/random_scan/json')
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--landmark',default='L2RM')
    parser.add_argument('--radius',type=float,default=0.5)
    parser


    args = parser.parse_args()

    main(args)

