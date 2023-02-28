import argparse

import os


import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils_GCN import WriteLandmark
from module_net_GCN import GCNNet
from dataset import DatasetGCNSegTeethPrediction, DatasetGCN, DatasetGCNSegTeeth
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import  plot_scene
from torch_geometric.transforms import FaceToEdge
from pytorch3d.utils import ico_sphere
from typing import Union

def ListToMesh(list,radius=0.05):
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
    

    model = GCNNet()

    # model.load_state_dict(torch.load(args.model)['state_dict'])
    device = torch.device('cuda')




    ds = DatasetGCNSegTeeth(args.csv_train,landmark=args.landmark[0],transfrom=None,radius=args.radius)
    # ds = DatasetGCN(path = args.csv_train, landmark=args.landmark, transfrom=FaceToEdge(remove_faces=False),radius=args.radius)

    # dataloader = DataLoader(ds, batch_size=1, num_workers =args.num_workers, pin_memory = True, persistent_workers = True )

    data = ds[0].to(device)
    pos_landmark = [ds.getLandmark(0)]
    print(f'name :{ds.getName(0)}')
    print(f'data {data}')

    vertex = data.x[...,:3]

    texture = torch.zeros((vertex.shape[0],vertex.shape[1]),device=device)
    x = data.segmentation_labels.squeeze()
    print(f'x size {x.shape}')

    # texture[...,0] = torch.where(x == 0 ,0,torch.randint(0,255,(1,),device=device))
    # texture[...,1] = torch.where(x == 0 ,0,torch.randint(0,255,(1,),device=device))
    # texture[...,2] = torch.where(x == 0 ,0,torch.randint(0,255,(1,),device=device))

    texture[...,0] = torch.where(x == 0 ,0,255)
    texture[...,1] = torch.where(x == 0 ,0,255)
    texture[...,2] = torch.where(x == 0 ,0,255)
    # texture = TexturesVertex(texture.unsqueeze(0))

    # mesh = Meshes(verts=data.x.unsqueeze(0),faces=data.face.t().unsqueeze(0),textures=texture)

    # print(f'normal {texture.unsqueeze(0).shape}, data {data.x.unsqueeze(0).shape}')
    mesh = Pointclouds(vertex.unsqueeze(0))
    # print(f'argwhere : {torch.argwhere(data.segmentation_labels.squeeze())}')
    print(f'unique label :{torch.unique(data.segmentation_labels)}')
    points = vertex[torch.argwhere(data.segmentation_labels.squeeze())].squeeze()
    # print(f'point : {points.shape}')
    points = Pointclouds(points.unsqueeze(0))


    print(f'post landmark {pos_landmark}, mean x : {torch.mean(vertex,0)} ')


    fig = plot_scene({'subplot 1':{
        'mesh':mesh,
        # 'landmakr' : ListToMesh(pos_landmark),
        'point':points
    }})



    
    
    fig.show()







if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/train_LL1O.csv') 
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Scan/Or/scan_Or')      
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/GCN/model/args.landmark=0_epoch=499-val_loss=0.09.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Scan/Or/scan_Or_json_GCN_segteeth/')
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--radius',help='radius of landmark on mesh',default=0.1)
    parser.add_argument('--landmark',type= Union[str,list],default=['LL1O','LL2O'])


    args = parser.parse_args()

    main(args)

