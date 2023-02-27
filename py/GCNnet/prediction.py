import argparse

import os


import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils_GCN import WriteLandmark
from module_net_GCN import GCNNet
from dataset import DatasetGCNPrecdition,  DatasetGCNSegTeethPrediction
from torch_geometric.transforms import FaceToEdge
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import  plot_scene
from pytorch3d.utils import ico_sphere

def ListToMesh(list,radius=0.05):
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
    

    model = GCNNet()

    model.load_state_dict(torch.load(args.model)['state_dict'])




    ds = DatasetGCNPrecdition(args.input,FaceToEdge(remove_faces=False))
    # ds = DatasetGCNSegTeethPrediction(args.input,landmark=args.landmark,transfrom=None)

    dataloader = DataLoader(ds, batch_size=1, num_workers =args.num_workers, pin_memory = True, persistent_workers = True )
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            data , mean , scale = batch



            data = data.to(device)
            mean = mean.to(device)
            scale = scale.to(device)

            pred_segmentation = model(data)
            


            x = pred_segmentation.argmax(dim = -1, keepdim = True).squeeze()
            # print(f'x ; {x}, size {x.shape}')


            where = torch.argwhere(x)
            # print(f'where : {where}, shape {where.shape}')
            # pos  = torch.take(data.x,where)
            pos = data.x[where].squeeze()


            # print(f'pos : {pos}, size : {pos.shape}')
            landmark_pos = torch.mean(pos,0).unsqueeze(0)
            # print(f'landmakr pos : {landmark_pos}, data : {data}')

            
            distance = torch.cdist(landmark_pos,data.x,p=2)
            minvarg = torch.argmin(distance)
            # print(f'minvarg : {minvarg}, distance {distance}')
            landmark_pos = data.x[minvarg]
            

            landmark_pos_scale = landmark_pos*scale+mean
            print(f'landmakr pos : {landmark_pos}')
            # quit()

            name = ds.getName(idx)

            dic={args.landmark:landmark_pos_scale.cpu().tolist()[0]}

            WriteLandmark(dic,os.path.join(args.out,f'{name}_{args.landmark}.json'))

            texture = torch.zeros((data.x.shape[0],data.x.shape[1]),device=device)


            texture[...,0] = torch.where(x == 0 ,0,255)
            texture[...,1] = torch.where(x == 0 ,0,255)
            texture[...,2] = torch.where(x == 0 ,0,255)
            texture = TexturesVertex(texture.unsqueeze(0))

            mesh = Meshes(verts=data.x.unsqueeze(0),faces=data.face.t().unsqueeze(0),textures=texture)

            fig = plot_scene({'subplot 1':{
                'mesh':mesh,
                'landmark':ListToMesh([landmark_pos.cpu().tolist()])
            }})

            fig.show()
            quit()






if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--input',help='path folder',type=str,default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Scan/Or/scan_Or')      
    parser.add_argument('--model', help='Model to continue training', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/GCN/model/all_teeth/gcn_loss=0.00_epoch=493_raidus=0.01.ckpt")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Scan/Or/scan_Or_json_GCN/')
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--landmark',default='LL1O')


    args = parser.parse_args()

    main(args)

