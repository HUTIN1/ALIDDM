import numpy as np
import torch
from utils import search, ReadSurf, WriteSurf
import vtk
from icp import PrePreAso, vtkMeanTeeth, ToothNoExist
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
import os
from tqdm import tqdm
import argparse

fpath = '/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Palete/GCNnet'
sys.path.append(fpath)

from segmented_from_point import Segmentation


class Segment2D :
    def __init__(self,point1,point2,name_point1 =None , name_point2 = None) -> None:
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.a = point2[0] - point1[0]
        self.b = point2[1] - point1[1]

        self.x0 = point1[0]
        self.y0 = point1[1] 

        self.name_point1 = name_point1
        self.name_point2 = name_point2

        # print(f' a : {self.a}, b : {self.b}, x0 : {self.x0}, y0 : {self.y0}')

    def __call__(self, t) :
        x , y = self.x0 + self.a * t , self.y0 + self.b * t

        return np.array([x ,y])
    

def Bezier_bled(point1,point2,point3,pas):
    range = np.arange(0,1,pas)
    matrix_t = np.array([np.square( 1 - range) , 2*(1 - range)*range, np.square(range)]).T
    matrix_point = np.array([[point1],[point2],[point3]]).squeeze()
    # print(f'shape matrix_t {matrix_t.shape}, matrix point {matrix_point.shape}')
    return np.matmul(matrix_t,matrix_point)


def Difference(t1,t2):
    # print(f't2 : {t2}')
    # print(f't1 shape {t1.shape}')
    t1 = t1.unsqueeze(0).expand(len(t2),-1)
    t2 = t2.unsqueeze(1)
    d = torch.count_nonzero(t1 -t2,dim=-1)
    arg = torch.argwhere(d == t1.shape[1])
    dif = torch.unique(t2[arg])
    return dif

# def Neighbours_secours(arg_point,F):
#     divide_batch = 3
#     len_batch = len(arg_point)
#     arg_list = []
#     for i in range(divide_batch):
#         if int(len_batch/divide_batch)*(i+1) >= len_batch:
#             arg_point_small_batch = arg_point[int(len_batch/divide_batch)*i : -1]
#         else :
#             arg_point_small_batch = arg_point[int(len_batch/divide_batch)*i : int(len_batch/divide_batch)*(i+1)]
#         neighbours = torch.tensor([]).cuda()
#         F2 = F.unsqueeze(0).expand(len(arg_point_small_batch),-1,-1)
#         arg_point_small_batch = arg_point_small_batch.unsqueeze(1).unsqueeze(2)
#         arg_list.append(torch.argwhere((F2-arg_point_small_batch) == 0))
#     # print(f' arg {arg.shape}, arg : {arg}')
#     arg = torch.cat(arg_list,dim=0)

#     neighbours = torch.unique(F[arg[:,1],:])
#     return neighbours

def Neighbours(arg_point,F):
    neighbours = torch.tensor([]).cuda()
    F2 = F.unsqueeze(0).expand(len(arg_point),-1,-1)
    arg_point = arg_point.unsqueeze(1).unsqueeze(2)
    arg = torch.argwhere((F2-arg_point) == 0)
    # print(f' arg {arg.shape}, arg : {arg}')

    neighbours = torch.unique(F[arg[:,1],:])
    return neighbours

def Dilation(arg_point,V,F,texture):
    # print(f'arg point {arg_point}')
    arg_point = torch.tensor([arg_point]).cuda().to(torch.int64)
    F = F.cuda()
    texture = texture.cuda()
    neighbour = Neighbours(arg_point,F)
    arg_texture = torch.argwhere(texture == 1).squeeze()
    # dif = NoIntersection(arg_texture,neighbour)
    dif = neighbour.to(torch.int64)
    dif  = Difference(arg_texture,dif)
    n = 0
    while len(dif)!= 0 :#and n < 50:
        # print(f'n = {n}, len : {len(dif)}')
        texture[dif] = 1
        neighbour = Neighbours(dif,F)
        arg_texture = torch.argwhere(texture == 1).squeeze()
        # dif = NoIntersection(arg_texture,neighbour)
        dif = neighbour.to(torch.int64)
        dif  = Difference(arg_texture,dif)
        n+=1
    return texture



def main(args):
    file = args.file
    path_out = args.folder_out
    ratio_rect_front = args.ratio_anterior
    ratio_rect_back = args.ratio_posterior
    radius = args.radius_draw_line

    surf = ReadSurf(file)
    surf_out = vtk.vtkPolyData()
    surf_out.DeepCopy(surf)
    # centroidf = vtkMeanTeeth([6,11,3,14],property='Universal_ID')
    centroidf = vtkMeanTeeth([3,14,6,11,5,12],property='Universal_ID')

    # centroidf = vtkMeanTeeth([2,5,12,15,6,11,3,14],property='Universal_ID')
    try :
        surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],['3','8','9','14'])

        centroid = centroidf(surf)

    except ToothNoExist as error:
        print(f' Error with  : {file} \n {error}')
        quit()

    V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
    F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
    
    ratio = 0.0
    plus_back = -1
    
    # haut_gauche1 = ((1- ratio ) * centroid['5']+ ratio * centroid['6'])
    # haut_droite1 = ((1- ratio ) * centroid['12']+ ratio * centroid['11'])
    # haut_gauche1 = centroid['5']
    # haut_droite1 = centroid['12']
    haut_gauche1 = centroid['5'] + np.array([0,-plus_back,0],dtype=np.float32)
    haut_droite1 = centroid['12'] + np.array([0,-plus_back,0],dtype=np.float32)
    # haut_gauche1 =  ((1- ratio ) * centroid['5']+ ratio * centroid['4']) +np.array([0,-plus_back,0],dtype=np.float32)
    # haut_droite1 = centroid['12']  + np.array([0,-plus_back,0],dtype=np.float32)

        
    # haut_gauche1 = ((1- ratio ) * centroid['5']+ ratio * centroid['6'])  +np.array([0,-plus_back  ,0],dtype=np.float32)
    # haut_droite1 = ((1- ratio ) * centroid['12']+ ratio * centroid['11'])  +np.array([0,-plus_back ,0],dtype=np.float32)


    ratio = 0.5
    plus_back= 1.9
    plus_back2 = 1
    # bas_gauche1 = centroid['2']
    # bas_droite1 = centroid['15']
    bas_gauche1 = centroid['3'] + np.array([0,-plus_back,0],dtype=np.float32)
    bas_droite1 = centroid['14']+ np.array([0,-plus_back ,0],dtype=np.float32)
    # bas_droite1 = ((1-ratio)*centroid['14']+ratio  * centroid['15']) + np.array([0,-plus_back2,0],dtype=np.float32)
    # bas_gauche1 = ((1-ratio)*centroid['3']+ratio * centroid['2']) + np.array([0,-plus_back2,0],dtype=np.float32)
    # bas_droite1 = ((1-ratio)*centroid['14']+ ratio  * centroid['15']) 
    # bas_gauche1 = ((1-ratio)*centroid['3']+ ratio * centroid['2']) 


    ratio_rect_front = 0.31
    haut_droite = (1-ratio_rect_front) * haut_gauche1 + ratio_rect_front * haut_droite1
    ratio_rect_front = 0.30
    haut_gauche = (1-ratio_rect_front) * haut_droite1 + ratio_rect_front * haut_gauche1
    haut_middle = (haut_gauche + haut_droite) / 2


    ratio_rect_back = 0.33
    bas_droite = (1-ratio_rect_back) * bas_gauche1 + ratio_rect_back * bas_droite1
    ratio_rect_back = 0.33
    bas_gauche = (1- ratio_rect_back) * bas_droite1 + ratio_rect_back * bas_gauche1
    bas_middle = (bas_droite + bas_gauche) / 2



    middle_droite = (bas_droite + haut_droite) /2
    middle_gauche = (bas_gauche + haut_gauche)/2

    middle = (bas_droite + haut_gauche) / 2
    middle_droite = (haut_droite + bas_droite) /2 



    #rectangle limit
    t = np.arange(0,1,0.01)
    haut_seg = Segment2D(haut_droite,haut_gauche)
    haut_seg = torch.tensor(haut_seg(t)).t().to(torch.float32)
    # print(haut_seg)
    dis = torch.cdist(haut_seg,V[:,:2])
    arg_haut_seg = torch.unique(torch.argwhere(dis < radius).squeeze()[:,1])


    bas_seg = Segment2D(bas_droite,bas_gauche)
    bas_seg = torch.tensor(bas_seg(t)).t().to(torch.float32)
    dis = torch.cdist(bas_seg,V[:,:2])
    arg_bas_seg = torch.unique(torch.argwhere(dis < radius).squeeze()[:,1])



    



    #bezier droite
    bezier = Bezier_bled(bas_gauche[:2],bas_middle[:2],haut_gauche[:2],0.01)
    v_bezier = bezier - np.expand_dims(bas_gauche[:2],axis=0)
    v_norm_bezier = np.expand_dims(np.linalg.norm(v_bezier, axis=1),axis=0).T
    v_bezier = v_bezier / v_norm_bezier

    v = np.expand_dims(haut_gauche[:2] - bas_gauche[:2], axis=0).T
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    # print(f'v {v}')
    P = np.matmul(v , v.T)

    bezier_proj = ( P @ v_bezier.T).T *v_norm_bezier + bas_gauche[:2]
    sym = 2*bezier_proj - bezier

    bezier = torch.tensor(sym,dtype=torch.float32)
    dist = torch.cdist(bezier,V[:,:2])
    arg_bezier = torch.argwhere(dist < radius)[:,1]





    #bezier gauche
    bezier2 = Bezier_bled(bas_droite[:2],bas_middle[:2],haut_droite[:2],0.01)
    v_bezier = bezier2 - np.expand_dims(bas_droite[:2],axis=0)
    v_norm_bezier = np.expand_dims(np.linalg.norm(v_bezier, axis=1),axis=0).T
    v_bezier = v_bezier / v_norm_bezier

    v = np.expand_dims(haut_droite[:2] - bas_droite[:2], axis=0).T
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    # print(f'v {v}')
    P = np.matmul(v , v.T)

    bezier_proj = ( P @ v_bezier.T).T *v_norm_bezier + bas_droite[:2]
    sym = 2*bezier_proj - bezier2

    bezier2 = torch.tensor(sym,dtype=torch.float32)
    dist = torch.cdist(bezier2,V[:,:2])
    arg_bezier2 = torch.argwhere(dist < radius)[:,1]





    V_label = torch.zeros((V.shape[0]))
    V_label[arg_haut_seg] = 1
    V_label[arg_bas_seg] = 1
    V_label[arg_bezier] = 1
    V_label[arg_bezier2] = 1



    dist = torch.cdist(torch.tensor(middle[:2]).unsqueeze(0),V[:,:2]).squeeze()
    middle_arg = torch.argmin(dist)
    V_label = Dilation(middle_arg,V,F,V_label)



    V_labels_prediction = numpy_to_vtk(V_label.cpu().numpy())
    V_labels_prediction.SetName('Butterfly')



    surf_out.GetPointData().AddArray(V_labels_prediction)



    basename = os.path.basename(file)
    WriteSurf(surf_out,os.path.join(path_out,basename))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file',help='file to make patch',type=str)
    parser.add_argument('--folder_out',type=str,default='/home/luciacev/Desktop/Data/IOSReg/Meg/CHECKONEBYONE/test/')
    parser.add_argument('--ratio_anterior',type=float,default=0.3)
    parser.add_argument('--ratio_posterior',type=float,default=0.33)
    parser.add_argument('--radius_draw_line',type=float,default=0.7)

    args = parser.parse_args()

    main(args)
