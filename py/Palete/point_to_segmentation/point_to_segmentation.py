import numpy as np
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import plotly.graph_objects as go

from utils import ReadSurf, get_landmarks_position
from icp import PrePreAso
import torch
from torch import tensor
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm

def ListToMesh(list,radius=0.1):
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
    

    def __contains__(self,point):
        t = (point[0] - self.x0 ) / self.a
        out = False
        if 0 <= t <= 1 :
            out = True

        return out
    
    def name(self):
        return self.name_point1, self.name_point2
    
    def toDisplay(self,color='blue'):
        point1 = self.__call__(0)
        point2 = self.__call__(1)

        return go.Scatter3d(x=[point1[0],point2[0]],y=[point1[1],point2[1]],z=[0,0],mode='lines',
        line=go.scatter3d.Line(color=color,width=1))
        
    
    

def intersection(Aline : Segment2D, Bline : Segment2D):
    u = (Bline.y0 - Aline.b *( Bline.a - Aline.x0)/Aline.a - Aline.y0) / ((Aline.b * Bline.a / Aline.a - Bline.b))
    t = ( Bline.a * u + Bline.x0 - Aline.x0) / Aline.a

    out = Bline(u)
    # out2 = Aline(t)
    # print(f' double out {out} , out2: {out2}')

    return out#, u , t

def intersectiom2(Aline : Segment2D, Bline : Segment2D):
    """_summary_

    Args:
        Aline (Segment2D): search if the point is inside of the object
        Bline (Segment2D): object

    Returns:
        _type_: _description_
    """
    u = (Aline.y0 - Bline.y0)/Bline.b
    t = ( Bline.a * u + Bline.x0 - Aline.x0) / Aline.a
    out = False
    if 0 <= u <= 1 and 0<= t <= 1:
        out = True

    return out

def intersectionU(Aline : Segment2D, Bline : Segment2D):
    u = (Bline.y0 - Aline.b *( Bline.a - Aline.x0)/Aline.a - Aline.y0) / ((Aline.b * Bline.a / Aline.a - Bline.b))
    
    return u


def intersectionBool(Aline : Segment2D, Bline : Segment2D) : 
    u = intersectionU(Aline, Bline)
    t = ( Bline.a * u + Bline.x0 - Aline.x0) / Aline.a
    # print(f'u : {u}, v: {t} , Aline {Aline.name()}, Bline : {Bline.name()}')
    # print(f'u {u}, name seg : {Bline.name()}')
    out = False
    if 0 <= u <= 1 and 0<= t <= 1:
        out = True

    return out


def Segmentation(list_point,surf= None , vertex = None):
    

    if surf is not None :
        V = vtk_to_numpy(surf.GetPoints().GetData())
    else :
        V = vertex 
    V_translate = V + np.array([[100,0,0]])
    list_segment_vertex = []
    for v , v_translate in zip(V,V_translate) :
        list_segment_vertex.append(Segment2D(v,v_translate ))

    list_point_segment = []
    for idx in range(-len(list_point),0):
         list_point_segment.append(Segment2D(list_point[idx],list_point[idx+1]))

    list_inside = []
    list_index = []
    for idx , seg_surf in enumerate(list_segment_vertex):
        intersection_number = 0 
        for seg_landmark in list_point_segment :
            if intersectiom2(seg_surf,seg_landmark):
                intersection_number += 1 

        if intersection_number % 2 :
            list_inside.append(torch.tensor(V[idx]).unsqueeze(0))
            list_index.append(torch.tensor(idx))

    list_inside = torch.cat(list_inside,dim = 0)
    print(f'list inside {list_inside.shape}')
    # list_index = torch.cat(list_index, dim = 0)
    list_index = torch.tensor(list_index)

    return list_inside, list_index



path = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/1stmeasurement/CF01T1_out.vtk'
path_json = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/1stmeasurement/CF01T1.json'
matrix = np.eye(4)


#surf
surf = ReadSurf(path)
surf , matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['4','9','10','15'])

translation = np.array([100,0,0])
V = vtk_to_numpy(surf.GetPoints().GetData())
list_seg_surf = [Segment2D(v, np.array(v)+translation) for v in V]





#landmark
list_name_landmark = ['R3RL','R2RM','L2RM','L3RL','L3RM','LPR','RPR','R3RM']
landmarks = get_landmarks_position(path_json,list_name_landmark,matrix=matrix)


list_line = []
landmark_noZ = [np.append(landmark[:2],0) for landmark in landmarks ]

for idx in range(-len(landmarks),0):
    list_line.append(Segment2D(landmarks[idx],landmarks[idx+1],name_point1=list_name_landmark[idx], name_point2=list_name_landmark[idx+1]))



#test

# pointA = np.array([-5,3,0])
# pointB = np.array([-20,-5,0])
 
# list_point = [pointA, pointB]

# translation = np.array([100,0,0])

# segA = Segment2D(pointA, pointA+translation,name_point1='point A')
# segB = Segment2D(pointB, pointB+ translation,name_point1='point B')

# print(f' seg A : 1 {segA(1)} , 2 {segA(0)}')

# list_seg = [segA, segB]
# list_intersection = []
# print(f' len seg landmark {len(list_line)}')
# for idx , seg in enumerate(list_seg) :
#     intersection_number = 0
#     for landmark_seg in list_line :
#         pos_intersection= intersection(seg,landmark_seg)
#         # pos_intersection = np.append(landmark_seg(intersectionU(seg,landmark_seg)),0)
#         # if pos_intersection in seg :    
#         if pos_intersection in seg and pos_intersection in landmark_seg:
#             intersection_number += 1
#             list_intersection.append(np.append(pos_intersection,0))
#     print(f' {idx} intersection number {intersection_number} ')


# print(f'pos_intersection {list_intersection} ')



# #detection
# list_inside = []
# list_outside = []
# for idx , seg_surf in tqdm(enumerate(list_seg_surf),total = len(list_seg_surf) ):
#     intersection_number = 0 
#     for seg_landmark in list_line :
#         # pos_intersection = intersection(seg_surf,seg_landmark)
#         # if pos_intersection in seg_landmark and pos_intersection in seg_surf:
#         if intersectiom2(seg_surf,seg_landmark) :
#             intersection_number += 1 

#     if intersection_number % 2 :
#         list_inside.append(V[idx])
    
#     else :
#         list_outside.append(V[idx])



#faster
# x = torch.linspace(start = -10 , end = 10, steps =100).unsqueeze(1).expand(-1,3)

# V = x

# print(f' V { V.shape}')


# V = torch.tensor(V).to(torch.float64)
# translation = torch.tensor([[100,0,0]])
# V_translation = V + translation
# AV = V - V_translation
# print(f' AV {AV.shape}') 

# LandmarkT1 = torch.cat([torch.tensor(landmarks[idx]).unsqueeze(0).to(torch.float64) for idx in range(-len(landmarks),0)],dim=0)
# print(f'landmarkT1 {LandmarkT1}')
# LandmarkT2 = torch.cat([torch.tensor(landmarks[idx+1]).unsqueeze(0).to(torch.float64) for idx in range(-len(landmarks),0)],dim=0)
# print(f'landmarkT2 {LandmarkT2}')

# ALandmark = LandmarkT1 - LandmarkT2

# LandmarkT2 = LandmarkT2.unsqueeze(1).expand(-1,V.shape[0],-1)
# ALandmark = ALandmark.unsqueeze(1).expand(-1,V.shape[0],-1)
# print(f' A Landmark 0 : {ALandmark[:,0,:]}, 1 : {ALandmark[:,1,:]}')

# AV = AV.unsqueeze(0).expand(LandmarkT2.shape[0],-1,-1)
# V_translation= V_translation.unsqueeze(0).expand(LandmarkT2.shape[0],-1,-1)
# print(f' V_translation 0 {V_translation[:,0,:]}, 1 {V_translation[:,1,:]}')

# print(f' Vtranslation {V_translation[...,1].shape}')
# print(f' LandmarkT2 {LandmarkT2[...,1].shape}')
# print(f' ALandmark {ALandmark[...,1].shape}')
# print(f' AV {AV[...,1].shape}')

# U = torch.mul(V_translation[...,1] - LandmarkT2[...,1] ,  1 / ALandmark[...,1])
# intersection_landmark = 


# print(f'U {U.shape}')
# T = torch.mul(torch.mul(ALandmark[...,0], U) + LandmarkT2[...,0] - V_translation[...,0]  ,1/AV[...,0])

# # Uintersection = torch.where((U > 0)& (U < 1) ,1,0)
# Uintersection = torch.where((U > -1)& (U < 0) ,1,0)

# Tintersection = torch.where((T > 0)& (T < 1) ,1,0)


# print(f' unique U {torch.unique(Uintersection)}, V {torch.unique(Tintersection)}')

# Uintersection = torch.sum(Uintersection, dim= 0 )
# Tintersection = torch.sum(Tintersection, dim= 0 )

# print(f' unique U {torch.unique(Uintersection)}, V {torch.unique(Tintersection)}')

# Uintersection = torch.where( Uintersection%2 == 1    ,1,0)
# Tintersection = torch.where( Tintersection%2 == 1  ,1,0)

# Intersection = torch.where((Uintersection + Tintersection) == 2, 1, 0)
# Intersection = torch.argwhere(Intersection)

# print(f'V {V.shape}')
# Vinside = V[Intersection,:].squeeze()


# print(f'intersection {Vinside.shape}')


Vinside , index = Segmentation(landmarks,vertex = V)


# print(f' shape list inside : {torch.tensor(list_inside).unsqueeze(0).shape}')
inside = Pointclouds(torch.tensor(Vinside).unsqueeze(0))
# outside = Pointclouds(torch.tensor(list_outside).unsqueeze(0))

fig = plot_scene({
"subplot1": {
    'landmark': ListToMesh(landmark_noZ),
    # 'point' : ListToMesh(list_point),
    # 'intersection' : ListToMesh(list_intersection),
    'inside' : inside,
    # 'outside' : outside
}
})

for line in list_line :
    fig.add_trace(line.toDisplay())

# # # fig.add_trace(segA.toDisplay())
fig.show()
print("fait")