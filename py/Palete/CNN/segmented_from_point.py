
import numpy as np
import plotly.graph_objects as go
from vtk.util.numpy_support import vtk_to_numpy
import torch

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
    u = (Aline.y0 - Bline.y0)/Bline.b
    t = ( Bline.a * u + Bline.x0 - Aline.x0) / Aline.a
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
            if intersection(seg_surf,seg_landmark):
                intersection_number += 1 

        if intersection_number % 2 :
            list_inside.append(V[idx].unsqueeze(0))
            list_index.append(torch.tensor(idx))

    list_inside = torch.cat(list_inside,dim = 0)
    print(f'list inside {list_inside.shape}')

    list_index = torch.tensor(list_index)

    

    return list_inside, list_index