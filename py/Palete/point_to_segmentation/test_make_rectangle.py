import os
import sys
import torch
from vtk.util.numpy_support import vtk_to_numpy
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
TexturesVertex, FoVPerspectiveCameras, look_at_rotation,look_at_view_transform
)
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import vtk
fpath = os.path.join(os.path.dirname(__file__), '..','CNN')
sys.path.append(fpath)

from icp import vtkMeanTeeth, PrePreAso
from utils import ReadSurf, RandomRotation, WriteSurf

fpath = '/home/luciacev/Desktop/Project/ALIDDM/ALIDDM/py/Palete/GCNnet'
sys.path.append(fpath)

from segmented_from_point import Segmentation

def ListToMesh(list,radius=0.5):
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

def Neighbours(arg_point,F):
    neighbours = torch.tensor([]).cuda()
    F2 = F.unsqueeze(0).expand(len(arg_point),-1,-1)
    arg_point = arg_point.unsqueeze(1).unsqueeze(2)
    arg = torch.argwhere((F2-arg_point) == 0)
    # print(f' arg {arg.shape}, arg : {arg}')

    neighbours = torch.unique(F[arg[:,1],:])
    return neighbours

# def NoIntersection(t1,t2):
#     t1 = t1.squeeze()
#     t2 = torch.tensor(t2).squeeze()
#     combined = torch.cat((t1, t2))
#     uniques, counts = combined.unique(return_counts=True)
#     difference = uniques[counts == 1]
#     combined = torch.cat((difference, t2))
#     uniques, counts = combined.unique(return_counts=True)
#     difference = uniques[counts == 1]
#     return difference

def Difference(t1,t2):
    # print(f't2 : {t2}')
    # print(f't1 shape {t1.shape}')
    t1 = t1.unsqueeze(0).expand(len(t2),-1)
    t2 = t2.unsqueeze(1)
    d = torch.count_nonzero(t1 -t2,dim=-1)
    arg = torch.argwhere(d == t1.shape[1])
    dif = torch.unique(t2[arg])
    return dif

def Dilation(arg_point,V,F,texture):
    arg_point = torch.tensor([arg_point]).cuda().to(torch.int64)
    F = F.cuda()
    texture = texture.cuda()
    neighbour = Neighbours(arg_point,F)
    arg_texture = torch.argwhere(texture == 1).squeeze()
    # dif = NoIntersection(arg_texture,neighbour)
    dif = neighbour.to(torch.int64)
    dif = Difference(arg_texture,dif)
    n = 0
    while len(dif)!= 0 :
        # print(f'n = {n}, len : {len(dif)}')
        texture[dif] = 1
        neighbour = Neighbours(dif,F)
        arg_texture = torch.argwhere(texture == 1).squeeze()
        # dif = NoIntersection(arg_texture,neighbour)
        dif = neighbour.to(torch.int64)
        dif = Difference(arg_texture,dif)
        n+=1
    return texture

    

path = '/home/luciacev/Desktop/Data/IOSReg/ARON_GOLD/organize/test/input/A4_UpperT2.vtk'
path_out = '/home/luciacev/Desktop/Data/IOSReg/ARON_GOLD/organize/test/test/'
surf = ReadSurf(path)
surf_out = vtk.vtkPolyData()
surf_out.DeepCopy(surf)

surf, matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],['3','8','9','14'])
# surf, _ ,_ = RandomRotation(surf)


centroid = vtkMeanTeeth([5,12,2,15,6,11,3,14],property='Universal_ID')
centroid = centroid(surf)
V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
print(f'V {V.shape}')
F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
# ratio_rect = 0.28
ratio_rect = 0.3
# haut_gauche1 = (centroid['5']+centroid['6'])/2
# haut_droite1 = (centroid['11']+centroid['12'])/2
haut_droite1 = centroid['12']
haut_gauche1 = centroid['5']

haut_droite = (1-ratio_rect) * haut_gauche1 + ratio_rect * haut_droite1
haut_gauche = (1-ratio_rect) * haut_droite1 + ratio_rect * haut_gauche1
haut_middle = (haut_gauche + haut_droite) / 2


# bas_gauche1 = centroid['3']
# bas_droite1 = centroid['14']

# bas_gauche1 = centroid['2']
# bas_droite1 = centroid['15']

ratio_bas = 0.25
bas_gauche1 = ((1-ratio_bas) * centroid['3'] + ratio_bas * centroid['2'])
bas_droite1 = ((1-ratio_bas) * centroid['14']+ ratio_bas * centroid['15'])

# ratio_rect = 0.35
ratio_rect =0.3
bas_droite = (1-ratio_rect) * bas_gauche1 + ratio_rect * bas_droite1
bas_gauche = (1- ratio_rect) * bas_droite1 + ratio_rect * bas_gauche1
bas_middle = (bas_droite + bas_gauche) / 2


print(f' bas droite {bas_droite}, haut droite {haut_droite}, haut gauche {haut_gauche}, bas gauche {bas_gauche}')
middle_droite = (bas_droite + haut_droite) /2
middle_gauche = (bas_gauche + haut_gauche)/2

middle = (bas_droite + haut_gauche) / 2
middle_droite = (haut_droite + bas_droite) /2 

distance = torch.cdist(torch.tensor(middle_droite[:2]).unsqueeze(0),V[:,:2])
radius = 2
arg2 = torch.argwhere(distance < radius).squeeze()

# V_center = V - torch.tensor(middle).unsqueeze(0)

# height_vector = torch.tensor(haut_middle - middle).unsqueeze(0)
# side_vector = torch.tensor(middle_droite - middle).unsqueeze(0)



# print(f' height vector {height_vector}, side vector {side_vector}, V center {V_center.shape} middle {middle}')

# arg = torch.argwhere((torch.abs(V_center[:,1]) < torch.abs(height_vector[:,1])) & (torch.abs(V_center[:,0]) < torch.abs(side_vector[:,0])))
# arg = torch.argwhere( (torch.abs(V_center[:,:2]) < torch.abs(torch.tensor(bas_gauche).unsqueeze(0)[:,:2])))
# arg = torch.argwhere((torch.abs(V_center[:,0]) < height_vector[:,0]) & (torch.abs(V_center[:,1]) <height_vector[:,1]))

# middle_byside = (middle_droite + middle_gauche) / 2
# V_center_side = V - torch.tensor(middle_byside).unsqueeze(0)
# droite_center = torch.tensor(middle_droite - middle_byside).unsqueeze(0)

# arg1= torch.argwhere(torch.abs(V_center_side[:,0]) < torch.abs(droite_center[:,0]))



# middle_bys = (haut_middle + bas_middle) / 2
# V_center_bys = V - torch.tensor(middle_bys).unsqueeze(0)
# bas_vector = torch.tensor(bas_middle - middle_bys).unsqueeze(0)

# arg2= torch.argwhere(torch.abs(V_center_bys[:,1]) < torch.abs(bas_vector[:,1]))


# V_center = V - torch.tensor(bas_droite).unsqueeze(0)
# vector = torch.tensor(haut_gauche - bas_droite).unsqueeze(0)

# arg= torch.argwhere((V_center[:,0] < vector[:,0]) & (V_center[:,1] < vector[:,1]) & (V_center[:,0] > 0) & (V_center[:,1] > 0))

# V_center = V - torch.tensor(haut_droite).unsqueeze(0)
# x_vector = torch.tensor(haut_gauche - haut_droite).unsqueeze(0)
# y_vector = torch.tensor(bas_droite - haut_droite).unsqueeze(0)
# arg = torch.argwhere((V_center[:,0] < x_vector[:,0]) & (V_center[:,0] > 0 ) & (V_center[:,1] > y_vector[:,1]) & (V_center[:,1] < 0))

# AB = torch.tensor(bas_gauche - bas_droite)
# # BC = torch.tensor(haut_gauche - bas_gauche)
# # CD = torch.tensor(haut_droite - haut_gauche)
# AD = torch.tensor(haut_droite - bas_droite)
# print(AB[:2])
# dotABAB = torch.dot(AB[:2],AB[:2])
# # dotBCBC = torch.dot(BC[:2],BC[:2])
# # dotCDCD = torch.dot(CD[:2],CD[:2])
# dotADAD = torch.dot(AD[:2],AD[:2])

# A = torch.tensor(bas_droite)
# B = torch.tensor(bas_gauche)
# # C = torch.tensor(haut_gauche)
# a1 = np.norm((haut_droite[:2] - haut_gauche[:2]))
# a2 = np.norm((haut_gauche[:2] - bas_gauche[:2]))
# a3 = np.norm((bas_gauche[:2] - bas_droite[:2]))
# a4 = np.norm((bas_droite[:2] - haut_droite[:2]))


# list_arg = []
# for ar, v in enumerate(V) :
#     v = v.squeeze()
    # AM = v - A
    # # BM = v - B
    # # CM = v - C
    # dotABAM = torch.dot(AB[:2], AM[:2])
    # # dotBCBM = torch.dot(BC[:2],BM[:2])
    # # dotCDCM = torch.dot(CD[:2],CM[:2])
    # dotAMAD = torch.dot(AM[:2],AD[:2])
    # # if 0 <= dotABAM and dotABAM <= dotABAB and 0 <= dotBCBM and dotBCBM <= dotBCBC : 
    # if 0 <= dotABAM and dotABAM <= dotABAB and 0 <= dotAMAD and dotAMAD <= dotADAD :
    #     list_arg.append(torch.tensor(ar).unsqueeze(0))

# bezier = Bezier_bled(haut_gauche,middle,bas_gauche,0.01)
bezier = Bezier_bled(bas_gauche[:2],bas_middle[:2],haut_gauche[:2],0.01)
# bezier = Bezier_bled(bas_gauche[:2],haut_middle[:2],haut_gauche[:2],0.01)
v_bezier = bezier - np.expand_dims(bas_gauche[:2],axis=0)
v_norm_bezier = np.expand_dims(np.linalg.norm(v_bezier, axis=1),axis=0).T
v_bezier = v_bezier / v_norm_bezier

print(f'v norm bezier {v_norm_bezier.shape}, v_bezier  {v_bezier.shape}')

v = np.expand_dims(haut_gauche[:2] - bas_gauche[:2], axis=0).T
v_norm = np.linalg.norm(v)
v = v / v_norm
# print(f'v {v}')
P = np.matmul(v , v.T)

bezier_proj = ( P @ v_bezier.T).T *v_norm_bezier + bas_gauche[:2]
sym = 2*bezier_proj - bezier

bezier = torch.tensor(sym,dtype=torch.float32)
dist = torch.cdist(bezier,V[:,:2])
radius = 0.5
arg_bezier = torch.argwhere(dist < radius)[:,1]




bezier2 = Bezier_bled(bas_droite[:2],bas_middle[:2],haut_droite[:2],0.01)
v_bezier = bezier2 - np.expand_dims(bas_droite[:2],axis=0)
v_norm_bezier = np.expand_dims(np.linalg.norm(v_bezier, axis=1),axis=0).T
v_bezier = v_bezier / v_norm_bezier

v = np.expand_dims(haut_droite[:2] - bas_droite[:2], axis=0).T
v_norm = np.linalg.norm(v)
v = v / v_norm
print(f'v {v}')
P = np.matmul(v , v.T)

bezier_proj = ( P @ v_bezier.T).T *v_norm_bezier + bas_droite[:2]
sym = 2*bezier_proj - bezier2

bezier2 = torch.tensor(sym,dtype=torch.float32)
dist = torch.cdist(bezier2,V[:,:2])
radius = 0.5
arg_bezier2 = torch.argwhere(dist < radius)[:,1]










# b = np.expand_dims((middle - bas_droite),axis=0).T
# a = np.expand_dims((haut_droite - bas_droite),axis=0).T
# x =  np.linalg.inv(a.T @ a) @ a.T @ b
# # projection = np.expand_dims(x[0,0]*a.T[0:] + bas_gauche,axis=0)
# projection = x[0,0]*a.T[0:] + bas_gauche
# print(f'x {projection}')

# v = np.expand_dims(haut_gauche - bas_gauche, axis=0).T
# v_norm = np.linalg.norm(v)
# v = v / v_norm
# print(f'v {v}')
# P = np.matmul(v , v.T)
# v_middle = np.expand_dims(middle - bas_gauche ,axis=0).T
# middle_norm = np.linalg.norm(v_middle)
# v_middle = v_middle / middle_norm
# projection = ((P @ v_middle).T  )* middle_norm + np.expand_dims(bas_gauche,axis=0)
# sym = 2*projection - bezier
# print(f' projecion {projection}')





# _ , arg= Segmentation([haut_droite,haut_gauche,bas_gauche,bas_droite],vertex = V)
radius=0.5

t = np.arange(0,1,0.01)
haut_seg = Segment2D(haut_droite,haut_gauche)
haut_seg = torch.tensor(haut_seg(t)).t().to(torch.float32)
# print(haut_seg)
dis = torch.cdist(haut_seg,V[:,:2])
arg_haut_seg = torch.argwhere(dis < radius).squeeze()[:,1]
print(f'arg haut seg {arg_haut_seg.shape}  seg {arg_haut_seg}')


bas_seg = Segment2D(bas_droite,bas_gauche)
bas_seg = torch.tensor(bas_seg(t)).t().to(torch.float32)
dis = torch.cdist(bas_seg,V[:,:2])
arg_bas_seg = torch.argwhere(dis < radius).squeeze()[:,1]



# arg = torch.cat(list_arg,dim=0).squeeze(0)
# print(f'arg {arg.shape}')
# # texture1 = torch.zeros_like(V)
# # texture1[arg1,1] += 1
# # texture1[arg2,1] += 1

# arg = torch.argwhere(texture1[:,1] == 2)
texture2 = torch.zeros_like(V)
texture2[arg_bas_seg,1] =255
texture2[arg_haut_seg,1] = 255
texture2[arg_bezier,2] =255
texture2[arg_bezier2,2] =255
# texture2[arg2,1] = 255

V_label = torch.zeros((V.shape[0]))
V_label[arg_bas_seg] =1
V_label[arg_haut_seg] = 1
V_label[arg_bezier] = 1
V_label[arg_bezier2] = 1

# point = bezier[int(bezier.shape[0]/2),:].unsqueeze(0) - torch.tensor([[2,0]])
# print(f'point {point}')
# dist = torch.cdist(point,V[:,:2]).squeeze()
# min_bez = torch.argmin(dist)
# V_label = Dilation(min_bez,V,F,V_label)

# point = bezier2[int(bezier2.shape[0]/2),:].unsqueeze(0) - torch.tensor([[-2,0]])
# print(f'point {point}')
# dist = torch.cdist(point,V[:,:2]).squeeze()
# min_bez2 = torch.argmin(dist)
# V_label = Dilation(min_bez2,V,F,V_label)


dist = torch.cdist(torch.tensor(middle[:2]).unsqueeze(0),V[:,:2]).squeeze()
middle_arg = torch.argmin(dist)
V_label = Dilation(middle_arg,V,F,V_label)

V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
texture = TexturesVertex(texture2.unsqueeze(0))
mesh = Meshes(verts=V.unsqueeze(0),faces=F.unsqueeze(0),textures=texture)
# mesh = Meshes(verts=V.unsqueeze(0),faces=F.unsqueeze(0))

dist = torch.cdist(torch.tensor(bas_droite[:2]).unsqueeze(0),V[:,:2])
min = torch.argmin(dist)
min_value = V[min,:]
print(f'min value {min_value}')


plt = plot_scene({
    'subplot 1':{
    'mesh': mesh,
    'haut droite' : ListToMesh(np.expand_dims(haut_droite,axis=0)),
    'haut gauche': ListToMesh(np.expand_dims(haut_gauche,axis=0)),
    'bas droite ': ListToMesh(np.expand_dims(bas_droite,axis=0)),
    'bas gauche': ListToMesh(np.expand_dims(bas_gauche,axis=0)),
    'middle' : ListToMesh(np.expand_dims(middle,axis=0)),
    'middle haut':ListToMesh(np.expand_dims(haut_middle,axis=0)),
    'middle side ': ListToMesh(np.expand_dims(middle_droite,axis=0)),
    'projection':ListToMesh(min_value.unsqueeze(0)),
    # 'bezier' : ListToMesh(bezier,radius=0.2),
    # 'projection middle' : ListToMesh(projection),
    # 'sym':ListToMesh(sym)
    # 'min1':ListToMesh(V[min_bez,:].unsqueeze(0)),
    # 'min2':ListToMesh(V[min_bez2,:].unsqueeze(0)),
    # 'point':ListToMesh(point)
    }
})

plt.show()



V_labels_prediction = numpy_to_vtk(V_label.cpu().numpy())
V_labels_prediction.SetName('Palete_sym')



surf_out.GetPointData().AddArray(V_labels_prediction)



basename = os.path.basename(path)
WriteSurf(surf_out,os.path.join(path_out,basename))