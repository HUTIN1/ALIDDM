import numpy as np
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import plotly.graph_objects as go

from utils import ReadSurf, get_landmarks_position, GetUnitSurf
from icp import PrePreAso
import torch
from torch import tensor
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm


def ListToMesh(list,radius=0.05):
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


path = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/1stmeasurement/CF01T1_out.vtk'
path_json = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/1stmeasurement/CF01T1.json'
matrix = np.eye(4)


#surf
surf = ReadSurf(path)
surf , matrix = PrePreAso(surf,[[-0.5,-0.5,0],[0,0.5,0],[0.5,-0.5,0]],['4','9','10','15'])
surf, _, _ = GetUnitSurf(surf)



V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData()))

mean = torch.mean(V,dim=0)


arg = torch.argsort(V[...,0],dim=0)[10]

new_tensor = []
pos_max = V[arg]
pos_max2 = pos_max[2]
print(f'pos max {pos_max}')
for v in V :
    if v[2] > pos_max2 or  torch.dist(v,mean) < 0.4:
        new_tensor.append(v.unsqueeze(0))


new_tensor = torch.cat(new_tensor,dim=0)
print(f'new tensor { new_tensor.shape}')
surf_without_base = Pointclouds(new_tensor.unsqueeze(0))

fig = plot_scene({
"subplot1": {
    'surf' : surf_without_base,
    'max' : ListToMesh(pos_max.unsqueeze(0)),
    'mean' : ListToMesh(mean.unsqueeze(0))
}
})
fig.show()
