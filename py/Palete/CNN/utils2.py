
from icp import vtkMeanTeeth
import torch
from vtk.util.numpy_support import vtk_to_numpy






def rectangle_patch_texture(surf,matrix):
    centroid = vtkMeanTeeth([5,6,11,12,3,14],property='Universal_ID')
    centroid = centroid(surf)
    V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
    ratio_rect = 0.25
    haut_gauche1 = (centroid['5']+centroid['6'])/2
    haut_droite1 = (centroid['11']+centroid['12'])/2

    haut_droite = (1-ratio_rect) * haut_gauche1 + ratio_rect * haut_droite1
    haut_gauche = (1-ratio_rect) * haut_droite1 + ratio_rect * haut_gauche1
    haut_middle = (haut_gauche + haut_droite) / 2

    bas_gauche1 = centroid['3']
    bas_droite1 = centroid['14']

    bas_droite = (1-ratio_rect) * bas_gauche1 + ratio_rect * bas_droite1
    bas_gauche = (1- ratio_rect) * bas_droite1 + ratio_rect * bas_gauche1


    print(f' bas droite {bas_droite}, haut droite {haut_droite}')
    middle_side = (bas_droite + haut_droite) /2

    middle = (bas_droite + haut_gauche) / 2

    V_center = V - torch.tensor(middle)

    height_vector = torch.tensor(haut_middle - middle).unsqueeze(0)
    side_vector = torch.tensor(middle_side - middle).unsqueeze(0)

    print(f' height vector {height_vector.shape}, side vector {side_vector.shape}, V center {V_center.shape}')

    # arg = torch.argwhere((torch.abs(V_center[:,:2]) < height_vector[:,:2]) & (torch.abs(V_center[:,:2]) < side_vector[:,:2]))
    arg = torch.argwhere( (torch.abs(V_center[:,:2]) < side_vector[:,:2]))


    texture = torch.zeros_like(V)
    texture[arg,1] = 255

    return texture
    





