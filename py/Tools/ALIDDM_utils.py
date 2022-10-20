import os
import glob
import json
import csv
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

from vtk import vtkMatrix4x4, vtkMatrix3x3, vtkPoints
import torch
import pandas as pd
from utils import ReadSurf
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,look_at_rotation,AmbientLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex,blending

from torch import tensor

from monai.transforms import (
    ToTensor
)


from shader import *
import utils
from utils import GetColorArray

def numberTooth2Landmark(tooth):
    if isinstance(tooth,int):
        tooth = str(tooth)
    Lower = ['LL7O', 'LL7MB', 'LL7DB', 'LL6O', 'LL6MB', 'LL6DB', 'LL5O', 'LL5MB', 
'LL5DB', 'LL4O', 'LL4MB', 'LL4DB', 'LL3O', 'LL3MB', 'LL3DB', 'LL2O', 'LL2MB',
 'LL2DB', 'LL1O', 'LL1MB', 'LL1DB', 'LR1O', 'LR1MB', 'LR1DB', 'LR2O', 'LR2MB', 
 'LR2DB', 'LR3O', 'LR3MD', 'LR3DB', 'LR4O', 'LR4MB', 'LR4DB', 'LR5O', 'LR5MB',
  'LR5DB', 'LR6O', 'LR6MB', 'LR6DB', 'LR7O', 'LR7MB', 'LR7DB', 'LL7R', 'LL6R', 
  'LL5R', 'LL4R', 'LL3R', 'LL2R', 'LL1R']

    Upper = ['UL7O', 'UL7MB', 'UL7DB', 'UL6O', 'UL6MB', 'UL6DB', 'UL5O', 'UL5MB', 
    'UL5DB', 'UL4O', 'UL4MB', 'UL4DB', 'UL3O', 'UL3MB', 'UL3DB', 'UL2O', 'UL2MB', 
    'UL2DB', 'UL1O', 'UL1MB', 'UL1DB', 'UR1O', 'UR1MB', 'UR1DB', 'UR2O', 'UR2MB', 
    'UR2DB', 'UR3O', 'UR3MD', 'UR3DB', 'UR4O', 'UR4MB', 'UR4DB', 'UR5O', 'UR5MB', 
    'UR5DB', 'UR6O', 'UR6MB', 'UR6DB', 'UR7O', 'UR7MB', 'UR7DB', 'UL7R', 'UL6R', 
    'UL5R', 'UL4R', 'UL3R', 'UL2R', 'UL1R']
    LANDMARKS = {
    "L":Lower,
    "U":Upper
}
    LABEL = {
    "15" : LANDMARKS["U"][0:3],
    "14" : LANDMARKS["U"][3:6],
    "13" : LANDMARKS["U"][6:9],
    "12" : LANDMARKS["U"][9:12],
    "11" : LANDMARKS["U"][12:15],
    "10" : LANDMARKS["U"][15:18],
    "9" : LANDMARKS["U"][18:21],
    "8" : LANDMARKS["U"][21:24],
    "7" : LANDMARKS["U"][24:27],
    "6" : LANDMARKS["U"][27:30],
    "5" : LANDMARKS["U"][30:33],
    "4" : LANDMARKS["U"][33:36],
    "3" : LANDMARKS["U"][36:39],
    "2" : LANDMARKS["U"][39:42],

    "18" : LANDMARKS["L"][0:3],
    "19" : LANDMARKS["L"][3:6],
    "20" : LANDMARKS["L"][6:9],
    "21" : LANDMARKS["L"][9:12],
    "22" : LANDMARKS["L"][12:15],
    "23" : LANDMARKS["L"][15:18],
    "24" : LANDMARKS["L"][18:21],
    "25" : LANDMARKS["L"][21:24],
    "26" : LANDMARKS["L"][24:27],
    "27" : LANDMARKS["L"][27:30],
    "28" : LANDMARKS["L"][30:33],
    "29" : LANDMARKS["L"][33:36],
    "30" : LANDMARKS["L"][36:39],
    "31" : LANDMARKS["L"][39:42],

}

    return LABEL[tooth]


    
def SplitCSV_train_Val(csvPath,val_p):
    df = pd.read_csv(csvPath)
    df_test = df.loc[df['for'] == "test"]
    df_train = df.loc[df['for'] == "train"]
    samples = int(len(df_train.index)*val_p)

    for i in range(samples):
        random_num = random.randint(1, 131)
        # print(random_num)
        df_train['for'][random_num] = "val"

    # df.to_csv(csvPath,index=False)

    df_fold = pd.concat([df_train, df_test])
    # print(df_fold['for'])
    df_fold.to_csv(csvPath,index=False)
    

def GenDataSplitCSV(dir_data,csv_path,val_p,test_p,device):
    patient_dic = {}
    normpath = os.path.normpath("/".join([dir_data, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn):
            basename = os.path.basename(img_fn).split('.')[0].split("_P")
            jow = basename[0][0]
            patient_ID = "P"+basename[1]
            if patient_ID not in patient_dic.keys():
                patient_dic[patient_ID] = {"L":{},"U":{}}
            
            if ".json" in img_fn:
                patient_dic[patient_ID][jow]["lm"] = img_fn
            elif ".vtk" in img_fn:
                patient_dic[patient_ID][jow]["surf"] = img_fn

    # print(patient_dic)
    test_p = test_p/100
    val_p = val_p/100
    val_p = val_p/(1-test_p)

    patient_dic = list(patient_dic.values())
    random.shuffle(patient_dic)

    # print(patient_dic)
    df_train, df_test = train_test_split(patient_dic, test_size=test_p)
    df_train, df_val = train_test_split(df_train, test_size=val_p)

    data_dic = {
        "train":df_train,
        "val":df_val,
        "test":df_test
        }
    # print(data_dic)
    fieldnames = ['for','jaw','surf', 'landmarks']
    for lab in range(2,32):
        fieldnames.append(str(lab))
    data_list = []
    for type,dic in data_dic.items():
        for patient in dic:
            for jaw,data in patient.items():
                if jaw == "U":
                    rows = {
                        'for':type,
                        'jaw':jaw,
                        'surf':data["surf"].replace(dir_data,"")[1:],
                        'landmarks':data["lm"].replace(dir_data,"")[1:],
                        }
                    # print(data["surf"])
                    read_surf = ReadSurf(data["surf"])
                    ids = ToTensor(dtype=torch.int64,device = device )(vtk_to_numpy(read_surf.GetPointData().GetScalars("PredictedID")))
                    # print(ids)

                    for label in range(2,32):
                        
                        if label in ids:
                            present = 1
                        else:
                            present = 0
                        
                        rows[str(label)] = present

                    data_list.append(rows)
    
    with open(csv_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)
    # return outfile

def Gen_patch(V, RED, LP, label, radius):
    lst_landmarks = Get_lst_landmarks(LP,GV.LABEL[label])
    color_index=0
    for landmark_coord in lst_landmarks:
        landmark_coord =landmark_coord.unsqueeze(1)
        distance = torch.cdist(landmark_coord, V, p=2)
        distance = distance.squeeze(1)
        index_pos_land = (distance<radius).nonzero(as_tuple=True)
        color = [torch.tensor([1,0,0]),torch.tensor([0,1,0]),torch.tensor([0,0,1])]
        for i,index in enumerate(index_pos_land[0]):
            RED[index][index_pos_land[1][i]] = color[color_index]
        color_index +=1 
    
    return RED

def Gen_mesh_patch(surf,V,F,CN,LP,label):
    verts_rgb = torch.ones_like(CN)[None].squeeze(0)  # (1, V, 3)
    verts_rgb[:,:, 0] *= 0  # red
    verts_rgb[:,:, 1] *= 0  # green
    verts_rgb[:,:, 2] *= 0  # blue 
    # patch_region = Gen_patch(surf, V, verts_rgb, LP, label, 3)
    patch_region = Gen_patch(V, verts_rgb, LP, label, 0.02)    
    textures = TexturesVertex(verts_features=patch_region)
    meshes = Meshes(
        verts=V,   
        faces=F, 
        textures=textures
    ) # batchsize
    
    return meshes


def MeanScale(surf =None ,verts = None):
    if surf : 
        verts = vtk_to_numpy(surf.GetPoints().GetData())
    min_coord = np.min(verts, axis=0)
    max_coord= np.max(verts, axis=0)
    mean = (max_coord + min_coord)/2.0
    scale = np.linalg.norm(max_coord - mean)

    return mean, scale

def FocusTeeth(surf,surf_property,number_teeth):
    region_id = tensor((vtk_to_numpy(surf.GetPointData().GetScalars(surf_property))),dtype=torch.int64)
    crown_ids = torch.argwhere(region_id == number_teeth).reshape(-1)

    verts = vtk_to_numpy(surf.GetPoints().GetData())
    verts_crown = verts[crown_ids]

    mean, scale = MeanScale(verts = verts_crown)
    
    return mean , scale , utils.GetUnitSurf(surf, mean_arr= mean, scale_factor = 1/scale)



def get_landmarks_position(dataset_dir,df,idx, mean_arr, scale_factor,lst_landmarks):
       
        data = json.load(open(os.path.join(dataset_dir,df.iloc[idx]["landmarks"])))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']

        landmarks_position = np.zeros([len(lst_landmarks), 3])
        # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
        for landmark in landmarks_lst:
            label = landmark["label"]
            if label in lst_landmarks:
                landmarks_position[lst_landmarks.index(label)] = Downscale(landmark["position"],mean_arr,scale_factor)

        landmarks_pos = np.array([np.append(pos,1) for pos in landmarks_position])
        return landmarks_pos[:, 0:3]


def pos_landmard2texture(vertex,list_landmark_pos):
    texture = torch.zeros_like(vertex.unsqueeze(0))
    vertex = vertex.to(torch.float64)
    radius = 0.02
    for i, landmark_pos in enumerate(list_landmark_pos):

        landmark_pos = tensor([landmark_pos])
        distance = torch.cdist(landmark_pos,vertex,p=2)
        minvalue = torch.min(distance)
        distance = distance - minvalue
        #print(min(distance,1))
        index_pos_land = torch.nonzero((distance<radius),as_tuple=True)
        for index in index_pos_land:
            texture[0,index,i]=255

    return texture

def arrayFromVTKMatrix(vmatrix):
  """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
  The returned array is just a copy and so any modification in the array will not affect the input matrix.
  To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
  :py:meth:`updateVTKMatrixFromArray`.
  """

  if isinstance(vmatrix, vtkMatrix4x4):
    matrixSize = 4
  elif isinstance(vmatrix, vtkMatrix3x3):
    matrixSize = 3
  else:
    raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
  narray = np.eye(matrixSize)
  vmatrix.DeepCopy(narray.ravel(), vmatrix)
  return narray

def Upscale(landmark_pos,mean_arr,scale_factor):
    new_pos_center = (landmark_pos/scale_factor) + mean_arr
    return new_pos_center

def Downscale(pos_center,mean_arr,scale_factor):
    landmarks_position = (pos_center - mean_arr) * scale_factor
    return landmarks_position


def DecomposeSurf(surf,surf_property,idx,mean_arr,scale_factor,lst_landmarks):
    landmark_pos = get_landmarks_position(idx, mean_arr, scale_factor,lst_landmarks)
    color_normals = ToTensor(dtype=torch.float32)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])

    region_id = ToTensor(dtype=torch.int64)(vtk_to_numpy(surf.GetPointData().GetScalars(surf_property)))

        
    region_id = torch.clamp(region_id, min=0)
    landmark_pos = torch.tensor(landmark_pos,dtype=torch.float32)
    mean_arr = torch.tensor(mean_arr,dtype=torch.float64)
    scale_factor = torch.tensor(scale_factor,dtype=torch.float64)

    return surf, verts, faces, region_id, color_normals, landmark_pos, mean_arr, scale_factor



def isRotationMatrix(M):
    M = M.numpy()
    tag = False
    I= np.identity(M.shape[0])
    if np.all(np.round_(np.matmul(M,M.T),decimals=5) == I) and (abs(np.linalg.det(M)) == 1 ): 
        tag = True
    return tag


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N,1, H, W, M) array images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for index ,ax in enumerate(axarr.ravel()):

        if index < images.shape[0]:
            ax.imshow(images[index,0,...])
            if not show_axes:
                ax.set_axis_off()

