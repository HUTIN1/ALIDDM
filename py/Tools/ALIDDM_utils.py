import os
import glob
import json
import csv
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import random
import vtk
from vtk import vtkMatrix4x4, vtkMatrix3x3, vtkPoints
import torch
import pandas as pd
from GlobVar import SELECTED_JAW
from ALIDDM.py.Tools.utils import ReadSurf
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from ALIDDM.py.Tools.utils import(
    PolyDataToTensors
) 
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex,blending

from torch.utils.data import DataLoader
from torch import tensor

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from monai.transforms import (
    ToTensor
)
import monai
import torchvision.transforms as transforms
from ALIDDM.py.Tools.shader import *
import utils
from utils import GetColorArray


def GenPhongRenderer(image_size,blur_radius,faces_per_pixel,device):
    
    # cameras = FoVOrthographicCameras(znear=0.1,zfar = 10,device=device) # Initialize a ortho camera.

    cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90, device= device) # Initialize a perspective camera.

    raster_settings = RasterizationSettings(        
        image_size=image_size, 
        blur_radius=blur_radius, 
        faces_per_pixel=faces_per_pixel, 
    )

    lights = PointLights(device = device) # light in front of the object. 

    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
    b = blending.BlendParams(background_color=(0,0,0))
    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader( device = device ,cameras=cameras, lights=lights,blend_params=b)
    )
    mask_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=MaskRenderer(device = device ,cameras=cameras, lights=lights,blend_params=b)
    )
    return phong_renderer,mask_renderer
    



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
    
    return mean , scale , utils.GetUnitSurf(surf, mean, scale)



def get_landmarks_position(dataset_dir,df,idx, mean_arr, scale_factor,lst_landmarks):
       
        data = json.load(open(os.path.join(dataset_dir,df.iloc[idx]["landmarks"])))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']

        lst_lm =  lst_landmarks
        landmarks_position = np.zeros([len(lst_lm), 3])
        # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
        for landmark in landmarks_lst:
            label = landmark["label"]
            if label in lst_lm:
                landmarks_position[lst_lm.index(label)] = Downscale(landmark["position"],mean_arr,scale_factor)

        landmarks_pos = np.array([np.append(pos,1) for pos in landmarks_position])
        return landmarks_pos[:, 0:3]



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
