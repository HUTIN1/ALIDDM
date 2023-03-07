import json
import vtk
import numpy as np
import torch
from torch import int64, tensor
import os

def WriteLandmark(dic_landmark,path):
    true = True
    false = False

    cp_list = []
    model={
                    "id": "1",
                    "label": '',
                    "description": "",
                    "associatedNodeID": "",
                    "position": [],
                    "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                    "selected": true,
                    "locked": false,
                    "visibility": true,
                    "positionStatus": "defined"
                }
    for idx , (landmark, pos) in enumerate(dic_landmark.items()):
        dic = model.copy()
        dic['id'] = f'{idx+1}'
        dic['label'] = f'{landmark}'
        dic['position'] = pos
        cp_list.append(dic)

    true = True
    false = False
    file = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": cp_list,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
            }
            ]
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)
    f.close




def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)


    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points
    
def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOff();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()

def segmentationLandmarks(vertex , landmarks_pos,radius):
    texture = torch.zeros(size=(vertex.shape[0],1),dtype=int64)
    vertex = vertex.to(torch.float64)
    for index , landmark_pos in enumerate(landmarks_pos) :
        landmark_pos = tensor(np.array(landmark_pos)).unsqueeze(0)
        distance = torch.cdist(landmark_pos,vertex,p=2)
        minvalue = torch.min(distance)
        distance = distance - minvalue
        _, index_pos_land = torch.nonzero((distance<radius),as_tuple=True)
        for i in index_pos_land:

            texture[i]=index + 1
    return texture

def ReadSurf(path):


    fname, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()

    return surf


def get_landmarks_position(path,landmark, mean_arr, scale_factor,matrix_rotation = np.identity(3)):
    matrix_rotation = np.array(matrix_rotation)
    if isinstance(landmark,str):
        landmark = [landmark]

    data = json.load(open(os.path.join(path)))
    markups = data['markups']
    landmarks_lst = markups[0]['controlPoints']

    landmarks_pos = []
    # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
    for lm in landmarks_lst:
        label = lm["label"]
        if label in landmark:
            position = lm["position"]
            position = np.squeeze(np.matmul(matrix_rotation,np.expand_dims(position,0).T).T)
            position = Downscale(position,mean_arr,scale_factor)
            landmarks_pos.append(position)
    
    return landmarks_pos


def Downscale(pos_center,mean_arr,scale_factor):
    landmarks_position = (pos_center - mean_arr) / scale_factor
    return landmarks_position


def MeanScale(verts = None):

    min_coord = torch.min(verts,0)[0]
    max_coord= torch.max(verts,0)[0]
    mean = (max_coord + min_coord)/2.0
    mean= mean.numpy()
    scale = np.linalg.norm(max_coord.numpy() - mean)

    return mean, scale