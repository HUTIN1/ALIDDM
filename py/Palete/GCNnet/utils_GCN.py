import json
import vtk
import numpy as np
import torch
from torch import int64, tensor
import os
from vtk.util.numpy_support import vtk_to_numpy

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


def get_landmarks_position(path,landmarks, matrix):

        data = json.load(open(os.path.join(path)))
        markups = data['markups']
        landmarks_lst = markups[0]['controlPoints']

        landmarks_pos = []
        tmp_dic_landmark = {}
        for lm in landmarks_lst :
            tmp_dic_landmark[lm['label']] = lm['position']


        for landmark in landmarks :
                landmark_pos = np.matmul(matrix,np.append(tmp_dic_landmark[landmark],1).T).T
                landmarks_pos.append(landmark_pos[:3])        

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



def RemoveBase(surf=None,vertex=None):
    '''
    To use this function it mandatory to have oriented the surf/vertex and unitsurf
    '''
    if surf is not None :
        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData()))
    else :
        V = vertex

    mean = torch.mean(V,dim=0)


    arg = torch.argsort(V[...,0],dim=0)[10]

    new_tensor = []
    pos_max = V[arg]
    pos_max2 = pos_max[2]
    list_index = []
    for index , v in enumerate(V) :
        if v[2] > pos_max2 or  torch.dist(v,mean) < 0.4:
            new_tensor.append(v.unsqueeze(0))
            list_index.append(torch.tensor(index))

             
    new_tensor = torch.cat(new_tensor,dim=0)
    list_index = torch.tensor(list_index)

    new_tensor2 = []
    list_index2 = []
    arg = torch.argsort(new_tensor[...,2],dim=0)[0]
    minargs = new_tensor[arg,2] + torch.tensor(0.1)
    for index , v in zip(list_index,new_tensor):
        if v[2] > minargs:
            new_tensor2.append(v.unsqueeze(0))
            list_index2.append(index)

    new_tensor2 = torch.cat(new_tensor2,dim=0)
    list_index2 = torch.tensor(list_index2)
    return new_tensor2, list_index2


def MatrixScale(scale):
    return np.array([[scale,0,0,0],
                     [0, scale ,0 ,0],
                     [0, 0, scale ,0],
                     [0, 0, 0, 1]])

def MatrixTranspose(transpose):
    return np.array([[1 , 0, 0, transpose[0]],
                     [0, 1, 0, transpose[1]],
                     [0 ,0,1, transpose[2]],
                     [0, 0, 0, 1]])


def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)
    return surf


def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : np.array
        Axis of rotation
    theta : float
        Angle of rotation in radians
    
    Returns
    -------
    np.array
        Rotation matrix
    """

    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



def RotateTransform(surf, transform):

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()


def GetUnitSurf(surf, mean_arr = None, scale_factor = None):
  surf, surf_mean, surf_scale = ScaleSurf(surf, mean_arr, scale_factor)
  return surf, surf_mean, surf_scale



def ScaleSurf(surf, mean_arr = None, scale_factor = None):

    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy


    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)
    shape_points = np.array(shape_points)
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points_scaled = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    for i in range(shapedatapoints.GetNumberOfPoints()):
       shapedatapoints.SetPoint(i, shape_points_scaled[i])    

    surf.SetPoints(shapedatapoints)

    return surf, mean_arr, scale_factor