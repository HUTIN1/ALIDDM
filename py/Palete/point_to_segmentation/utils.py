import vtk
import numpy as np
import math 
import os
import sys

import pandas as pd
from multiprocessing import Pool, cpu_count
from vtk.util.numpy_support import vtk_to_numpy
import torch
from monai.transforms import (
    ToTensor
)
import glob
from torch import tensor
from vtk import vtkMatrix4x4, vtkMatrix3x3, vtkPoints
import json
import matplotlib.pyplot as plt


def search(path,*args):
    """
    Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

    Example:
    args = ('json',['.nii.gz','.nrrd'])
    return:
        {
            'json' : ['path/a.json', 'path/b.json','path/c.json'],
            '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
            '.nrrd.gz' : ['path/c.nrrd']
        }
    """
    arguments=[]
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}


def Downscale(pos_center,mean_arr,scale_factor):
    landmarks_position = (pos_center - mean_arr) * scale_factor
    return landmarks_position





def ReadSurf(fileName):

    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()    
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(fileName)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                obj_import.SetTexturePath(textures_path)
            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
            
            append.Update()
            surf = append.GetOutput()
            
        else:
            reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            surf = reader.GetOutput()

    return surf

def WriteSurf(surf, fileName):
    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    # print("Writing:", fileName)
    if extension == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif extension == ".stl":
        writer = vtk.vtkSTLWriter()

    writer.SetFileName(fileName)
    writer.SetInputData(surf)
    writer.Update()




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

def GetActor(surf):
    surfMapper = vtk.vtkPolyDataMapper()
    surfMapper.SetInputData(surf)

    surfActor = vtk.vtkActor()
    surfActor.SetMapper(surfMapper)


    return surfActor
def GetTransform(rotationAngle, rotationVector):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
    return transform

def RotateSurf(surf, rotationAngle, rotationVector):
    transform = GetTransform(rotationAngle, rotationVector)
    return RotateTransform(surf, transform)

def RotateInverse(surf, rotationAngle, rotationVector):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
   
    transform_i = vtk.vtkTransform()
    m_inverse = vtk.vtkMatrix4x4()
    transform.GetInverse(m_inverse)
    transform_i.SetMatrix(m_inverse)

    return RotateTransform(surf, transform_i)

def RotateTransform(surf, transform):

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()

def RotateNpTransform(surf, angle, np_transform):
    np_tran = np.load(np_transform)

    rotationAngle = -angle
    rotationVector = np_tran
    return RotateInverse(surf, rotationAngle, rotationVector)

def RandomRotation(surf):
    rotationAngle = np.random.random()*360.0
    rotationVector = np.random.random(3)*2.0 - 1.0
    rotationVector = rotationVector/np.linalg.norm(rotationVector)
    return RotateSurf(surf, rotationAngle, rotationVector), rotationAngle, rotationVector

def GetUnitSurf(surf, mean_arr = None, scale_factor = None):
  surf, surf_mean, surf_scale = ScaleSurf(surf, mean_arr, scale_factor)
  return surf, surf_mean, surf_scale

def GetColoredActor(surf, property_name, range_scalars = None):

    if range_scalars == None:
        range_scalars = surf.GetPointData().GetScalars(property_name).GetRange()

    hueLut = vtk.vtkLookupTable()
    hueLut.SetTableRange(0, range_scalars[1])
    hueLut.SetHueRange(0.0, 0.9)
    hueLut.SetSaturationRange(1.0, 1.0)
    hueLut.SetValueRange(1.0, 1.0)
    hueLut.Build()

    surf.GetPointData().SetActiveScalars(property_name)

    actor = GetActor(surf)
    actor.GetMapper().ScalarVisibilityOn()
    actor.GetMapper().SetScalarModeToUsePointData()
    actor.GetMapper().SetColorModeToMapScalars()
    actor.GetMapper().SetUseLookupTableScalarRange(True)

    actor.GetMapper().SetLookupTable(hueLut)

    return actor


def GetPropertyActor(surf, property_name):

    #display property on surface
    point_data = vtk.vtkDoubleArray()
    point_data.SetNumberOfComponents(1)

    with open(property_name) as property_file:
        for line in property_file:
            point_val = float(line[:-1])
            point_data.InsertNextTuple([point_val])
                
        surf.GetPointData().SetScalars(point_data)

    surf_actor = GetActor(surf)
    surf_actor.GetProperty().LightingOff()
    surf_actor.GetProperty().ShadingOff()
    surf_actor.GetProperty().SetInterpolationToFlat()

    surfMapper = surf_actor.GetMapper()
    surfMapper.SetUseLookupTableScalarRange(True)

    
    #build lookup table
    number_of_colors = 512
    low_range = 0
    high_range = 1  
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(low_range, high_range)
    lut.SetNumberOfColors(number_of_colors)

    #Color transfer function  
    ctransfer = vtk.vtkColorTransferFunction()
    ctransfer.AddRGBPoint(0.0, 1.0, 1.0, 0.0) # Yellow
    ctransfer.AddRGBPoint(0.5, 1.0, 0.0, 0.0) # Red

    #Calculated new colors for LUT via color transfer function
    for i in range(number_of_colors):
        new_colour = ctransfer.GetColor( (i * ((high_range-low_range)/number_of_colors) ) )
        lut.SetTableValue(i, *new_colour)

    lut.Build()

    surfMapper.SetLookupTable(lut)

    # return surfActor
    return surfMapper

def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOff();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()

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

def GetNormalsActor(surf):
    try:
        
        surf = ComputeNormals(surf)
        # mapper
        surf_actor = GetActor(surf)

        if vtk.VTK_MAJOR_VERSION > 8:

            sp = surf_actor.GetShaderProperty();
            sp.AddVertexShaderReplacement(
                "//VTK::Normal::Dec",
                True,
                "//VTK::Normal::Dec\n" + 
                "  varying vec3 myNormalMCVSOutput;\n",
                False
            )

            sp.AddVertexShaderReplacement(
                "//VTK::Normal::Impl",
                True,
                "//VTK::Normal::Impl\n" +
                "  myNormalMCVSOutput = normalMC;\n",
                False
            )

            sp.AddVertexShaderReplacement(
                "//VTK::Color::Impl",
                True, "VTK::Color::Impl\n", False)

            sp.ClearVertexShaderReplacement("//VTK::Color::Impl", True)

            sp.AddFragmentShaderReplacement(
                "//VTK::Normal::Dec",
                True,
                "//VTK::Normal::Dec\n" + 
                "  varying vec3 myNormalMCVSOutput;\n",
                False
            )

            sp.AddFragmentShaderReplacement(
                "//VTK::Light::Impl",
                True,
                "//VTK::Light::Impl\n" +
                "  gl_FragData[0] = vec4(myNormalMCVSOutput*0.5f + 0.5, 1.0);\n",
                False
            )

        else:
            
            colored_points = GetColorArray(surf, "Normals")
            surf.GetPointData().SetScalars(colored_points)

            surf_actor = GetActor(surf)
            surf_actor.GetProperty().LightingOff()
            surf_actor.GetProperty().ShadingOff()
            surf_actor.GetProperty().SetInterpolationToFlat()


        return surf_actor
    except Exception as e:
        print(e, file=sys.stderr)
        return None

def GetCellIdMapActor(surf):

    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('cell_ids')
    colored_points.SetNumberOfComponents(3)

    for cell_id in range(0, surf.GetNumberOfCells()):
        r = cell_id % 255.0 + 1
        g = int(cell_id / 255.0) % 255.0
        b = int(int(cell_id / 255.0) / 255.0) % 255.0
        colored_points.InsertNextTuple3(r, g, b)

        # cell_id_color = int(b*255*255 + g*255 + r - 1)

    surf.GetCellData().SetScalars(colored_points)

    surf_actor = GetActor(surf)
    surf_actor.GetMapper().SetScalarModeToUseCellData()
    surf_actor.GetProperty().LightingOff()
    surf_actor.GetProperty().ShadingOff()
    surf_actor.GetProperty().SetInterpolationToFlat()

    return surf_actor

def GetPointIdColors(surf):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('point_ids')
    colored_points.SetNumberOfComponents(3)

    for cell_id in range(0, surf.GetNumberOfCells()):

        point_ids = vtk.vtkIdList()
        surf.GetCellPoints(cell_id, point_ids)

        point_id = point_ids.GetId(0)

        r = point_id % 255.0 + 1
        g = int(point_id / 255.0) % 255.0
        b = int(int(point_id / 255.0) / 255.0) % 255.0
        colored_points.InsertNextTuple3(r, g, b)
    
    return colored_points

def GetPointIdMapActor(surf):

    colored_points = GetPointIdColors(surf)

    # cell_id_color = int(b*255*255 + g*255 + r - 1)

    surf.GetCellData().SetScalars(colored_points)

    surf_actor = GetActor(surf)
    surf_actor.GetMapper().SetScalarModeToUseCellData()
    surf_actor.GetProperty().LightingOff()
    surf_actor.GetProperty().ShadingOff()
    surf_actor.GetProperty().SetInterpolationToFlat()

    return surf_actor
  
class ExtractPointFeaturesClass():
    def __init__(self, point_features_np, zero):
        self.point_features_np = point_features_np
        self.zero = zero

    def __call__(self, point_ids_rgb):

        point_ids_rgb = point_ids_rgb.reshape(-1, 3)
        point_features = []

        for point_id_rgb in point_ids_rgb:
            r = point_id_rgb[0]
            g = point_id_rgb[1]
            b = point_id_rgb[2]

            point_id = int(b*255*255 + g*255 + r - 1)

            point_features_np_shape = np.shape(self.point_features_np)
            if point_id >= 0 and point_id < point_features_np_shape[0]:
                point_features.append([self.point_features_np[point_id]])
            else:
                point_features.append(self.zero)

        return point_features

def ExtractPointFeatures(surf, point_ids_rgb, point_features_name, zero=0, use_multi=True):

    point_ids_rgb_shape = point_ids_rgb.shape

    if point_features_name == "coords" or point_features_name == "points":
        points = surf.GetPoints()
        point_features_np = vtk_to_numpy(points.GetData())
        number_of_components = 3
    else:    
        point_features = surf.GetPointData().GetScalars(point_features_name)
        point_features_np = vtk_to_numpy(point_features)
        number_of_components = point_features.GetNumberOfComponents()
    
    zero = np.zeros(number_of_components) + zero

    if use_multi:
        with Pool(cpu_count()) as p:
        	feat = p.map(ExtractPointFeaturesClass(point_features_np, zero), point_ids_rgb)
    else:
        feat = ExtractPointFeaturesClass(point_features_np, zero)(point_ids_rgb)
    return np.array(feat).reshape(point_ids_rgb_shape[0:-1] + (number_of_components,))

def ReadImage(fName, image_dimension=2, pixel_dimension=-1):
    if(image_dimension == 1):
        if(pixel_dimension != -1):
            ImageType = itk.Image[itk.Vector[itk.F, pixel_dimension], 2]
        else:
            ImageType = itk.VectorImage[itk.F, 2]
    else:
        if(pixel_dimension != -1):
            ImageType = itk.Image[itk.Vector[itk.F, pixel_dimension], image_dimension]
        else:
            ImageType = itk.VectorImage[itk.F, image_dimension]

    img_read = itk.ImageFileReader[ImageType].New(FileName=fName)
    img_read.Update()
    img = img_read.GetOutput()

    return img

def GetImage(img_np):

    img_np_shape = np.shape(img_np)
    ComponentType = itk.ctype('float')

    Dimension = img_np.ndim - 1
    PixelDimension = img_np.shape[-1]
    print("Dimension:", Dimension, "PixelDimension:", PixelDimension)

    if Dimension == 1:
        OutputImageType = itk.VectorImage[ComponentType, 2]
    else:
        OutputImageType = itk.VectorImage[ComponentType, Dimension]
    
    out_img = OutputImageType.New()
    out_img.SetNumberOfComponentsPerPixel(PixelDimension)

    size = itk.Size[OutputImageType.GetImageDimension()]()
    size.Fill(1)
    
    prediction_shape = list(img_np.shape[0:-1])
    prediction_shape.reverse()


    if Dimension == 1:
        size[1] = prediction_shape[0]
    else:
        for i, s in enumerate(prediction_shape):
            size[i] = s

    index = itk.Index[OutputImageType.GetImageDimension()]()
    index.Fill(0)

    RegionType = itk.ImageRegion[OutputImageType.GetImageDimension()]
    region = RegionType()
    region.SetIndex(index)
    region.SetSize(size)

    out_img.SetRegions(region)
    out_img.Allocate()

    out_img_np = itk.GetArrayViewFromImage(out_img)
    out_img_np.setfield(img_np.reshape(out_img_np.shape), out_img_np.dtype)

    return out_img

def GetTubeFilter(vtkpolydata):

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetNumberOfSides(50)
    tubeFilter.SetRadius(0.01)
    tubeFilter.SetInputData(vtkpolydata)
    tubeFilter.Update()

    return tubeFilter.GetOutput()

def ExtractFiber(surf, list_random_id) :
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    ids.InsertNextValue(list_random_id) 

    # extract a subset from a dataset
    selectionNode = vtk.vtkSelectionNode() 
    selectionNode.SetFieldType(0)
    selectionNode.SetContentType(4)
    selectionNode.SetSelectionList(ids) 

    # set containing cell to 1 = extract cell
    selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1) 

    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    # extract the cell from the cluster
    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, surf)
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()

    # convert the extract cell to a polygonal type (a line here)
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(extractSelection.GetOutput())
    geometryFilter.Update()


    tubefilter = GetTubeFilter(geometryFilter.GetOutput())

    return tubefilter

def Write(vtkdata, output_name):
    outfilename = output_name
    print("Writting:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()

def json2vtk(jsonfile,number_landmarks,radius_sphere,outdir):
    
    json_file = pd.read_json(jsonfile)
    json_file.head()
    markups = json_file.loc[0,'markups']
    controlPoints = markups['controlPoints']
    number_landmarks = len(controlPoints)
    L_landmark_position = []
    
    for i in range(number_landmarks):
        L_landmark_position.append(controlPoints[i]["position"])
        # Create a sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(L_landmark_position[i][0],L_landmark_position[i][1],L_landmark_position[i][2])
        sphereSource.SetRadius(radius_sphere)

        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)
        sphereSource.Update()
        vtk_landmarks = vtk.vtkAppendPolyData()
        vtk_landmarks.AddInputData(sphereSource.GetOutput())
        vtk_landmarks.Update()

        basename = os.path.basename(jsonfile).split(".")[0]
        filename = basename + "_landmarks.vtk"
        output = os.path.join(outdir, filename)
        Write(vtk_landmarks.GetOutput(), output)
    return output
    
def ArrayToTensor(vtkarray):
    return ToTensor(dtype=torch.int64)(vtk_to_numpy(vtkarray))

def PolyDataToTensors(surf):

    verts = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData()),dtype=torch.float32)
    faces = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:],dtype=torch.float32)
    
    return verts, faces


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



def pos_landmard2texture(vertex,landmark_pos):
    texture = torch.zeros_like(vertex.unsqueeze(0))
    vertex = vertex.to(torch.float64)
    radius = 0.04


    landmark_pos = tensor(np.array(landmark_pos)).unsqueeze(0)
    distance = torch.cdist(landmark_pos,vertex,p=2)
    minvalue = torch.min(distance)
    distance = distance - minvalue
    #print(min(distance,1))
    _, index_pos_land = torch.nonzero((distance<radius),as_tuple=True)

    texture[0,index_pos_land,1]=255

    return texture

def pos_landmard2seg(vertex,landmark_pos):
    texture = torch.zeros(size=(vertex.shape[0],))
    vertex = vertex.to(torch.float64)
    radius = 0.04


    landmark_pos = tensor(np.array(landmark_pos)).unsqueeze(0)
    distance = torch.cdist(landmark_pos,vertex,p=2)
    minvalue = torch.min(distance)
    distance = distance - minvalue
    _, index_pos_land = torch.nonzero((distance<radius),as_tuple=True)
    for i in index_pos_land:

        texture[i]=1
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

def MeanScale(surf =None ,verts = None):
    if surf : 
        verts = tensor(vtk_to_numpy(surf.GetPoints().GetData()))

    min_coord = torch.min(verts,0)[0]
    max_coord= torch.max(verts,0)[0]
    mean = (max_coord + min_coord)/2.0
    mean= mean.numpy()
    scale = np.linalg.norm(max_coord.numpy() - mean)

    return mean, scale, surf


def TransformVTK(surf,mean,scale):
    scale = np.double(scale)
    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())
    points = (points-mean)/scale

    vpoints= vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])
    surf.SetPoints(vpoints)

    return surf

def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)

    return surf


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




def normalize_points(poly, radius):
    polypoints = poly.GetPoints()
    for pid in range(polypoints.GetNumberOfPoints()):
        spoint = polypoints.GetPoint(pid)
        spoint = np.array(spoint)
        norm = np.linalg.norm(spoint)
        spoint = spoint/norm * radius
        polypoints.SetPoint(pid, spoint)
    poly.SetPoints(polypoints)
    return poly




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