
from vtk import vtkPolyDataNormals,vtkPolyDataReader, vtkPolyData, vtkPlatonicSolidSource, vtkPolyDataMapper, vtkActor, vtkLookupTable
import os

import numpy as np
import LinearSubdivisionFilter as lsf

def ComputeNormals(surf):
    normals = vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOff();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()


def ReadSurf(fileName):
    fname, extension = os.path.splitext(fileName)
    reader = vtkPolyDataReader()
    reader.SetFileName(fileName)
    reader.Update()
    surf = reader.GetOutput()
    return surf

def ScaleSurf(surf, mean_arr = None, scale_factor = None):
    surf_copy = vtkPolyData()
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

def ComputeNormals(surf):
    normals = vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOff();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()


def Boundingbox(surf, mean_arr = None, scale_factor = None, copy=True):
    surf_copy = vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)
    shape_points = np.array(shape_points)
    

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

    return surf



def CreateIcosahedron(radius, sl=0):
    icosahedronsource = vtkPlatonicSolidSource()
    icosahedronsource.SetSolidTypeToIcosahedron()
    icosahedronsource.Update()
    icosahedron = icosahedronsource.GetOutput()
    
    subdivfilter = lsf.LinearSubdivisionFilter()
    subdivfilter.SetInputData(icosahedron)
    subdivfilter.SetNumberOfSubdivisions(sl)
    subdivfilter.Update()

    icosahedron = subdivfilter.GetOutput()
    icosahedron = normalize_points(icosahedron, radius)

    return icosahedron




def GetActor(surf):
    surfMapper = vtkPolyDataMapper()
    surfMapper.SetInputData(surf)

    surfActor = vtkActor()
    surfActor.SetMapper(surfMapper)





def GetColoredActor(surf, property_name, range_scalars = None):

    if range_scalars == None:
        range_scalars = surf.GetPointData().GetScalars(property_name).GetRange()

    hueLut = vtkLookupTable()
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
