import numpy as np
import vtk
from utils_GCN import RotationMatrix, TransformSurf
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

cross = lambda a,b: np.cross(a,b)

def make_vector(points2,point1):
    perpen = points2[1]-points2[0]
    perpen= perpen/np.linalg.norm(perpen)

    vector1 = points2[0] - point1
    vector1 = vector1/np.linalg.norm(vector1)

    vector2 = points2[1] - point1
    vector2 = vector2/np.linalg.norm(vector2)

    normal = cross(vector1,vector2)
    normal = normal/np.linalg.norm(normal)

    direction = cross(normal,perpen)
    direction = direction/np.linalg.norm(direction)
    return normal ,direction



def PrePreAso(source,target,landmarks):



    left =landmarks[0]
    middle1 = landmarks[1]
    middle2 = landmarks[2]
    right = landmarks[3]

         
    meanTeeth = vtkMeanTeeth([int(left),int(middle1),int(middle2),int(right)],property='PredictedID')
    mean_source = meanTeeth(source)

    left_source, middle1_source, middle2_source , right_source = mean_source[left], mean_source[middle1], mean_source[middle2],mean_source[right]
    left_target, middle_target , right_target = np.array(target[0]), np.array(target[1]), np.array(target[2])

    middle_source = (middle1_source + middle2_source) / 2


    normal_source, direction_source = make_vector([right_source,left_source],middle_source)
    normal_target , direction_target = make_vector([right_target,left_target],middle_target)



    dt = np.dot(normal_source,normal_target)
    if dt > 1.0 :
        dt = 1.0

    angle_normal = np.arccos(dt)


    normal_normal = cross(normal_source,normal_target)



    matrix_normal = RotationMatrix(normal_normal,angle_normal)
    
    
    
    direction_source = np.matmul(matrix_normal,direction_source.T).T
    direction_source = direction_source / np.linalg.norm(direction_source)

    
    direction_normal = cross(direction_source,direction_target)

    dt = np.dot(direction_source,direction_target)
    if dt > 1.0:
        dt = 1.0

    angle_direction = np.arccos(dt)
    matrix_direction = RotationMatrix(direction_normal ,angle_direction)



    matrix = np.matmul(matrix_direction, matrix_normal)

    left_source = np.matmul(matrix,left_source)   
    middle_source = np.matmul(matrix,middle_source)
    right_source = np.matmul(matrix,right_source)

    mean_source = np.mean(np.array([left_source,middle_source,right_source]),axis=0)
    mean_target = np.mean(np.array([left_target, middle_target, right_target]),axis=0)

    mean = (mean_target- mean_source)



    

    matrix = np.concatenate((matrix,np.array([mean]).T),axis=1)
    matrix = np.concatenate((matrix,np.array([[0,0,0,1]])),axis=0)


    # matrix = np.matmul(matrix,matrix_translation)



    



    output = vtk.vtkPolyData()
    output.DeepCopy(source)


    output = TransformSurf(output,matrix)



    return output , matrix






class vtkTeeth:
    def __init__(self,list_teeth,property =None):
        self.property = property
        self.list_teeth = list_teeth

    def CheckLabelSurface(self,surf,property):
        if not self.isLabelSurface(surf,property):
            property = self.GetLabelSurface(surf)
        self.property = property



    def GetLabelSurface(self,surf,Preference='Universal_ID'):
        out = None

        list_label = [surf.GetPointData().GetArrayName(i) for i in range(surf.GetPointData().GetNumberOfArrays())]

        if len(list_label)!=0 :
            for label in list_label:
                out = label
                if Preference == label:
                    out = Preference
                    continue
                    
        return out



    def isLabelSurface(self,surf,property):
        out = False
        list_label = [surf.GetPointData().GetArrayName(i) for i in range(surf.GetPointData().GetNumberOfArrays())]
        if property in list_label:
            out = True
        return out




class vtkIterTeeth(vtkTeeth):
    def __init__(self, list_teeth, surf, property=None):
        super().__init__(list_teeth, property)
        self.CheckLabelSurface(surf,property)
        if not self.isLabelSurface(surf,self.property):
            raise NoSegmentationSurf(self.property)
        self.region_id = vtk_to_numpy(surf.GetPointData().GetScalars(self.property))
        self.verts = vtk_to_numpy(surf.GetPoints().GetData())

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter >= len(self.list_teeth):
            raise StopIteration
        
        verts_crown = np.argwhere(self.region_id==self.list_teeth[self.iter])
        if len(verts_crown)== 0 :
            raise ToothNoExist(self.list_teeth[self.iter])

        self.iter += 1 
        return np.array(self.verts[verts_crown]) , self.list_teeth[self.iter-1]



class vtkMeanTeeth(vtkTeeth):
    def __init__(self, list_teeth, property=None):
        super().__init__(list_teeth, property)

    def __call__(self, surf) :
        dic ={}
        for points, tooth in vtkIterTeeth(self.list_teeth,surf,property=self.property):
            dic[str(tooth)]= np.array(np.mean(points,0).squeeze(0))
        return dic


class vtkMiddleTeeth(vtkTeeth):
    def __init__(self, list_teeth, property=None):
        super().__init__(list_teeth, property)

    def __call__(self,surf):
        dic ={} 
        for points, tooth in vtkIterTeeth(self.list_teeth,surf,property=self.property):
            dic[str(tooth)]= ((np.amax(points,axis=0)+np.amin(points,axis = 0))/2).squeeze(0)
        return dic


class vtkMeshTeeth(vtkTeeth):
    def __init__(self, list_teeth=None, property=None):
        super().__init__(list_teeth, property)
    def __call__(self,surf):
        self.CheckLabelSurface(surf,self.property)
        region_id = vtk_to_numpy(surf.GetPointData().GetScalars(self.property))
        list_teeth = np.unique(region_id)[1:-1]
        list_points = []
        size = 0

        for points, _ in  vtkIterTeeth(list_teeth,surf,property=self.property):
            list_points.append(points)
            size+= points.shape[0]

    

        Points = vtk.vtkPoints()
        Vertices = vtk.vtkCellArray()
        labels = vtk.vtkStringArray()
        labels.SetNumberOfValues(size)
        labels.SetName("labels")
        index = 0 
        for  points in list_points:
            for i in range(points.shape[0]):
                sp_id = Points.InsertNextPoint(points[i,:].squeeze(0))
                Vertices.InsertNextCell(1)
                Vertices.InsertCellPoint(sp_id)
                labels.SetValue(index, str(index))
                index+=1
            
        output = vtk.vtkPolyData()
        output.SetPoints(Points)
        output.SetVerts(Vertices)
        output.GetPointData().AddArray(labels)

        return output



class ToothNoExist(Exception):
    def __init__(self, tooth ) -> None:
        dic = {1: 'UR8', 2: 'UR7', 3: 'UR6', 4: 'UR5', 5: 'UR4', 6: 'UR3', 7: 'UR2', 8: 'UR1', 9: 'UL1', 10: 'UL2', 11: 'UL3',
         12: 'UL4', 13: 'UL5', 14: 'UL6', 15: 'UL7', 16: 'UL8', 17: 'LL8', 18: 'LL7', 19: 'LL6', 20: 'LL5', 21: 'LL4', 22: 'LL3', 
         23: 'LL2', 24: 'LL1', 25: 'LR1', 26: 'LR2', 27: 'LR3', 28: 'LR4', 29: 'LR5', 30: 'LR6', 31: 'LR7', 32: 'LR8'}
        if isinstance(tooth,int):
            tooth = dic[tooth]
        self.message =f'This tooth {tooth} is not segmented or doesnt exist '
        super().__init__(self.message)


    def __str__(self) -> str:
        return self.message



class NoSegmentationSurf(Exception):
    def __init__(self, property) -> None:
        self.message = f'This surf doesnt have this property {property}'
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message