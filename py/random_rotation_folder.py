import os 
from utils import RandomRotation, ReadSurf
from ALIDDM_utils import get_landmarks_position2, WriteLandmark
import glob
from utils import WriteSurf
from torch import tensor,cos, sin,rand
import torch
from math import pi
from vtk.util.numpy_support import vtk_to_numpy
from vtk import vtkPoints
import numpy as np

def search(path,extension):
    out =[]
    files = glob.glob(os.path.join(path,extension))
    folders = os.listdir(path)
    for file in files:
        out.append(file)
    for folder  in folders:
        if os.path.isdir(os.path.join(path,folder)):
            out+=search(os.path.join(path,folder),extension)
    return out


def foundmountpoint(listfile):

    best = {0}
    first = listfile[0].split('/')
    listfile.pop(0)
    for file in listfile:
        dif = set(first)-set(file.split('/'))
        if len(dif)>len(best):
            best=dif 
    notmountpoint = list(set(first)-dif)[1:]
    mountpoint=[]
    for f in first:
        for n in notmountpoint:
            if f==n:
                mountpoint.append(f)
    mountpoint = '/'.join(mountpoint)
    mountpoint = '/'+mountpoint
    while not os.path.isdir(mountpoint):
        mountpoint, _ = os.path.split(mountpoint)


        
def RandomRotation(surf):
    alpha, beta , gamma  = np.random.random()*pi, np.random.random()*pi, np.random.random()*pi
    Rx = np.array([[1,0,0],[0,np.cos(alpha),np.sin(alpha)],[0,-np.sin(alpha),np.cos(alpha)]])
    Ry = np.array([[np.cos(beta),0,-np.sin(beta)],[0,1,0],[np.sin(beta),0,np.cos(beta)]])
    Rz = np.array([[np.cos(gamma),np.sin(gamma),0],[-np.sin(gamma),np.cos(gamma),0],[0,0,1]])

    matrix_rotation = np.matmul(Rx,Ry)
    matrix_rotation = np.matmul(matrix_rotation,Rz)

    vtkpoint = surf.GetPoints()
    points = vtk_to_numpy(vtkpoint.GetData())
    points = np.matmul(matrix_rotation,points.T).T


    vpoints= vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i,points[i])
    surf.SetPoints(vpoints)

    return surf, matrix_rotation
    


path = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data_random_rotation'

jsonfiles = search(path,'*.json')
vtkfiles = search(path,'*.vtk')
mount_point = foundmountpoint(jsonfiles)
files = []

for jsonfile in jsonfiles:
    jsonname, _ = os.path.splitext(os.path.basename(jsonfile))
    i = 0 
    stop = False
    while i<len(vtkfiles) and not stop:
        vtkname , _ = os.path.splitext(os.path.basename(vtkfiles[i]))
        if jsonname in vtkname:
            files.append([jsonfile,vtkfiles[i]])
            vtkfiles.pop(i)
            stop = True
        i+=1



for file in files :
    surf = ReadSurf(file[1])
    surf , matrix_rotation = RandomRotation(surf)
    dic_landmarkpos = get_landmarks_position2(file[0],matrix_rotation=matrix_rotation)
    WriteLandmark(dic_landmarkpos,file[0])
    WriteSurf(surf,file[1])

