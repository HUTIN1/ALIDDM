import glob 
import os
import sys
import torch
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,'..','..')
sys.path.append(mymodule_dir)
from utils import ReadSurf

def search(path,type):
    out=[]
    # path  = os.path.join(path,'*'+type)
    a = glob.glob(path+'/*'+type)
    for p in a: 
        if os.path.isfile(p):
            out.append(p)
        else:
            out+= search(path=p,type=type)
    return out




path = '/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/challenge_teeth_all_vtk/train'

paths = search(path,'vtk')

listlen=[]
for path in paths:
    surf = ReadSurf(path)
    region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars('PredictedID'))),dtype=torch.int64)
    unique_ids = torch.unique(region_id).cpu().numpy().tolist()
    listlen.append(len(unique_ids))



print(min(listlen))

    