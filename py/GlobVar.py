import torch
import os
import numpy as np
from scipy import linalg

global DEVICE 
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

# ALL
id_lst = range(1,43)
# CUSTOM
# id_lst = [1,25]

for i in id_lst:
    Lower.append("Lower_O-"+str(i))
    Upper.append("Upper_O-"+str(i))

LANDMARKS = {
    "L":Lower,
    "U":Upper
}

global SELECTED_JAW
SELECTED_JAW = "L"

sphere_points = ([0,0,1],
                np.array([0.5,0.,1.0])/linalg.norm([0.5,0.5,1.0]),
                np.array([-0.5,0.,1.0])/linalg.norm([-0.5,-0.5,1.0]),
                np.array([0,0.5,1])/linalg.norm([1,0,1]),
                np.array([0,-0.5,1])/linalg.norm([0,1,1])
                )
global CAMERA_POSITION 
CAMERA_POSITION = np.array(sphere_points)

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




