from utils import search
import os
from dataclasses import dataclass, asdict
from ALIDDM_utils import ReadLandmark
import numpy as np

@dataclass
class Patient:
    name : str
    groundthruth : str
    ali : str
    alinew :str


path_groundthruth = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Test_jaw/jaw'
path_ali = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Test_jaw/jaw_json_ali'
path_newali = '/home/luciacev/Desktop/Data/ALI_IOS/landmark/Prediction/Data/Test_jaw/jaw_json_12images'

dataset = []
error_ali =[]
error_alinew = []
landmark = 'LR6O'
list_groundthruth = search(path_groundthruth,'.json')['.json']
list_ali = search(path_ali,'.json')['.json']
list_alinew = search(path_newali,'.json')['.json']

print('Read File')

for file_groundthruth in list_groundthruth:
    name, _ = os.path.splitext(os.path.basename(file_groundthruth))
    ali=None
    alinew = None
    for file_ali in list_ali:
        alibasename = os.path.basename(file_ali).replace('.json','')
        if name == alibasename:
            ali = file_ali
            break
    for file_newali in list_alinew:
        alinewbasename = os.path.basename(file_newali).replace(f'_{landmark}','').replace('.json','')
        if name == alinewbasename:
            alinew = file_newali
            break
    if not (ali == None or alinew == None) :
        dataset.append(Patient(name,file_groundthruth,ali,alinew))

print('='*50)
print('Compute')
for data in dataset:
    data = asdict(data)
    groudthruth = ReadLandmark(data['groundthruth'],name=landmark)[landmark]
    ali = ReadLandmark(data['ali'],name=landmark)[landmark]
    alinew = ReadLandmark(data['alinew'],name=landmark)[landmark]

    distance_ali = np.linalg.norm(groudthruth - ali)
    distance_alinew = np.linalg.norm(groudthruth - alinew)
    print(f'Name :{data["name"]}, distance ali :{distance_ali}, distance ali new :{distance_alinew}')
    # print(f'ali: {ali} path :{data["ali"]}')
    # print(f'ali new: {alinew}, path: {data["alinew"]}')
    if True in np.isnan(alinew) or True in np.isnan(ali):
        print(f'There is an nan {data["name"]}')
    else :
        error_ali.append(distance_ali)
        error_alinew.append(distance_alinew)


    
print(f'error ali :{np.mean(np.array(error_ali))}')
print(f'error new ali: {np.mean(np.array(error_alinew))}')
