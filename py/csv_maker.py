import pandas as pd
import os
import utils
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split

path = '/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/Denise'



files = utils.search(path,'.vtk','.json')
vtk_files = files['.vtk']
json_files = files['.json']
time_list = ['T1','T2','T3']

list_files = []

for vtk in tqdm(vtk_files,total = len(vtk_files) ) :
    vtk_name, _ = os.path.splitext(os.path.basename(vtk))
    vtk_folder = vtk.split('/')[-2]
    vtk_times = None
    for ti in time_list :
        if ti in vtk_name :
            vtk_times =ti
            vtk_name.replace(ti,'')
            break

    for json in json_files :
        json_name, _ = os.path.splitext(os.path.basename(json))
        json_folder = json.split('/')[-2]
        json_times = None
        for ti in time_list :
            if ti in json_name :
                json_times =ti
                json_name.replace(ti,'')
                break

        if json_name.replace('_out','') == vtk_name.replace('_out','') and  json_times ==vtk_times and json_folder== vtk_folder:
            vtk = vtk.replace(path,'')
            json = json.replace(path,'')
            list_files.append([vtk,json])
            break



for file in list_files:
    if 'Denise' in file[0]:
        print(f'denise file {file}')

train , val_test = train_test_split(list_files,test_size=0.3,shuffle=42)
# val, test = train_test_split(val_test, test_size=0.66,shuffle=42)
val = val_test



head = ['surf','landmark']


with open(os.path.join(path,'train_palete.csv'),'w') as f:
    writer = csv.writer(f)
    writer.writerow(head)
    writer.writerows(train)

with open(os.path.join(path,'val_palete.csv'),'w') as f:
    writer = csv.writer(f)
    writer.writerow(head)
    writer.writerows(val)

# with open(os.path.join(path,'test_palete.csv'),'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(head)
#     writer.writerows(test)