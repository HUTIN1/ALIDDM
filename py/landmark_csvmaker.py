import glob
import os
from sklearn.model_selection import train_test_split
import argparse
import json
import csv



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

    return mountpoint




def checklandmarkinjson(path,landmark):
    data = json.load(open(os.path.join(path)))
    markups = data['markups']
    landmarks_lst = markups[0]['controlPoints']

    # resc_landmarks_position = np.zeros([number_of_landmarks, 3])
    for lm in landmarks_lst:
        label = lm["label"]
        if label == landmark:
            return True
    
    return False




def main(args):
    list_landmark = ['UL7CL', 'UL7CB', 'UL7O', 'UL7DB', 'UL7MB', 'UL7R', 'UL7RIP', 'UL7OIP', 'UL6CL', 'UL6CB', 'UL6O', 'UL6DB', 'UL6MB', 'UL6R', 
    'UL6RIP', 'UL6OIP', 'UL5CL', 'UL5CB', 'UL5O', 'UL5DB', 'UL5MB', 'UL5R', 'UL5RIP', 'UL5OIP', 'UL4CL', 'UL4CB', 'UL4O', 'UL4DB', 'UL4MB', 'UL4R', 
    'UL4RIP', 'UL4OIP', 'UL3CL', 'UL3CB', 'UL3O', 'UL3DB', 'UL3MB', 'UL3R', 'UL3RIP', 'UL3OIP', 'UL2CL', 'UL2CB', 'UL2O', 'UL2DB', 'UL2MB', 'UL2R', 
    'UL2RIP', 'UL2OIP', 'UL1CL', 'UL1CB', 'UL1O', 'UL1DB', 'UL1MB', 'UL1R', 'UL1RIP', 'UL1OIP', 'UR1CL', 'UR1CB', 'UR1O', 'UR1DB', 'UR1MB', 'UR1R', 
    'UR1RIP', 'UR1OIP', 'UR2CL', 'UR2CB', 'UR2O', 'UR2DB', 'UR2MB', 'UR2R', 'UR2RIP', 'UR2OIP', 'UR3CL', 'UR3CB', 'UR3O', 'UR3DB', 'UR3MB', 'UR3R', 
    'UR3RIP', 'UR3OIP', 'UR4CL', 'UR4CB', 'UR4O', 'UR4DB', 'UR4MB', 'UR4R', 'UR4RIP', 'UR4OIP', 'UR5CL', 'UR5CB', 'UR5O', 'UR5DB', 'UR5MB', 'UR5R', 
    'UR5RIP', 'UR5OIP', 'UR6CL', 'UR6CB', 'UR6O', 'UR6DB', 'UR6MB', 'UR6R', 'UR6RIP', 'UR6OIP', 'UR7CL', 'UR7CB', 'UR7O', 'UR7DB', 'UR7MB', 'UR7R', 
    'UR7RIP', 'UR7OIP', 'LL7CL', 'LL7CB', 'LL7O', 'LL7DB', 'LL7MB', 'LL7R', 'LL7RIP', 'LL7OIP', 'LL6CL', 'LL6CB', 'LL6O', 'LL6DB', 'LL6MB', 'LL6R', 
    'LL6RIP', 'LL6OIP', 'LL5CL', 'LL5CB', 'LL5O', 'LL5DB', 'LL5MB', 'LL5R', 'LL5RIP', 'LL5OIP', 'LL4CL', 'LL4CB', 'LL4O', 'LL4DB', 'LL4MB', 'LL4R', 
    'LL4RIP', 'LL4OIP', 'LL3CL', 'LL3CB', 'LL3O', 'LL3DB', 'LL3MB', 'LL3R', 'LL3RIP', 'LL3OIP', 'LL2CL', 'LL2CB', 'LL2O', 'LL2DB', 'LL2MB', 'LL2R', 
    'LL2RIP', 'LL2OIP', 'LL1CL', 'LL1CB', 'LL1O', 'LL1DB', 'LL1MB', 'LL1R', 'LL1RIP', 'LL1OIP', 'LR1CL', 'LR1CB', 'LR1O', 'LR1DB', 'LR1MB', 'LR1R', 
    'LR1RIP', 'LR1OIP', 'LR2CL', 'LR2CB', 'LR2O', 'LR2DB', 'LR2MB', 'LR2R', 'LR2RIP', 'LR2OIP', 'LR3CL', 'LR3CB', 'LR3O', 'LR3DB', 'LR3MB', 'LR3R', 
    'LR3RIP', 'LR3OIP', 'LR4CL', 'LR4CB', 'LR4O', 'LR4DB', 'LR4MB', 'LR4R', 'LR4RIP', 'LR4OIP', 'LR5CL', 'LR5CB', 'LR5O', 'LR5DB', 'LR5MB', 'LR5R', 
    'LR5RIP', 'LR5OIP', 'LR6CL', 'LR6CB', 'LR6O', 'LR6DB', 'LR6MB', 'LR6R', 'LR6RIP', 'LR6OIP', 'LR7CL', 'LR7CB', 'LR7O', 'LR7DB', 'LR7MB', 'LR7R', 'LR7RIP', 'LR7OIP']





    path = args.path



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


    for landmark in list_landmark:
        listtest=[]
        for file in files :
            if checklandmarkinjson(file[0],landmark):
                listtest.append([file[0][len(mount_point)+1:],file[1][len(mount_point)+1:]])

        if  not len(listtest) == 0:
            train, val_test = train_test_split(listtest,test_size=0.3,shuffle=42)
            val, test = train_test_split(val_test,test_size=0.66,shuffle=42)
            

            head = ['landmark','surf']
            with open(os.path.join(args.out,f'train_{landmark}.csv'),'w') as f :
                writer = csv.writer(f)

                writer.writerow(head)
                writer.writerows(train)

            with open(os.path.join(args.out,f'val_{landmark}.csv'),'w') as f :
                writer = csv.writer(f)

                writer.writerow(head)
                writer.writerows(val)

            with open(os.path.join(args.out,f'test_{landmark}.csv'),'w') as f :
                writer = csv.writer(f)

                writer.writerow(head)
                writer.writerows(test)


    # train, val_test = train_test_split(files,test_size=0.3,shuffle=42)
    # val, test = train_test_split(val_test,test_size=0.66,shuffle=42)


    # print(test)





















if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',default='/home/luciacev/Desktop/Data/ALI_IOS/landmark')
    parser.add_argument('--out',default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/csv')


    args = parser.parse_args()

    main(args)