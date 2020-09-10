import os
from os.path import join
import scipy.io
import cv2
import numpy as np
from PIL import Image

path_trainval = '/home/yu/DataSet/PASCAL2010/trainval'
path_save = '/home/yu/DataSet/PASCAL2010/edges'
if not os.path.isdir(path_trainval):
    print("path trainval {} does not exists!".format(path_trainval))
if not os.path.isdir(path_save):
    os.makedirs(path_save)
    print("{} created for saving files.".format(path_save))

folders = ['train', 'test']
for folder in folders:
    print(folder)
    filelist = [file for file in os.listdir(join(path_trainval, folder)) if '.mat' in file]
    filelist.sort()
    numFiles = len(filelist)
    fmtStr = "{{:0>{}}}/{}".format(len(str(numFiles)), numFiles)
    count = 1
    for file in filelist:
        print(fmtStr.format(count))
        path_matfile = os.path.join(path_trainval, folder, file)
        mat = scipy.io.loadmat(path_matfile)
        xx = cv2.Sobel(mat['LabelMap'], cv2.CV_32F, 1, 0)
        yy = cv2.Sobel(mat['LabelMap'], cv2.CV_32F, 0, 1)

        absX = cv2.convertScaleAbs(xx)
        absY = cv2.convertScaleAbs(yy)

        edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        edge = edge != 0

        # path_npfile = os.path.join(path_save, folder, file[:-4])
        # np.save(path_npfile, edge)
        count += 1

        imgedge = Image.fromarray(edge)
        path_edgeFile = join(path_save, folder, file)
        path_edgeFile = path_edgeFile.replace('.mat', '.jpg')
        # print(path_edgeFile)
        if not os.path.isfile(path_edgeFile):
            imgedge.save(path_edgeFile)
        # break


