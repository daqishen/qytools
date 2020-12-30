#-*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import argparse
#from progressbar import ProgressBar, Percentage, Bar, Timer ,ETA#, FileTransferSpeed


FILE_PATH = "./"
HEIGHT = 720
WIDTH = 1280
# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str,help='Image file path.', default=FILE_PATH)
parser.add_argument('-t', '--target', type=str,help='Target file path.', default='')
parser.add_argument('-f', '--format', type=str,help='Image format(tiff/jpeg).', default='tiff')
args = parser.parse_args()   

def read_and_convert(file_path,target_file,img_format):
    sizes=0;
    amounts=0;
    
    path_lists 	=  os.listdir(file_path)
    img_lists = []
    for file in path_lists:
        if file.endswith('.yuyv422'):
            img_lists.append(file) 
    n = len(img_lists)

    for i in range(n):
        timeStamp16_s = img_lists[i].split('_')[0]
        yuyv_file = os.path.join(file_path, img_lists[i])
        yuyv_data = np.fromfile(yuyv_file, dtype='u1')[0:HEIGHT*WIDTH*2].reshape(HEIGHT, WIDTH, -1)
        bgr_data = cv2.cvtColor(yuyv_data, cv2.COLOR_YUV2RGB_YVYU)
        if img_format == 'tiff':
            target_name = target_file+'/'+timeStamp16_s+'.tif'
        else:
            target_name = target_file+'/'+timeStamp16_s+'.jpg'
        cv2.imwrite(target_name,bgr_data)
        amounts+=1
        sizes+=os.path.getsize(target_name)
        if i%10 == 0:
            print("convert", i, " of ", n)


def main():
    masterPath = args.path             #this is the masterPath
    if args.target == '':
        mdDirPath  = masterPath+"/../convert" #this is the convert folder
    else:
        mdDirPath  = args.target+"/../convert"
    lists = os.listdir(masterPath)     #get every name in  the master folder
    if mdDirPath in lists:
        print ("文件夹已存在")
        try:
            os.rmdir(mdDirPath)
        except:
            pass
    try:
        os.mkdir(mdDirPath)
    except:
        print("创建文件夹失败")
        s = mdDirPath
        print("请删除"+s+"目录")
        print("退出程序")
        sys.exit()
        
    read_and_convert(masterPath,mdDirPath,args.format)
    

if __name__=="__main__":
    main()
'''
#%%
file_paths = ["/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/20-01/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/20-02/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/30-01/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/30-02/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/40-01/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/40-02/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/50-01/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/50-02/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/60-01/image_capturer_0",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/60-02/image_capturer_0"]
target_files = ["/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/20-01/image_capturer0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/20-02/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/30-01/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/30-02/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/40-01/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/40-02/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/50-01/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/50-02/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/60-01/image_capturer_0_jpg",
              "/home/qy/Desktop/project/Save_Crossing/data/benchmark/20200806/moving_car_01/dump_images/60-02/image_capturer_0_jpg"]
for ele in target_files:
    if not os.path.exists(ele):
        os.mkdir(ele)
    else:
        print("path exist: ", ele)

img_format = 'jpeg'

for i in range(len(file_paths)):
    print("start process folder: ", i, " of", len(file_paths))
    read_and_convert(file_paths[i], target_files[i], img_format)

'''
