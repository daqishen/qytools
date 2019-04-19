#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:57:17 2019

@author: qy
"""

import pandas as pd
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Help:
    def __init__(self):
        
        return

    def using_tutorial():
        """
        Update at Sat Apr 13 10:34:02 2019
        
        Depend libraries:
            
        python -m pip install numpy 
        python -m pip install matplotlib    
        python -m pip install pandas
        python -m pip install opencv-contrib-python 
        
        
        v0.1.0 support different cv test mode
        currently there are 2 modes
        
        ---------------------------------------------------------------------------------------
        *** Mode 0 ***
        useage:
        python CVMatchTest_v0.1.0.py img1 file1 img2 file2 0
        
        Mode 0 will display the bf match result in 2 test files
        "number of query keypoints" : the number of query keypoints
        "number of BF matches" :      the number of matches pairs process by brute-force matching
        "number of good matches" :    the number of matches pairs process by brute-force matching
                                      and hamming distance thresholding (hamming distance <=25)
        ---------------------------------------------------------------------------------------
        *** Mode 1 ***
        useage:
        python CVMatchTest_v0.1.0.py img1 file1 img2 file2 1
        
        Mode 1 will display the bf match result in 2 test files
        "number of query keypoints" : the number of query keypoints
        "number of BF matches" :      the number of matches pairs process by brute-force matching
        "number of good matches" :    the number of matches pairs process by brute-force matching
                                      and point location distance thresholding (L1 distance <50)      
        "x_diff mean" :               mean value in x direction difference (x2-x1)
        "x_diff stddev" :             standard deviation in x direction difference
        "y_diff mean" :               mean value in y direction difference (y2-y1)
        "y_diff stddev" :             standard deviation in y direction difference      
        
        !CARE!: Mode 1 is used for testing the ABSOLUTE location difference in same image with difference algorithms.
                The number of good matches is strongly depended by the keypoint location.
                It two images have items shift, the result will be changed a lot
                You can change the illuminance in different images, the result is not very sensitive with the hamming
                distance in descriptors pairs.
                
                
        
        
        
        -------------------------------------------------------------------------------------
        
        @author: qy
        """
        return
    def howToUseThisFile(self):
        
        return help(self.using_tutorial)



class CVTest:
    
    
    def __init__(self):
        
        
        
        return
    
    def readfile(self, filename):
        '''
        read file from a fix format
        some operation might be stupid, shoud be refined
        '''
        
        df = pd.read_csv(filename,header = None, error_bad_lines=False)
        df_kpinfo = df[0].str.split(";", expand = True)
        df_kp = df_kpinfo[[i for i in range(df_kpinfo.shape[1]-1)]].copy()
        
        # get descriptors data frame
        df_desc = df.copy()
        df_desc[0] = df_kpinfo[[df_kpinfo.shape[1]-1]].copy()
        df_desc[0] = df_desc[0].str.split("[", expand = True)[1]
        df_desc[0] = df_desc[0].astype("int")
        df_desc[df_desc.shape[1]-1] = df_desc[df_desc.shape[1]-1].str.split("]", expand = True)[0]
        df_desc[df_desc.shape[1]-1] = df_desc[df_desc.shape[1]-1].astype("int")
        
        # get keypoints data frame
        df_kp.rename(columns={0:'x', 1:'y', 2:'angle', 3:'octave', 4:'class_id', 
                              5:'response', 6:'size', 7:'kp_desc_type'}, inplace = True)
        df_kp = df_kp.astype("float")
        df_kp['octave'] = df_kp['octave'].astype("int")
        df_kp['class_id'] = df_kp['class_id'].astype("int")
        
        df_desc = df_desc.astype("uint8")
        
        kps = []
        des = []
        
        n = df.shape[0]
        for i in range(n):
            kp = cv2.KeyPoint(df_kp['x'][i],df_kp['y'][i],df_kp['angle'][i], 
                              df_kp['response'][i],df_kp['octave'][i],df_kp['class_id'][i])
            kp.octave = int(df_kp['octave'][i])
            kps.append(kp)
        
        for i in range(n):
            desc = []
            for j in range(df_desc.shape[1]):
                desc.append(df_desc[j][i])
            desc = np.array(desc)
            des.append(desc)
        des = np.array(des)    
    
        return kps, des
        
        
    
    def KPmatch(self, kp1, des1, kp2, des2, thr = 25, kpmatchType = 0):
        '''
        matching keypoints by descriptors
        
        matchType = 0 :brute force
        
        '''

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        goodMatches = []
        if kpmatchType == 0:
        
            for match in matches:
                if match.distance > thr:
                    continue
                else:  
                    goodMatches.append(match)
            return matches, goodMatches 
                
        
        if kpmatchType == 1:
            
            x_diff = []
            y_diff = []
            for match in matches:
                q = match.queryIdx
                t = match.trainIdx
                x1, y1 = kp1[q].pt
                x2, y2 = kp2[t].pt
                d_L1 = abs(x1-x2)+abs(y1-y2)
#                if d_L1 > 50 or match.distance > thr:
                if d_L1 > 50:
                    continue
                else:
                    x_d = x1-x2
                    y_d = y1-y2
                    x_diff.append(x_d)
                    y_diff.append(y_d)
                    goodMatches.append(match)       
            print("x_diff mean: " + str(np.mean(x_diff)))
            print("x_dfff stddev: "+ str(np.std(x_diff)))
            print("y_diff mean: " + str(np.mean(y_diff)))
            print("y_dfff stddev: "+ str(np.std(y_diff)))             
            return matches, goodMatches
                
        
        return matches, goodMatches    
    
    
    
    def KPmatches_BF(self, filename_1, filename_2, img1, img2, matchtype):

        kp1, des1 = self.readfile(filename_1)
        kp2, des2 = self.readfile(filename_2)
        matches, goodMatches = self.KPmatch(kp1, des1, kp2, des2, kpmatchType = matchtype) 
        img3 = img1.copy()
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,goodMatches, img3, flags=2)
        plt.imshow(img3)
        plt.title("number of query keypoints: " + str(len(kp1))+"\n"+
                  "number of BF matches: " + str(len(matches))+"\n"+
                  "number of good matches: " + str(len(goodMatches)))
        
        
        plt.show()
        cv2.imwrite(filename_1+"KPmatches_BF.jpg", img3)
        print("number of query keypoints: " + str(len(kp1)))
        print("number of BF matches: " + str(len(matches)))
        print("number of good matches: " + str(len(goodMatches)))
        return
        
        

        

    
    def MatchParisTest(self, matchPairs, matchParis_2):
        
        n_goodmatchpairs = 0
        
        return n_goodmatchpairs
    
    
   
def ModrProcess(argv):
#    print(argv)
    
    if len(argv)<=1:
        Help().howToUseThisFile()
        return
    MODETYPE = ['0','1']
    if argv[-1] == '0':
        #mode 0 
        
        img_name_1 = sys.argv[1]
        filename_1 = sys.argv[2]
        img_name_2 = sys.argv[3]
        filename_2 = sys.argv[4]        
        img1 = cv2.imread(img_name_1,0)
        img2 = cv2.imread(img_name_2,0)
        CVTEST = CVTest()
        CVTEST.KPmatches_BF(filename_1,filename_2,img1,img2,0)      
        return
    if argv[-1] == '1' :
        #mode 1
        img_name_1 = sys.argv[1]
        filename_1 = sys.argv[2]
        img_name_2 = sys.argv[3]
        filename_2 = sys.argv[4]        
        img1 = cv2.imread(img_name_1,0)
        img2 = cv2.imread(img_name_2,0)
        CVTEST = CVTest()
        CVTEST.KPmatches_BF(filename_1,filename_2,img1,img2,1)      
        return        
    
    if argv[-1] not in MODETYPE:
        print("please choose your test mode")
    return 
    
    
    
if __name__ == "__main__":
#    img_name_1 = sys.argv[1]
#    filename_1 = sys.argv[2]
#    img_name_2 = sys.argv[3]
#    filename_2 = sys.argv[4]
    
    
    if sys.argv[-1] in ('help', '-help','--help','h','-h','--h'):
        Help().howToUseThisFile()
    else:
        ModrProcess(sys.argv)
#        try:
#            df_raw = pd.read_csv(filename_1,"\t",header = None, error_bad_lines=False)
#            img = cv2.imread(img_name_1,0)
#        except:
#            print("can not read file name , please check")
#            Help().howToUseThisFile()
#        else:
#            img1 = cv2.imread(img_name_1,0)
#            img2 = cv2.imread(img_name_2,0)
#            CVTEST = CVTest()
#            CVTEST.KPmatches_BF(filename_1,filename_2,img1,img2)
            