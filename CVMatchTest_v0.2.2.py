#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:24:38 2019

@author: qy
"""


import sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

class Help:
    def __init__(self):
        
        return

    def using_tutorial():
        """
        Update at Sat Apr 13 10:34:02 2019
        author: Yue QI
        email: yue.qi@uisee.com
        
        Depend libraries:
        
        if you are using virtual environment , just use pip install LIB, 
        or else use python -m pip install LIB
        
        python -m pip install numpy 
        python -m pip install matplotlib    
        python -m pip install pandas
        python -m pip install opencv-contrib-python 
        python -m pip install pyyaml
        
        
        v0.2.0 support different cv test mode
        currently there are 2 modes
        please make sure use the correct config.yaml
        
        useage:
            python CVMatchTest_v0.2.*.py $CONFIG_DIR/config.yaml
        
        *** CONFIG DESCRIPTION ***
        
        MATCH_MODE : 1 # for cvMatchTest v0.2.0, we support mode 0 and 1

        IMG_1:
            IMAGE_DIR : ~/1.tiff # img1 directory
            FILE_DIR : ~.2.txt   # img1 info directory
            VALID_KP_LAYER : 
                            - -1 # -1 means select all layer, if you want to add more layers, create a new line
                                 # and use - n to add the nth layer
                                 
                            
            VALID_KP_CLASS : 
                            - -1 # -1 means select all classes, if you want to add more layers, create a new line
                                 # and use - n to add the nth layer
        
        IMG_2:                   # same as img1
            IMAGE_DIR : ~/2.tiff
            FILE_DIR : ~/2.txt
            VALID_KP_LAYER : 
                            - -1 # -1 means select all layer,
                            
            VALID_KP_CLASS : 
                            - -1 # -1 means select all classes
        
        
        
        
        
        ---------------------------------------------------------------------------------------
        *** Mode 0 ***
        useage:
        
        
        Mode 0 will display the bf match result in 2 test files
        "number of query keypoints" : the number of query keypoints
        "number of BF matches" :      the number of matches pairs process by brute-force matching
        "number of good matches" :    the number of matches pairs process by brute-force matching
                                      and hamming distance thresholding (hamming distance <=25)
        ---------------------------------------------------------------------------------------
        *** Mode 1 ***
        useage:
        
        
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
    
    
    def __init__(self,config):
        
        self.config = config
        
        
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
        #reformat:
        
        n = df.shape[0]
        for i in range(n):
            kp = cv2.KeyPoint(df_kp['x'][i],df_kp['y'][i],df_kp['angle'][i], 
                              df_kp['response'][i],df_kp['octave'][i],df_kp['class_id'][i])
            kp.octave = int(df_kp['octave'][i])
            kp.class_id = int(df_kp['class_id'][i])
            kp.response = df_kp['response'][i]
            kps.append(kp)
        
        for i in range(n):
            desc = []
            for j in range(df_desc.shape[1]):
                desc.append(df_desc[j][i])
            desc = np.array(desc)
            des.append(desc)
        des = np.array(des)    
        print('Total keypoints number from '+ filename+' : '+
              str(len(kps)))
        print('Total descriptors number from '+ filename+' : '+
              str(len(des)))
    
        return kps, des
        

        
        
    
        
    
    def KPmatch(self, kp1, des1, kp2, des2, thr = 25, kpmatchType = 0):
        '''
        matching keypoints by descriptors
        
        matchType = 0 :brute force, thr:descriptors'hamming distance
        matchType = 1 :brute force, thr:keypoints'point location
        
        '''
        

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
#        print(type(kp1), type(kp2))
        matches = bf.match(des1,des2)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        goodMatches = []
        # mode 0:
        if kpmatchType == 0:
            for match in matches:
                if match.distance > thr:
                    continue
                else:  
                    goodMatches.append(match)
            return matches, goodMatches 
                
        # mode 1:
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
        '''
        # brute-force matching
        '''
        print("decsriptor's matching type: brute force")
        kp1, des1 = self.readfile(filename_1)
        kp2, des2 = self.readfile(filename_2)
        
        # preprocessing

        kp1, des1, kp2, des2 = self.getValidPoints(kp1, des1, kp2, des2)       

        # find matches in different matchtype
        # matchType = 0 :brute force, thr:descriptors'hamming distance
        # matchType = 1 :brute force, thr:keypoints'point location

        matches, goodMatches = self.KPmatch(kp1, des1, kp2, des2, kpmatchType = matchtype) 
        img3 = img1.copy()

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,goodMatches, img3, flags=0)
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
        
    def getValidPoints(self, kp1, des1, kp2, des2):
        
        # set threshold        
        thr_class_1 = self.config['IMG_1']['VALID_KP_CLASS']
        thr_layer_1 = self.config['IMG_1']['VALID_KP_LAYER']
        thr_class_2 = self.config['IMG_2']['VALID_KP_CLASS']
        thr_layer_2 = self.config['IMG_2']['VALID_KP_LAYER']   

        def get_kp_and_desc(kps, descs, class_thr, layer_thr):
            # class_id filter 
            print('keypoint class_id threshold:' + str(class_thr))
            print('keypoint octave layer threshold:' + str(layer_thr))
            valid_class_kps = []
            valid_class_descs = []
            n = len(kps)
            if -1 not in class_thr:
                for i in range(n):
                    kp = kps[i]
                    desc = descs[i]
                    if kp.class_id in class_thr:
                        
                        valid_class_kps.append(kp)
                        valid_class_descs.append(desc)
            else:
                valid_class_kps = kps
                valid_class_descs = descs
            
            # layer filter
            valid_layer_kps = []
            valid_layer_descs = []
            n = len(valid_class_kps)
            if -1 not in layer_thr:
                for i in range(n):
                    kp = valid_class_kps[i]
                    desc = valid_class_descs[i]
                    
                    if kp.octave in layer_thr:
                        valid_layer_kps.append(kp)
                        valid_layer_descs.append(desc)
            else:
                valid_layer_kps = valid_class_kps
                valid_layer_descs = valid_class_descs
#            print('test', len(valid_layer_kps))
            # open for other filter method
            valid_kps = valid_layer_kps
            valid_descs = valid_layer_descs

            return valid_kps, valid_descs
        print("searching valid keypoints and descriptors from file 1...")
        valid_kp1,valid_des1 = get_kp_and_desc(kp1,des1,thr_class_1,thr_layer_1)
        print("searching valid keypoints and descriptors from file 2...")
        valid_kp2,valid_des2 = get_kp_and_desc(kp2,des2,thr_class_2,thr_layer_2)
        valid_des1 = np.array(valid_des1)
        valid_des2 = np.array(valid_des2)
        
#        print(len(valid_kp1),len(valid_des1))
        return valid_kp1, valid_des1, valid_kp2, valid_des2        

        

    
    def MatchParisTest(self, matchPairs, matchParis_2):
        
        n_goodmatchpairs = 0
        
        return n_goodmatchpairs
    
    
   
    

def ModeProcess(config):
    if config['MATCH_MODE'] == 0:
        print("Mode 0 has been activated!")
        img_name_1 = config['IMG_1']['IMAGE_DIR']
        filename_1 = config['IMG_1']['FILE_DIR']
        img_name_2 = config['IMG_2']['IMAGE_DIR']
        filename_2 = config['IMG_2']['FILE_DIR']      
        img1 = cv2.imread(img_name_1,0)
        img2 = cv2.imread(img_name_2,0)
        CVTEST = CVTest(config)
        CVTEST.KPmatches_BF(filename_1,filename_2,img1,img2,0)         
        return
    
    if config['MATCH_MODE'] == 1:
        print("Mode 1 has been activated!")
        img_name_1 = config['IMG_1']['IMAGE_DIR']
        filename_1 = config['IMG_1']['FILE_DIR']
        img_name_2 = config['IMG_2']['IMAGE_DIR']
        filename_2 = config['IMG_2']['FILE_DIR']  
        img1 = cv2.imread(img_name_1,0)
        img2 = cv2.imread(img_name_2,0)
        CVTEST = CVTest(config)
        CVTEST.KPmatches_BF(filename_1,filename_2,img1,img2,1)         
        return
    Help().howToUseThisFile()    
    return
    
        

if __name__ == "__main__":

    
    
    if sys.argv[-1] in ('help', '-help','--help','h','-h','--h'):
        Help().howToUseThisFile()
    else:

        yamlPath = sys.argv[1]
        try:
            f = open(yamlPath, 'r', encoding='utf-8')
        except:
            f = open(yamlPath, 'r')
            
        cfg = f.read()
        config = yaml.load(cfg)
#        print('config:')
#        print(config)
#        print('\n')
        ModeProcess(config)

            