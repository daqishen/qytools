#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:05:11 2019

@author: qy
"""


import sys
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np


#%%
class Help:
    def __init__(self):
        
        return

    def using_tutorial():
        """
        Update at 29 July
        author: Yue Qi
        email: yue.qi@uisee.com
        
        ------------------------------------------------------------------
        Useage: 
            python vtracker_va.b.c vtracker.yaml
            
        ------------------------------------------------------------------
        The vtracker.yaml should be a yaml file and lood like this:(copy between "#")
            
        ##################################################################    
        TEST_UNIT_0:
            TEST_MODE : 0 
            ENABLE : 1
            VTRACKER_LOG_DIR : vtracker_test.log
        
        TEST_UNIT_1:
            TEST_MODE : 1
            ENABLE : 1
            VTRACKER_LOG_DIR : vtracker_test.log
            GPS_DIR : cube_gps.txt
    
        TEST_UNIT_2:
            TEST_MODE : 2
            ENABLE : 1
            VTRACKER_LOG_DIR : vtracker_test.log
            TIMELOG_DIR : timelog.txt
    
        TEST_UNIT_3:
            TEST_MODE : 3
            ENABLE : 1
            VTRACKER_LOG_DIR : video0.log
            TIMELOG_DIR : TimeLog0.txt
            GPS_DIR : 
        ##################################################################    

        Guide:
            currently this script can support 4 different mode for log analysis
        ------------------------------------------------------------------
        TEST_MODE : 0
        To activate this mode, set TEST_MODE : 0 and ENABLE : 1 
        You need to get a vtracker_test log and set the log directory in VTRACKER_LOG_DIR.
        If you don't know how to run vtracker_test, please check uisee map-building guide.
        Mode 0 can show details about different stages in relocation. It shows the time cost and 
        count the number for each stage. 
        
        ------------------------------------------------------------------
        TEST_MODE : 1
        To activate this mode, set TEST_MODE :1 and ENABLE : 1
        Mode 1 shows the total behavior about whole tracking process.
        There is an extra choice to add GPS file to get regression test.
        If this mode has been activate, there will be a Locate_result.csv file after running the script
        (And Regression_result.csv file if you add gps).        
        !!! Please make sure the gps file matches the vtracker log, or the result will be wrong. !!!

        ------------------------------------------------------------------
        TEST_MODE : 2
        To activate this mode, set TEST_MODE :2 and ENABLE : 1
        Mode 2 shows the total behavior about whole tracking process.
        There is an extra choice to add Timelog file to get regression test.
        In Test mode 2, the vtracker log analysis part is same as test mode 1
        This mode is for testing regression result for the SLAM images(which are used for building map)
        
        ------------------------------------------------------------------
        TEST_MODE : 3
        To activate this mode, set TEST_MODE :3 and ENABLE : 1
        Mode 3 tells the information in vehicle coordinate, including the Yaw, vertical and horizontal.
        This mode can support both gps file and timelog file, but only process gps file if user add both of them.
        Notice:
        1. NOT all tracked data will be used for regression, the timestamp in timelog and camera are different.
        2. The difference in yaw, vertical and horizontal is absolute value.
        
        
        
        @author: qy
        """
        return
    
    def howToUseThisFile(self):
        
        return help(self.using_tutorial)

#%%
class Relocate:
    
    def __init__(self, df_raw):
        # initialization, save the raw data
        # some logs somehow have double space split naive solution
        df_raw = df_raw[0].str.split("    ",expand = True)    
        df_raw = df_raw[0].str.split("   ",expand = True)
        df_raw = df_raw[0].str.split("  ",expand = True)
        
        if df_raw.shape[1] > 1:
            df_raw[0] = df_raw[0]+' '+df_raw[1]
        df_raw = df_raw[0].str.split(" ", expand = True)
        df_raw[3] = df_raw[3].str.split(":", expand = True)
        self.df_raw = df_raw
        
        return
    
    def trackerAnalysis(self):
        
        # This part is mainly working for analysing log from tracker.cc
        df_tracker = self.df_raw[self.df_raw[3]=="tracker.cc"]  # log from tracker.cc
        
        #preprocess
        df_reloc_bow = df_tracker[df_tracker[4] == "reloc_bow:"].copy()
        df_reloc_bow[5] = df_reloc_bow[5].astype("float")
        df_reloc_pnp = df_tracker[df_tracker[4] == "reloc_pnp:"].copy()
        df_reloc_pnp[5] = df_reloc_pnp[5].astype("float")
        df_reloc_opt = df_tracker[df_tracker[4] == "reloc_opt:"].copy()
        df_reloc_opt[5] = df_reloc_opt[5].astype("float")
        df_reloc_proj = df_tracker[df_tracker[4] == "reloc_proj:"].copy()
        df_reloc_proj[5] = df_reloc_proj[5].astype("float")
        
        
        def plot_reloc_bow(df):
            thr = [15] * df.shape[0]
            plt.plot(df[5].values)
            plt.plot(thr,"r--")
            plotName = ["reloc_bow", "thr=15"]
            plt.legend(tuple(plotName), loc='upper right')
            n_pass = len(df[df[5] >= 15])
            plt.title("Times for searching by bow: "+ str(df.shape[0])+
                      " valid number:" + str(n_pass))
            
        def plot_reloc_pnp(df):
            thr = [3] * df.shape[0]
            plt.plot(df[5].values)
            plt.plot(thr,"r--")
            plotName = ["reloc_pnp", "thr=3"]
            plt.legend(tuple(plotName), loc='upper right')
            n_pass = len(df[df[5] >= 3])
            plt.title("Times for solving pnp naive: "+ str(df.shape[0])+
                      " valid number:" + str(n_pass))
            
        def plot_reloc_opt(df):
            thr = [10] * df.shape[0]
            thr_safe = [50] * df.shape[0]
            plt.plot(df[5].values)
            plt.plot(thr,"r--")
            plt.plot(thr_safe,"b--")
            plotName = ["reloc_opt", "thr=10","pass=50"]
            plt.legend(tuple(plotName), loc='upper right')
            n_pass = len(df[df[5] >= 10])
            plt.title("Times for solving pose opt: "+ str(df.shape[0])+
                      " valid number:" + str(n_pass))   
         
        def plot_reloc_proj(df):
            thr = [50] * df.shape[0]
            plt.plot(df[5].values)
            plt.plot(thr,"r--")
            plotName = ["Proj", "pass=50"]
            plt.legend(tuple(plotName), loc='upper right')
            n_pass = len(df[df[5] >= 50])
            plt.title("Times for searching by proj: "+ str(df.shape[0])+
                      " valid number:" + str(n_pass))  
            
            return
        
        plt.figure("tracker.cc relocate analysis ") #figure tracker.cc
        plt.subplot(2,2,1)    
        plot_reloc_bow(df_reloc_bow)
        plt.subplot(2,2,2)
        plot_reloc_pnp(df_reloc_pnp)
        plt.subplot(2,2,3)
        plot_reloc_opt(df_reloc_opt)
        plt.subplot(2,2,4)
        plot_reloc_proj(df_reloc_proj)

#        plt.show()
        
        return
        
        
    def commonLogAnalysis(self):
        
        df_common = self.df_raw[self.df_raw[3]=="common.h"]  # log from common.h

        df_search_by_bow = df_common[df_common[6] == "search_by_bow:"].copy()
        
        df_search_by_bow[7] = df_search_by_bow[7].str.split("ms", expand = True)   
        df_search_by_bow[7] = df_search_by_bow[7].astype("float")
        
        df_solve_pnp_naive = df_common[df_common[6] == "solve_pnp_naive:"].copy()
        df_solve_pnp_naive[7] = df_solve_pnp_naive[7].str.split("ms", expand = True)
        df_solve_pnp_naive[7] = df_solve_pnp_naive[7].astype("float")
        
        df_pose_optimize = df_common[df_common[6] == "pose_optimize:"].copy()
        df_pose_optimize[7] = df_pose_optimize[7].str.split("ms", expand = True)
        df_pose_optimize[7] = df_pose_optimize[7].astype("float")
        
        df_search_by_projection = df_common[df_common[6] == "search_by_projection:"].copy()
        df_search_by_projection[7] = df_search_by_projection[7].str.split("ms", expand = True)
        df_search_by_projection[7] = df_search_by_projection[7].astype("float")
        
        def plot_common_h(df, title):
            plt.plot(df[7].values)
            n = df.shape[0]
            plt.ylabel("ms")
            plt.title(title+"\n"+"Total Times = "+str(n)+" Avg time cost = "+'%.2f'%df[7].mean()+"ms")
        
        plt.figure("common.h relocate analysis ") #figure common.h
        
        plt.subplot(2,2,1)
        plot_common_h(df_search_by_bow,"Search By bow")
        plt.subplot(2,2,2)
        plot_common_h(df_solve_pnp_naive,"Solve_pnp_naive")
        plt.subplot(2,2,3)
        plot_common_h(df_pose_optimize,"Pose_optimize")
        plt.subplot(2,2,4)
        plot_common_h(df_search_by_projection,"search_by_projection")
        
#        plt.show()        
        

        return 

class Regression:
    
    def __init__(self):
        
        return
    
    def gpsPreprocess(self, df_gps):
        
        df_gps = df_gps[0].str.split(" ", expand = True)
        df_gps.columns = ["Image_id", "X", "Y", "Z",
                          "Yaw", "Roll", "Pitch", "Conf_Score"]
        
        # preprocess str to float
        df_gps["X"] = df_gps["X"].astype("float")
        df_gps["Y"] = df_gps["Y"].astype("float")
        df_gps["Z"] = df_gps["Z"].astype("float")
        df_gps["Yaw"] = df_gps["Yaw"].astype("float")
        df_gps["Roll"] = df_gps["Roll"].astype("float")
        df_gps["Pitch"] = df_gps["Pitch"].astype("float")
        
        df_gps["Conf_Score"] = df_gps["Conf_Score"].astype("float")
        
        df_gps.to_csv("df_gps.csv")
        
        return df_gps

    def gpsShift(self, df_gps, regression_shift):
        
        '''
        switch to the vehicle-centered coordinate
        for UOS coordinate, the heading base is 90 degree, when the vehicle is heading to east
        the Yaw is 0 degree
        '''
        regression_shift[5] -= 90 # switch to uos coordinate
        for i in range(3,6):
            regression_shift[i] *= (np.pi/180)

        df_gps["Roll"] = df_gps["Roll"] - regression_shift[3]
        df_gps["Pitch"] = df_gps["Pitch"] - regression_shift[4]
        df_gps["Yaw"] = df_gps["Yaw"] - regression_shift[5]
        
        return df_gps
        
    def timelog2Gps(self, df_timelog):
        """
        Adapt timelog form to gps form,
        timelog is mainly using for mono slam, gps is mainly for fish eyes
        """
        
        df_timelog = df_timelog[0].str.split("@", expand = True)
        # the old version timelog has no Z info
        # the first 8 colummns should be 
        # IMG_DIR @ Image_id @ Yaw @ X @ Y @ Conf_Score @ Time @ Z 
        
        if df_timelog.shape[1] == 7:
            # some timelogs dont have z
            df_timelog.columns = ["IMG_DIR", "Image_id", "Yaw", "X", "Y", "Conf_Score", 
                              "Time"]
            df_timelog["Z"] = ''
        else:
      
            df_timelog.columns = ["IMG_DIR", "Image_id", "Yaw", "X", "Y", "Conf_Score", 
                                  "Time", "Z"]
        df_timelog.dropna()
        
        # pad 0 to blank space
        # confidence score = 5 means the current information is not reliable

        df_timelog.loc[df_timelog.X == '', "Conf_Score"] = '5'
        df_timelog.loc[df_timelog.Y == '', "Conf_Score"] = '5'
        df_timelog.loc[df_timelog.Z == '', "Conf_Score"] = '5'
        df_timelog.loc[df_timelog.Yaw == '', "Conf_Score"] = '5'
        df_timelog[df_timelog["X"] == ''] = '0'
        df_timelog[df_timelog["Yaw"] == ''] = '0'
        df_timelog[df_timelog["Y"] == ''] = '0'
        df_timelog[df_timelog["Z"] == ''] = '0'
        df_timelog[df_timelog["Conf_Score"] == ''] = '5'
        
        df_timelog["X"] = df_timelog["X"].astype("float")
        df_timelog["Y"] = df_timelog["Y"].astype("float")
        df_timelog["Z"] = df_timelog["Z"].astype("float")
        df_timelog["Conf_Score"] = df_timelog["Conf_Score"].astype("float")
        df_timelog["Yaw"] = df_timelog["Yaw"].astype("float")
        df_timelog = df_timelog[df_timelog["Conf_Score"] >0 ]
        
        # rearrange datafame
        df_timelog2gps = pd.DataFrame(df_timelog["Image_id"])
        df_timelog2gps['X'] = df_timelog["X"]
        df_timelog2gps["Y"] = df_timelog["Y"]
        df_timelog2gps["Z"] = df_timelog["Z"]
        df_timelog2gps["Roll"] = 0   #timelog has no Roll output
        df_timelog2gps["Pitch"] = 0  #timelog has no pitch output
        df_timelog2gps["Yaw"] = df_timelog["Yaw"]
        df_timelog2gps["Conf_Score"] = df_timelog["Conf_Score"] == 4  # gps confident score = 4 means reliable
        df_timelog2gps["Conf_Score"] = df_timelog2gps["Conf_Score"].astype("float")
        
#        df_timelog2gps.to_csv("df_timelog2gps.csv")
        
        return df_timelog2gps
    
    def vtrackerPreprocess(self, df_vtracker):
        """
        process vtracker log
        """
        df_vtracker = df_vtracker[0].str.split("    ",expand = True)    
        df_vtracker = df_vtracker[0].str.split("   ",expand = True)
        df_vtracker = df_vtracker[0].str.split("  ",expand = True)
        
#        df_vtracker = df_vtracker[0].str.split("testing result: ", expand = True)
#        used to be the last column
        df_vtracker = df_vtracker[df_vtracker.shape[1]-1].str.split("testing result: ", expand = True) 

        df_vtracker[1] = df_vtracker[1].astype("str")
        
        df_locate_res = df_vtracker[df_vtracker[1] != "None"].copy()

        df_locate_res = df_locate_res[1].str.split(" ", expand = True)
        
        if df_locate_res.shape[0] < 10:
            print("The data size is too small, less than 10 frames")
            
        df_name = df_locate_res[0].str.split("/", expand = True)
        df_locate_res[0] = df_name[df_name.shape[1]-1].str[:14]
        
        # State 1:OK 0:LOST
        df_locate_res.columns = ["Image_id", "State", "X", "Y",
                          "Z", "Roll", "Pitch", "Yaw", "Matches"]
        df_locate_res["State"] = df_locate_res["State"].astype("int")
        df_locate_res["X"] = df_locate_res["X"].astype("float")
        df_locate_res["Y"] = df_locate_res["Y"].astype("float")
        df_locate_res["Z"] = df_locate_res["Z"].astype("float")
        df_locate_res["Roll"] = df_locate_res["Roll"].astype("float")
        df_locate_res["Pitch"] = df_locate_res["Pitch"].astype("float")
        df_locate_res["Yaw"] = df_locate_res["Yaw"].astype("float")
        df_locate_res["Matches"] = df_locate_res["Matches"].astype("float")
        
        df_locate_res.to_csv("Locate_result.csv")
        return df_locate_res
        
    def plotTrackResult(self, df_locate_res):
        plt.figure("Locate status ")
        plt.subplot(2,1,1)
        df_tracked = df_locate_res[df_locate_res["State"] == 1] # must be in tracking status
        tracking_rate = df_tracked.shape[0]/df_locate_res.shape[0]
        print("Total Frame Number: " + str(df_locate_res.shape[0]))
        print("Tracking Rate: " + str(tracking_rate) )
        #plot tracking state
        plt.plot(df_locate_res["State"].values)
        plt.title("Tracking Rate: " + str("%.3f"%tracking_rate) + '\n' + "Frame Number: " + str(df_locate_res.shape[0]))
                  
#        plt.xlabel("Frame" )
        plt.ylabel("Tracking States"  + "\n" + 
                  "1 : OK, 0 : LOST")
               
        # Matches number
        
        plt.subplot(2,1,2)
        plt.plot(df_locate_res["Matches"].values)
        plt.title("Matching result: mean = " + "%.3f"%df_locate_res["Matches"][df_locate_res["Matches"] > 0].mean() )
#        plt.xlabel("Frame")
        plt.ylabel("Matches Number")
        
        plt.show()
        
        return
        
    def plotRegressResult(self, df_gps, df_locate_res):
        """
        plot regression result
        """
        
        # merge gps and locate result based on Image_id
        df_regression = pd.merge(df_locate_res, df_gps, on="Image_id").copy()
        # set unit from meter to centimeter
        df_regression.columns = ["Image_id", "State", "X_loc", "Y_loc","Z_loc",
                                 "Yaw_loc", "Roll_loc", "Pitch_loc", "Matches",
                                 "X_gps", "Y_gps", "Z_gps", "Roll_gps", "Pitch_gps",
                                 "Yaw_gps", "Conf_Score"]
        
        
        df_regression["X_diff"] = abs(df_regression["X_loc"] - df_regression["X_gps"])
        df_regression["Y_diff"] = abs(df_regression["Y_loc"] - df_regression["Y_gps"])
        df_regression["Z_diff"] = abs(df_regression["Z_loc"] - df_regression["Z_gps"])
        # set lost diff to -0.1 to show the result
        # but the unreliable result will not be computed
        df_regression.loc[df_regression.State == 0, "X_diff"] = -0.1
        df_regression.loc[df_regression.State == 0, "Y_diff"] = -0.1
        df_regression.loc[df_regression.State == 0, "Z_diff"] = -0.1        
#        df_regression["X_diff"][df_regression["State"] == 0] = -0.1
#        df_regression["Y_diff"][df_regression["State"] == 0] = -0.1
#        df_regression["Z_diff"][df_regression["State"] == 0] = -0.1       
        
        plt.figure("Regression Error")
        plt.subplot(3, 1, 1)
        plt.plot(df_regression["X_diff"])
        #plt.xlabel("Frame")
        plt.ylabel("X_diff(m)")
        lost = [-0.1] * df_regression.shape[0]
        thr = [0.2] * df_regression.shape[0]
        plt.plot(lost,"r--")
        plt.plot(thr, "r")
        plotName = ["X_diff", "LOST","thr = 0.2"]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Abs difference in X direction" + ' mean: %.2f'%df_regression["X_diff"][df_regression["X_diff"]>0].mean())
        
        
        plt.subplot(3, 1, 2)
        plt.plot(df_regression["Y_diff"])
        #plt.xlabel("Frame")
        plt.ylabel("Y_diff(m)")
        lost = [-0.1] * df_regression.shape[0]
        thr = [0.2] * df_regression.shape[0]
        plt.plot(lost,"r--")
        plt.plot(thr, "r")
        plotName = ["Y_diff", "LOST","thr = 0.2"]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Abs difference in Y direction" + ' mean: %.2f'%df_regression["Y_diff"][df_regression["Y_diff"]>0].mean())
        
        
        plt.subplot(3, 1, 3)
        plt.plot(df_regression["Z_diff"])
        #plt.xlabel("Frame")
        plt.ylabel("Z_diff(m)")
        lost = [-0.1] * df_regression.shape[0]
        thr = [0.2] * df_regression.shape[0]
        plt.plot(lost,"r--")
        plt.plot(thr, "r")
        plotName = ["Z_diff", "LOST","thr = 0.2"]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Abs difference in Z direction" + ' mean: %.2f'%df_regression["Z_diff"][df_regression["Z_diff"]>0].mean())
        
        
        #df_regression = df_regression[df_regression["State"] == 1] # state should be ok
        #df_regression = df_regression[df_regression["Conf_Score"] > 0.7] # confidence score > 0.7

        plt.show()
        df_regression.to_csv("Regression_result.csv")
        return 


    def plotRegressResultInVehicleCoord(self, df_gps, df_locate_res, x_shift = 0, y_shift = 0):
        
        # create a vague id and make the error time range in 10ms
        df_gps["vague_id"] = df_gps["Image_id"].str[:13]
        df_locate_res["vague_id"] = df_locate_res["Image_id"].str[:13]
#        df_gps["vague_id"] = df_gps["Image_id"].str[:14]
#        df_locate_res["vague_id"] = df_locate_res["Image_id"].str[:14]

        df_regression = pd.merge(df_locate_res, df_gps, on="vague_id").copy()

        # rename the columns
        # TODO: Deal with the extended dataframe
        df_regression.columns = ["Image_id", "State", "X_loc", "Y_loc","Z_loc",
                                 "Yaw_loc", "Roll_loc", "Pitch_loc", "Matches", "Vague_id","Image_gps_id",
                                 "X_gps", "Y_gps", "Z_gps", "Roll_gps", "Pitch_gps",
                                 "Yaw_gps", "Conf_Score"]
        df_regression = df_regression[df_regression["State"] == 1].copy()
        df_regression.reset_index() 
        # calculate difference in World Coordinate
        df_regression["X_diff"] = df_regression["X_loc"] - df_regression["X_gps"]
        df_regression["Y_diff"] = df_regression["Y_loc"] - df_regression["Y_gps"]
        df_regression["Z_diff"] = df_regression["Z_loc"] - df_regression["Z_gps"]

        # set Yaw in range 0 ~ 2*pi
        df_regression.loc[df_regression.Yaw_loc > 2*np.pi, "Yaw_loc"] -= (2 * np.pi)
        df_regression.loc[df_regression.Yaw_loc < 0, "Yaw_loc"] += (2 * np.pi)
        df_regression.loc[df_regression.Yaw_gps > 2*np.pi, "Yaw_gps"] -= (2 * np.pi)
        df_regression.loc[df_regression.Yaw_gps < 0, "Yaw_gps"] += (2 * np.pi)        
        # set angle difference in absolute value
        # the data index such as mean and variance should be in absolute value
        
        df_regression["Yaw_diff_arc"] = df_regression["Yaw_loc"] - df_regression["Yaw_gps"]
        # difference in yaw should be less or equal than 180 degree
        df_regression.loc[df_regression.Yaw_diff_arc >= np.pi, "Yaw_diff_arc"] -= 2*np.pi
        df_regression.loc[df_regression.Yaw_diff_arc <= -np.pi, "Yaw_diff_arc"] += 2*np.pi
        # arc to angle 
        df_regression["Yaw_diff_ang"] = df_regression["Yaw_diff_arc"]*(180.0/np.pi)
     
        # calculate differece in vehicle coordinate
        # Vertical_diff means X diff in car coordinate, Horizontal_diff means Y diff in car coordinate
        """
        Formula derivation:
        Vector v = [X_w, Y_w] in world coordinate v = [X_v, Y_v] in vehicle coordinate
        unit vector in world coordinate : [e1_w, e2_w]
        unit vector in vehicle coordinate : [e1_v, e2_v]
        heading angle : Yaw
        so:
                                   
        v = [e1_w, e2_w] · trans([X_w, Y_w]) = [e1_v, e2_v] · trans([X_v, Y_v])
        
        trans([X_v, Y_v]) = trans([e1_v, e2_v]) · [e1_w, e2_w] · trans([X_w, Y_w])
                          
                             _                         _
                             | e1_v · e1_w  e1_v · e2_w | 
                          =  |                          |  · trans([X_w, Y_w])
                             | e2_v · e1_w  e2_v · e2_w |
                             |_                        _|
                             
                             
                             _                     _
                             | sin(Yaw)  -cos(Yaw) | 
                          =  |                     |  · trans([X_w, Y_w])
                             | cos(Yaw)   sin(Yaw) |
                             |_                   _|
                             
        
        """
        
        
        df_regression["Vehicle_x"] = np.sin(df_regression["Yaw_gps"]) * df_regression["X_diff"] + \
                                     (-1) * np.cos(df_regression["Yaw_gps"]) * df_regression["Y_diff"] + \
                                     x_shift
        df_regression["Vehicle_y"] = np.cos(df_regression["Yaw_gps"]) * df_regression["X_diff"] + \
                                     (+1) * np.sin(df_regression["Yaw_gps"]) * df_regression["Y_diff"] + \
                                     y_shift

                                       
        
        # all different needs to be set in absolute distance
#        df_regression["Vehicle_x"] = abs(df_regression["Vehicle_x"])
#        df_regression["Vehicle_y"] = abs(df_regression["Vehicle_y"])
 
        
        print("process frames number: " + str(df_regression.shape[0]))
        plt.figure("Regression Error")
        plt.subplot(3, 1, 1)
        Yaw_diff_ang = np.array(df_regression["Yaw_diff_ang"])
        plt.plot(Yaw_diff_ang)
        plt.ylabel("Yaw_diff(degree)")
        threshold = 2
        thr_pos = [threshold] * df_regression.shape[0]
        thr_neg = [-threshold] * df_regression.shape[0]
        plt.plot(thr_pos, "r")
        plt.plot(thr_neg, "r")
        plotName = ["Yaw_diff", "thr = " + str(threshold), "thr = " + str(-threshold)]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Abs difference in Yaw (degree) " + ' mean: %.2f'%abs(df_regression["Yaw_diff_ang"]).mean() \
                    + " Var: %.2f"%abs(df_regression["Yaw_diff_ang"]).var())
        
        
        plt.subplot(3, 1, 2)
        Vehicle_x = np.array(df_regression["Vehicle_x"])
        plt.plot(Vehicle_x)
        plt.ylabel("Vehicle_x(m)")
        threshold = 0.5
        thr_pos = [threshold] * df_regression.shape[0]
        thr_neg = [-threshold] * df_regression.shape[0]
        plt.plot(thr_pos, "r")
        plt.plot(thr_neg, "r")
        plotName = ["Vehicle_x", "thr = " + str(threshold), "thr = " + str(-threshold)]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Abs difference in Vehicle_x (m) " + ' mean: %.2f'%abs(df_regression["Vehicle_x"]).mean() \
                    + " Var: %.2f"%abs(df_regression["Vehicle_x"]).var())        
        
        plt.subplot(3, 1, 3)
        Vehicle_y = np.array(df_regression["Vehicle_y"])
        plt.plot(Vehicle_y)
        plt.ylabel("Vehicle_y(m)")
        threshold = 0.5
        thr_pos = [threshold] * df_regression.shape[0]
        thr_neg = [-threshold] * df_regression.shape[0]
        plt.plot(thr_pos, "r")
        plt.plot(thr_neg, "r")
        plotName = ["Vehicle_y", "thr = " + str(threshold), "thr = " + str(-threshold)]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Abs difference in Vehicle_y (m) " + ' mean: %.2f'%abs(df_regression["Vehicle_y"]).mean() \
                    + " Var: %.2f"%abs(df_regression["Vehicle_y"]).var())        
        print("Abs difference in Yaw (degree) " + ' mean: %.2f'%abs(df_regression["Yaw_diff_ang"]).mean() \
                    + " Std: %.2f"%abs(df_regression["Yaw_diff_ang"]).std())
        print("Abs difference in Vehicle_x (m) " + ' mean: %.2f'%abs(df_regression["Vehicle_x"]).mean() \
                    + " Var: %.2f"%abs(df_regression["Vehicle_x"]).var())        
        print("Abs difference in Vehicle_y (m) " + ' mean: %.2f'%abs(df_regression["Vehicle_y"]).mean() \
                    + " Var: %.2f"%abs(df_regression["Vehicle_y"]).var())        
        
        plt.show()
        df_regression.to_csv("Regression_result.csv")
        return 



def Process(config):
    """
    Process all the enabled unit
    """
    for ele in config:
        test_unit = config[ele]
        if test_unit["ENABLE"] == True:
            print(ele + " Mode " + str(test_unit["TEST_MODE"]) + " has been activated")
            modeProcess(test_unit)
                       
    return

def modeProcess(test_unit):
    
    # mode 0: plot information from vtracker_test.log
    # relocate check shows the performance in relocation when lost
    if test_unit["TEST_MODE"] == 0:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        try:
            df_raw = pd.read_csv(vtracker_log,"\t",header = None, error_bad_lines=False)
        except:
            print("can not read vtracker file, please check log dir")
            Help().howToUseThisFile()
            return
        
        try:
            relocateCheck = Relocate(df_raw)
            relocateCheck.trackerAnalysis()
            plt.show()
            relocateCheck.commonLogAnalysis()
            plt.show()
        except:
            print("Relcoate check failed, is there any successful relocation? ")
                
    elif test_unit["TEST_MODE"] == 1:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        gps_log = test_unit["GPS_DIR"]
        regressionCheck = Regression()
        Fvtracker = False
        Fgps = False
        try:
            print("load vtracker log")
            df_vtracker = pd.read_csv(vtracker_log, "\t", header = None, error_bad_lines=False)
            df_locate_res = regressionCheck.vtrackerPreprocess(df_vtracker)
        except:
            print("vtracker log preprocess failed, please check log dir")
            return
        else:
            Fvtracker = True
        try:
            df_gps = pd.read_csv(gps_log, "\t", header = None, error_bad_lines=False)
            df_gps = regressionCheck.gpsPreprocess(df_gps)
        except:
            print("gps log preprocess failed, there will be no regression result")
        else:
            Fgps = True
                
        if Fvtracker:
            regressionCheck.plotTrackResult(df_locate_res)
        if Fgps:
            regressionCheck.plotRegressResult(df_gps, df_locate_res)
            
    elif test_unit["TEST_MODE"] == 2:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        timelog = test_unit["TIMELOG_DIR"]
        regressionCheck = Regression()
        Fvtracker = False
        Fgps = False
        try:
            print("load vtracker log")
            df_vtracker = pd.read_csv(vtracker_log, "\t", header = None, error_bad_lines=False)
            df_locate_res = regressionCheck.vtrackerPreprocess(df_vtracker)
        except:
            print("vtracker log preprocess failed, please check log dir")
            return
        else:
            Fvtracker = True
        try:
            df_timelog = pd.read_csv(timelog, "\t", header = None, error_bad_lines=False)
            df_gps = regressionCheck.timelog2Gps(df_timelog)
        except:
            print("timelog preprocess failed, there will be no regression result")
        else:
            Fgps = True
                
        if Fvtracker:
            regressionCheck.plotTrackResult(df_locate_res)
        if Fgps:
            regressionCheck.plotRegressResult(df_gps, df_locate_res)
            
    elif test_unit["TEST_MODE"] == 3:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        timelog = test_unit["TIMELOG_DIR"]
        gps_log = test_unit["GPS_DIR"]
        regressionCheck = Regression()
        Fvtracker = False
        Fgps = False
        #load vtracker log
        try:
            print("load vtracker log")
            df_vtracker = pd.read_csv(vtracker_log, "\t", header = None, error_bad_lines=False)
            df_locate_res = regressionCheck.vtrackerPreprocess(df_vtracker)
        except:
            print("vtracker log preprocess failed, please check log dir")
            return
        else:
            Fvtracker = True
        #load timelog
        try:
            df_timelog = pd.read_csv(timelog, "\t", header = None, error_bad_lines=False)
            df_gps = regressionCheck.timelog2Gps(df_timelog)
        except:
            print("no timelog, checking gps")
        else:
            Fgps = True
        #load gps
        try:
            df_gps = pd.read_csv(gps_log, "\t", header = None, error_bad_lines=False)
            df_gps = regressionCheck.gpsPreprocess(df_gps)
        except:
            if not Fgps:              
                print("no gps and timelog detected, there will be no regression check")
        else:
            Fgps = True
        if Fvtracker:
            regressionCheck.plotTrackResult(df_locate_res)
        if Fgps:
            regressionCheck.plotRegressResultInVehicleCoord(df_gps, df_locate_res)
        else:
            print("no gps or timelog file detected")
        return     
  

    elif test_unit["TEST_MODE"] == 4:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        timelog = test_unit["TIMELOG_DIR"]
        gps_log = test_unit["GPS_DIR"]
        regression_shift = test_unit["REGRESSION_SHIFT"]
        regressionCheck = Regression()
        Fvtracker = False
        Fgps = False
        #load vtracker log
        try:
            print("load vtracker log")
            df_vtracker = pd.read_csv(vtracker_log, "\t", header = None, error_bad_lines=False)
            df_locate_res = regressionCheck.vtrackerPreprocess(df_vtracker)
        except:
            print("vtracker log preprocess failed, please check log dir")
            return
        else:
            Fvtracker = True
        #load timelog
        try:
            df_timelog = pd.read_csv(timelog, "\t", header = None, error_bad_lines=False)
            df_gps = regressionCheck.timelog2Gps(df_timelog)
        except:
            print("no timelog, checking gps")
        else:
            Fgps = True
        #load gps
        try:
            df_gps = pd.read_csv(gps_log, "\t", header = None, error_bad_lines=False)
            df_gps = regressionCheck.gpsPreprocess(df_gps)
        except:
            if not Fgps:              
                print("no gps and timelog detected, there will be no regression check")
        else:
            Fgps = True
        if Fvtracker:
            regressionCheck.plotTrackResult(df_locate_res)
        if Fgps:
            # check if there is regression shift
            # regression_shift = [ X(meter), Y(meter), Z(meter), Roll(angle), Pitch(angle), Yaw(angle)]
            # X, Y, Z in meters
            # Roll, Pitch, Yaw in angle
            # the default number is [0, 0, 0, 0, 0, 90]
            # 
            if len(regression_shift) != 6:
                regression_shift = [0, 0, 0, 0, 0, 90]
            print("the default regression_shift is [0, 0, 0, 0, 0, 90]")
            print("your regression shift: " + str(regression_shift))
            df_gps = regressionCheck.gpsShift(df_gps, regression_shift)
            regressionCheck.plotRegressResultInVehicleCoord(df_gps, df_locate_res,regression_shift[0], regression_shift[1])
        else:
            print("no gps or timelog file detected")
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

        Process(config)
'''
yamlPath = "vtracker.yaml"
#yamlPath = "vtracker-release-test.yaml"
try:
    f = open(yamlPath, 'r', encoding='utf-8')
except:
    f = open(yamlPath, 'r')
    
cfg = f.read()
config = yaml.load(cfg)
Process(config)
'''