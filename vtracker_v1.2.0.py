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


#%%
class Help:
    def __init__(self):
        
        return

    def using_tutorial():
        """
        Update at 25 June
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
            
        ##################################################################    

        Guide:
            currently this script can support 2 different mode for log analysis
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
                          "Roll", "Pitch", "Yaw", "Conf_Score"]
        # preprocess str to float
        df_gps["X"] = df_gps["X"].astype("float")
        df_gps["Y"] = df_gps["Y"].astype("float")
        df_gps["Z"] = df_gps["Z"].astype("float")
        df_gps["Roll"] = df_gps["Roll"].astype("float")
        df_gps["Pitch"] = df_gps["Pitch"].astype("float")
        df_gps["Yaw"] = df_gps["Yaw"].astype("float")
        df_gps["Conf_Score"] = df_gps["Conf_Score"].astype("float")
        
        return df_gps
    
    def vtrackerPreprocess(self, df_vtracker):
        
        df_vtracker = df_vtracker[0].str.split("    ",expand = True)    
        df_vtracker = df_vtracker[0].str.split("   ",expand = True)
        df_vtracker = df_vtracker[0].str.split("  ",expand = True)
        
        df_vtracker = df_vtracker[0].str.split("testing result: ", expand = True)
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
        plt.subplot(2,1,1)
        df_tracked = df_locate_res[df_locate_res["State"] == 1]
        tracking_rate = df_tracked.shape[0]/df_locate_res.shape[0]
        print("Total Frame Number: " + str(df_locate_res.shape[0]))
        print("Tracking Rate: " + str(tracking_rate) )
        #plot tracking state
        plt.plot(df_locate_res["State"].values)
        plt.title("Frame Number: " + str(df_locate_res.shape[0]) + "\n" +
                  "Tracking Rate: " + str(tracking_rate))
#        plt.xlabel("Frame" )
        plt.ylabel("Tracking States"  + "\n" + 
                  "1 : OK, 0 : LOST")
        
        
        # Matches number
        
        plt.subplot(2,1,2)
        plt.plot(df_locate_res["Matches"].values)
        plt.title("Frame Number: " + str(df_locate_res.shape[0]) + '\n'
                  +"mean = " + "%.2f"%df_locate_res["Matches"].mean())
#        plt.xlabel("Frame")
        plt.ylabel("Matches Number")
        plt.show()
        
        return
        
    def plotRegressResult(self, df_gps, df_locate_res):
        
        df_regression = pd.merge(df_locate_res, df_gps, on="Image_id").copy()
        # set unit from meter to centimeter
        df_regression.columns = ["Image_id", "State", "X_loc", "Y_loc","Z_loc",
                                 "Roll_loc", "Pitch_loc", "Yaw_loc", "Matches",
                                 "X_gps", "Y_gps", "Z_gps", "Roll_gps", "Pitch_gps",
                                 "Yaw_gps", "Conf_Score"]
        
        # set unit from meter to centimeter
        df_regression["X_diff"] = abs(df_regression["X_loc"] - df_regression["X_gps"])
        df_regression["Y_diff"] = abs(df_regression["Y_loc"] - df_regression["Y_gps"])
        df_regression["Z_diff"] = abs(df_regression["Z_loc"] - df_regression["Z_gps"])
        # set lost diff to -1
        df_regression.loc[df_regression.State == 0, "X_diff"] = -0.1
        df_regression.loc[df_regression.State == 0, "Y_diff"] = -0.1
        df_regression.loc[df_regression.State == 0, "Z_diff"] = -0.1        
#        df_regression["X_diff"][df_regression["State"] == 0] = -0.1
#        df_regression["Y_diff"][df_regression["State"] == 0] = -0.1
#        df_regression["Z_diff"][df_regression["State"] == 0] = -0.1       
        
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
        plt.title("Abs difference in X direction")
        
        
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
        plt.title("Abs difference in Y direction")
        
        
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
        plt.title("Abs difference in Z direction")
        
        
        #df_regression = df_regression[df_regression["State"] == 1] # state should be ok
        #df_regression = df_regression[df_regression["Conf_Score"] > 0.7] # confidence score > 0.7
        #TODO: what if no Z?
        plt.show()
        df_regression.to_csv("Regression_result.csv")
        return 




def Process(config):
    
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
            relocateCheck.commonLogAnalysis()
            plt.show()
        except:
            print("Relcoate check failed, is there any successful relocation? ")
                
    elif test_unit["TEST_MODE"] == 1:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        gps_log = test_unit["GPS_DIR"]
        regressionCheck = Regression()
        Fvtracker = False
        Fgps = True
        try:
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
try:
    f = open(yamlPath, 'r', encoding='utf-8')
except:
    f = open(yamlPath, 'r')
    
cfg = f.read()
config = yaml.load(cfg)
Process(config)
'''