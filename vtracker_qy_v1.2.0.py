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
import vslam


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
            relocateCheck = vslam.Relocate(df_raw)
            relocateCheck.trackerAnalysis()
            relocateCheck.commonLogAnalysis()
            plt.show()
        except:
            print("Relcoate check failed, is there any successful relocation? ")
                
    elif test_unit["TEST_MODE"] == 1:
        vtracker_log = test_unit["VTRACKER_LOG_DIR"]
        gps_log = test_unit["GPS_DIR"]
        regressionCheck = vslam.Regression()
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