#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:25:27 2019

@author: qy
"""
import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def checkVslam(filename= "data/uos_navigation_20190306_155438.log.bak.20190306155438"):
    
    df = pd.read_csv(filename, '\t')
    df.columns = ['test']
    
    # be careful about the form of input 
    # the log should start with "navi_log_input_"
    df2 = df['test'].str.split('navi_log_input_',n=1,expand = True)
    df2.columns = ['t1', 't2']
    tar =''
    df3 = df2[df2['t1'] == tar]['t2']
    
    # split by ' '
    df_vslam = df3.str.split(' ', expand = True)
    df_final = df_vslam.copy()
    
    # 8,73,121 should be the col number of t_navi, vslam_0, vslam_1
    df_final = df_final[[8,73,121]] 
    df_final.columns = ['t_navi',  't_vslam_0', 't_vslam_1']
    
    # calculate latency 
    df_final['diff_t0'] = df_final['t_navi'].astype("float")-df_final['t_vslam_0'].astype("float")
    df_final['diff_t1'] = df_final['t_navi'].astype("float")-df_final['t_vslam_1'].astype("float")
    df_final[df_final['diff_t0' ] > 300000] = 0 #clip 
    df_final[df_final['diff_t1' ] > 300000] = 0
    df_final[df_final['diff_t0' ] > 1] = 1 #clip 
    df_final[df_final['diff_t1' ] > 1] = 1
    plt.plot(df_final[['diff_t0', 'diff_t1']], label = 't0')
    
    thr_safe=[0.5] * df.shape[0]
    plt.plot(thr_safe, 'r--' )
    thr_notDetected = [0] * df.shape[0]
    plt.plot(thr_notDetected, 'c--')
    
    plt.xlabel("log")
    plt.ylabel("Latency")
    plt.title("Vslam latency check")
    plt.legend(("vslam0", "vslam1", "loss time thr","no input"), 
               loc='upper right')
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    checkVslam(filename)
    
    

