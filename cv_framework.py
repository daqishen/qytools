#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:38:38 2019

Libraries:
    
python -m pip install numpy 
python -m pip install matplotlib    
python -m pip install pandas


Example:
       
python cv_framework.py cv.log



@author: qy
"""

import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def deSquareBracket(pdframe, col_name):
    pdframe[col_name] = pdframe[col_name].str.split("[", expand = True)[1]
    pdframe[col_name] = pdframe[col_name].str.split("]", expand = True)[0]



def checkCvFramework(log = "cv.log"):
    filename = log
    df_raw = pd.read_csv(filename,'\t',header = None)
    df_raw = df_raw[0].str.split(' ', expand = True)
    S = df_raw.shape
    t_start = df_raw[1][0][:12]
    t_end = df_raw[1][S[0]-1][:12]
    df = df_raw[[0,1,4,12,13,14,15,16,19]].copy()
    
    #pick target columns
    df.columns = ["date","time","worker_id", "conf",
                  "stream_id", "ret", "tstamp","fps","latency"]
    df = df[~df['latency'].isin([None])]
    
    # preprocess
    deSquareBracket(df, "conf")
    df["conf"] = df["conf"].astype("float")
    deSquareBracket(df, "stream_id")
    df["stream_id"] = df["stream_id"].astype("float")
    deSquareBracket(df, "ret")
    df["ret"] = df["ret"].astype("float")
    deSquareBracket(df, "tstamp")
    df["tstamp"] = df["tstamp"].astype("float")
    deSquareBracket(df, "fps")
    df["fps"] = df["fps"].astype("float")
    deSquareBracket(df, "latency")
    df["latency"] = df["latency"].astype("float")
    df["date"] = df["date"].str.split("[", expand = True)[1]
    df["date"] = df["date"].astype("str")
    df["time"] = df["time"].str.split(":INFO", expand = True)[0]
    df["time"] = df["time"].astype("str")
    
    start_df  = df["time"][df.index[0]][:-1].split('.')[0]
    end_df   = df["time"][df.index[-1]][:-1].split('.')[0]
    
    # separate vslam0 and vslam1
    df0 = df[df["stream_id"] == 0]
    df1 = df[df["stream_id"] == 1]
#    start_0 = df0["time"][df0.index[0]].split('.')[0]
#    end_0   = df0["time"][df0.index[-1]].split('.')[0]
#    start_1 = df1["time"][df1.index[0]].split('.')[0]
#    end_1   = df1["time"][df1.index[-1]].split('.')[0]
    
    # plot result
    
    # confidence score
    plt.subplot(3,1,1)
    plt.plot(df0["conf"])
    plt.plot(df1["conf"])
    thr_conf=[0.7] * df.shape[0]
    plt.plot(thr_conf, 'r--' )
    mean_conf0 = '%.4f'%df0["conf"].mean()
    mean_conf1 = '%.4f'%df1["conf"].mean()
    plt.legend(("vslam0", "vslam1", "conf=0.7"),loc='upper right')
    plt.title("Confidence Score, mean: "+str(mean_conf0)+', '+str(mean_conf1))
    
    #fps
    plt.subplot(3,1,2)
    plt.plot(df0["fps"])
    plt.plot(df1["fps"])
    thr_fps=[7] * df.shape[0]
    plt.plot(thr_fps, 'r--' )
    mean_fps0 = '%.2f'%df0["fps"].mean()
    mean_fps1 = '%.2f'%df1["fps"].mean()
    plt.legend(("vslam0", "vslam1", "Fps=7"),loc='upper right')
    plt.title("Fps mean: "+str(mean_fps0)+', '+str(mean_fps1))
    
    #latency
    plt.subplot(3,1,3)
    plt.plot(df0["latency"])
    plt.plot(df1["latency"])
    thr_latency200=[0.2] * df.shape[0]
    thr_latency375=[0.35] * df.shape[0]
    plt.plot(thr_latency200, 'y--' )
    plt.plot(thr_latency375, 'r--')
    mean_latency0 = '%.2f'%df0["latency"].mean()
    mean_latency1 = '%.2f'%df1["latency"].mean()
    
    plt.legend(("vslam0", "vslam1","latency=0.2s","latency=0.35s"),loc='upper right')
    plt.title("Latency mean: "+ str(mean_latency0)+ ', '+ str(mean_latency1))
    
    plt.xlabel("Date: " +str(df0["date"][df0.index[0]]) +" From "+str(start_df) + ' to ' + str(end_df))
    plt.show()



if __name__ == "__main__":
    filename = sys.argv[1]
    checkCvFramework(filename)









