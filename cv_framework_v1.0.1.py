#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update at April 2nd 2019 13:33

Depend libraries:
    
python -m pip install numpy 
python -m pip install matplotlib    
python -m pip install pandas


Useage:
python CV_FRAMEWORK_DIR/cv_framework.py LOG       
such as:
python cv_framework.py cv.log


v1.0.0 can support 6 vlam plot with the new log format(update in March 2019) 
You might see there is a LONG and UGLY line in the plot. 
The long straight line means the vslam channel has no input during that time.




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


class Help:
    def __init__(self):
        
        return

    def using_tutorial():
        """
        Update at April 2nd 2019 13:33
        
        Depend libraries:
            
        python -m pip install numpy 
        python -m pip install matplotlib    
        python -m pip install pandas
        
        
        Useage:
        python CV_FRAMEWORK_DIR/cv_framework.py LOG       
        such as:
        python cv_framework.py cv.log
        
        
        v1.0.0 can support 6 vlam plot with the new log format(update in March 2019) 
        You might see there is a LONG and UGLY line in the plot. 
        The long straight line means the vslam channel has no input during that time.
        
        
        
        
        @author: qy
        """
        return
    
    def howToUseThisFile(self):
        
        return help(self.using_tutorial)

    
class cvPlot:
    
    def __init__(self, df_arr, df, S):
        self.df_arr = df_arr
        self.df = df
        self.S = S
        
    def safePlot(self, df, col_name):
        ind = df[col_name].index
        print(df[col_name][df[col_name].index[-1]])
        max_len = self.S[0]
        cur = np.zeros(max_len)
        print(max_len, len(cur), ind, self.df[col_name][df[col_name].index[1]])
        for i in range(len(ind)//300):
#            print(ele)
#            print(df[col_name].index[ele])
            cur[ind[i*300]] = df[col_name][df[col_name].index[i*300]]
#            cur[ele] = self.df[col_name][self.df[col_name].index[ele]]
#        print("finish!!")
        plt.plot(cur)
        
        


    def plotConfScore(self, name):
        mean_arr = []
        mean_plot = ''
        for ele in self.df_arr:
            plt.plot(ele["conf"])
#            self.safePlot(ele, 'conf')
            m = '%.4f'%ele["conf"].mean()
            mean_arr.append(m)
            mean_plot += str(m)+' '
        thr_conf=[0.7] * self.df.shape[0]
        print(self.df.shape[0])
        plt.plot(thr_conf, 'r--' )
        plotName = name + ['conf=0.7']
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Confidence Score, mean: "+ mean_plot)
        
        
    def plotFps(self, name):
        fps_arr = []
        fps_plot = ''
        for ele in self.df_arr:
            plt.plot(ele["fps"])
            f = '%.2f'%ele["fps"].mean()
            fps_arr.append(f)
            fps_plot += str(f) + " "
        thr_fps=[7] * self.df.shape[0]
        plt.plot(thr_fps, 'r--' )
        plotName = name + ['Fps=7']
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Fps mean: "+ fps_plot)
    
    def plotLatency(self, name):
        latency_arr = []
        latency_plot = " "
        for ele in self.df_arr:
            plt.plot(ele["latency"])
            l = '%.2f'%ele["latency"].mean()
            latency_arr.append(l)
            latency_plot += str(l) + " "
        thr_latency200=[0.2] * self.df.shape[0]
        thr_latency375=[0.35] * self.df.shape[0]
        plt.plot(thr_latency200, 'y--' )
        plt.plot(thr_latency375, 'r--')
        plotName = name + ["latency=0.2s", "latency=0.35s"]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Latency mean: "+ latency_plot)
        plt.ylim(0,0.5)

def checkCvFramework(log = "cv.log"):
    filename = log
    df_raw = pd.read_csv(filename,'\t',header = None, error_bad_lines=False)
    df_raw = df_raw[0].str.split(' ', expand = True)
    S = df_raw.shape
#    t_start = df_raw[1][0][:12]
#    t_end = df_raw[1][S[0]-1][:12]
    try:
        df = df_raw[[0,1,4,13,14,15,16,17,20]].copy()
        
        #pick target columns
        df.columns = ["date","time","worker_id", "conf",
                      "stream_id", "ret", "tstamp","fps","latency"]
        df = df[~df['latency'].isin([None])]
        
        # preprocess
        deSquareBracket(df, "conf")
        df["conf"] = df["conf"].astype("float")
        deSquareBracket(df, "stream_id")
    #    df["stream_id"] = df["stream_id"].astype("float")
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
    except:
        df = df_raw[[0,1,4,12,13,14,15,16,19]].copy()
        
        #pick target columns
        df.columns = ["date","time","worker_id", "conf",
                      "stream_id", "ret", "tstamp","fps","latency"]
        df = df[~df['latency'].isin([None])]
        
        # preprocess
        deSquareBracket(df, "conf")
        df["conf"] = df["conf"].astype("float")
        deSquareBracket(df, "stream_id")
    #    df["stream_id"] = df["stream_id"].astype("float")
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
    end_df   = df["time"][df.index[-2]][:-1].split('.')[0]

    
    # check which stream is activated
    streamMaxSize = 6
    streamMaxCapacity = 6
    
    v_candidate = []
    for i in range(streamMaxSize):
        if len(v_candidate) == streamMaxCapacity:
            break
        else:
            if df[df["stream_id"] == str(i)].shape[0]>10:
                v_candidate.append(str(i))
    
    print("stream: ", v_candidate)
    df_arr = []
    for i in range(len(v_candidate)):
        df_arr.append(df[df["stream_id"] == v_candidate[i]])
    legend_name = []
    for ele in v_candidate:
        legend_name.append('vslam'+str(ele))
    start_0 = df_arr[0]["time"][df_arr[0].index[0]].split('.')[0]
    end_0   = df_arr[0]["time"][df_arr[0].index[-1]].split('.')[0] 
        
    # plot
    CP = cvPlot(df_arr, df, S)
    plt.subplot(3,1,1)
    CP.plotConfScore(legend_name)
    
    plt.subplot(3,1,2)
    CP.plotFps(legend_name)
    
    plt.subplot(3,1,3)
    CP.plotLatency(legend_name)
    
    plt.xlabel("Date: " +str(df_arr[0]["date"][df_arr[0].index[0]]) +" From "+str(start_0) + ' to ' + str(end_0))
    plt.show()
        
        
        
        
'''        
    df0 = df[df["stream_id"] == v_candidate[0]]
    df1 = df[df["stream_id"] == v_candidate[1]]
#    start_0 = df0["time"][df0.index[0]].split('.')[0]
#    end_0   = df0["time"][df0.index[-1]].split('.')[0]
#    start_1 = df1["time"][df1.index[0]].split('.')[0]
#    end_1   = df1["time"][df1.index[-1]].split('.')[0]
    
    # plot result
    
    
    v1_name = "vslam" + str(v_candidate[0])
    v2_name = "vslam" + str(v_candidate[1])
    # confidence score
    plt.subplot(3,1,1)
    plt.plot(df0["conf"])
    plt.plot(df1["conf"])
    thr_conf=[0.7] * df.shape[0]
    plt.plot(thr_conf, 'r--' )
    mean_conf0 = '%.4f'%df0["conf"].mean()
    mean_conf1 = '%.4f'%df1["conf"].mean()
    plt.legend((v1_name, v2_name, "conf=0.7"),loc='upper right')
    plt.title("Confidence Score, mean: "+str(mean_conf0)+', '+str(mean_conf1))
    
    #fps
    plt.subplot(3,1,2)
    plt.plot(df0["fps"])
    plt.plot(df1["fps"])
    thr_fps=[7] * df.shape[0]
    plt.plot(thr_fps, 'r--' )
    mean_fps0 = '%.2f'%df0["fps"].mean()
    mean_fps1 = '%.2f'%df1["fps"].mean()
    plt.legend((v1_name, v2_name, "Fps=7"),loc='upper right')
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
    
    plt.legend((v1_name, v2_name,"latency=0.2s","latency=0.35s"),loc='upper right')
    plt.title("Latency mean: "+ str(mean_latency0)+ ', '+ str(mean_latency1))
    
    plt.xlabel("Date: " +str(df0["date"][df0.index[0]]) +" From "+str(start_df) + ' to ' + str(end_df))
    plt.show()

'''

if __name__ == "__main__":
    filename = sys.argv[1]
    if sys.argv[-1] in ('help', '-help','--help'):
        Help().howToUseThisFile()
    else:
        checkCvFramework(filename)









