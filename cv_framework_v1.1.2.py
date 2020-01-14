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
        python CV_FRAMEWORK_DIR/cv_framework.py CV_FRAMEWORK_LOG_DIR       
        or (if need tegra log analysis)
        python CV_FRAMEWORK_DIR/cv_framework.py CV_FRAMEWORK_LOG_DIR TEGRA_LOG_DIR
        such as:
        python cv_framework.py cv.log tegra.log
        
        
        v1.1.1 update:
        support log analysis with tegra log analysis
        
        v1.1.2 update:
        fix bug in timestamp process       
        
        
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
        print(S)
        
    def safePlot(self, df, col_name):
        ind = df[col_name].index
#        print(df[col_name][df[col_name].index[-1]])
        max_len = self.S[0]
        cur = np.zeros(max_len)
#        print(max_len, len(cur), ind, self.df[col_name][df[col_name].index[1]])
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
            plt.plot(ele["conf"][ele["conf"]>0])
#            self.safePlot(ele, 'conf')
            m = '%.4f'%ele["conf"].mean()
            mean_arr.append(m)
            mean_plot += str(m)+' '
        thr_conf=[0.7] * self.S[0]
#        print(self.df.shape[0])
        plt.plot(thr_conf, 'r--' )
        plotName = name + ['conf=0.7']
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Confidence Score, mean: "+ mean_plot)
        
        
    def plotFps(self, name):
        fps_arr = []
        fps_plot = ''
        for ele in self.df_arr:
#            ele.fillna(-1)
            plt.plot(ele["fps"][ele["fps"]>0])
            f = '%.2f'%ele["fps"].mean()
            fps_arr.append(f)
            fps_plot += str(f) + " "
        thr_fps=[7] * self.S[0]
        plt.plot(thr_fps, 'r--' )
        plotName = name + ['Fps=7']
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Fps mean: "+ fps_plot)
    
    def plotLatency(self, name):
        latency_arr = []
        latency_plot = " "
        for ele in self.df_arr:
            plt.plot(ele["latency"][ele["latency"]>0])
            l = '%.2f'%ele["latency"].mean()
            latency_arr.append(l)
            latency_plot += str(l) + " "
        thr_latency200=[0.2] * self.S[0]
        thr_latency375=[0.35] * self.S[0]
        plt.plot(thr_latency200, 'y--' )
        plt.plot(thr_latency375, 'r--')
        plotName = name + ["latency=0.2s", "latency=0.35s"]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("Latency mean: "+ latency_plot)
        plt.ylim(0,0.5)
        
    def plot_tegra_info(self,df):
        df_gpu_plot = df["gpu"][df["gpu"]>0]
        df_cpu_plot = df["cpu_load"][df["cpu_state"] == True]
        plt.plot(df_gpu_plot)
        plt.plot(df_cpu_plot/6)
        plotName = ["Gpu state", "Cpu state"]
        plt.legend(tuple(plotName), loc='upper right')
        plt.title("tegra info")


def checkCvFramework(log = "cv.log", tegra_log = ""):
    filename = log
    df_raw = pd.read_csv(filename, sep = '\n',header = None, error_bad_lines=False)
    df_raw = df_raw[0].str.split(' ', expand = True)
    S = df_raw.shape
    
#    print(S)
    
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
        
        try: 

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
            df_raw.to_csv("qytest.csv")
        except:
            df = df_raw[[0,2,11,12,13,14,15,18]].copy()
            
            #pick target columns
            df.columns = ["time","worker_id", "conf",
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
            df["time"] = df["time"].str.split(":INFO", expand = True)[0]
            df["time"] = df["time"].astype("str")    
            df_raw.to_csv("qytest.csv")

    #process time
    df_time = df["time"].str.split("[", expand = True)
    df["time"] = df_time[df_time.shape[-1]-1]
    df = df.sort_values(by = 'time')

    if tegra_log != "":
        df_tegra = pd.read_csv(tegra_log, sep='\t',header = None, error_bad_lines=False)
        df_tegra["tegra_raw_data"] = df_tegra[0]
        # process cpu info
        df_tegra_cpu = df_tegra["tegra_raw_data"].str.split("CPU", expand = True)[1]
        df_tegra_cpu = df_tegra_cpu.str.split("[",expand = True)[1]
        df_tegra_cpu = df_tegra_cpu.str.split("]",expand = True)[0]
        df_tegra_cpu = df_tegra_cpu.str.split(",",expand = True)
        df_tegra["cpu_0"] = df_tegra_cpu[0].str.split("%@",expand = True)[0].astype("float")
        df_tegra["cpu_1"] = df_tegra_cpu[1].str.split("%@",expand = True)[0].astype("float")
        df_tegra["cpu_2"] = df_tegra_cpu[2].str.split("%@",expand = True)[0].astype("float")
        df_tegra["cpu_3"] = df_tegra_cpu[3].str.split("%@",expand = True)[0].astype("float")
        df_tegra["cpu_4"] = df_tegra_cpu[4].str.split("%@",expand = True)[0].astype("float")
        df_tegra["cpu_5"] = df_tegra_cpu[5].str.split("%@",expand = True)[0].astype("float")
        #replace all Nan to -1
        df_tegra = df_tegra.fillna(-1)
        # check cpu state, should be 6 core, state == True => cpu state OK
        df_tegra["cpu_state"] = df_tegra["cpu_5"] != -1 
        df_tegra.eval("cpu_load = cpu_0 + cpu_1 + cpu_2 + cpu_3 + cpu_4 + cpu_5",inplace = True)
        # process gpu info
        df_tegra_gpu = df_tegra["tegra_raw_data"].str.split("GR3D_FREQ", expand = True)[1]
        df_tegra["gpu"] = df_tegra_gpu.str.split("%@", expand = True)[0].astype("float")
        #process time info
        df_tegra_time = df_tegra["tegra_raw_data"].str.split("]", expand = True)
        df_tegra_time = df_tegra_time[0].str.split("[", expand = True)
        df_tegra_time = df_tegra_time[1].str.split(" ", expand = True)
        df_tegra["date"] = df_tegra_time[0]
        df_tegra["time"] = df_tegra_time[1]
        
        # merge cv_framework and tegra data frame
        df_cv_plus_tegra = pd.merge(df, df_tegra, on="time",how = 'outer').copy()
        df_cv_plus_tegra = df_cv_plus_tegra.sort_values(by = 'time')
        
        df = df_cv_plus_tegra.copy()
        df = df.reset_index(drop = True)
    
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
    
#    print("stream: ", v_candidate)
    df_arr = []
    for i in range(len(v_candidate)):
        df_arr.append(df[df["stream_id"] == v_candidate[i]])
    legend_name = []
    for ele in v_candidate:
        legend_name.append('vslam'+str(ele))
    start_0 = df_arr[0]["time"][df_arr[0].index[0]].split('.')[0]
    end_0   = df_arr[0]["time"][df_arr[0].index[-1]].split('.')[0] 
    if start_0[0] == "[":
        start_0 = start_0[1:]
    if end_0[0] == "[":
        end_0 = end_0[1:]       
        

    # plot
    if tegra_log == "":
        
    #    CP = cvPlot(df_arr, df, S)
        CP = cvPlot(df_arr, df, S)
        plt.subplot(3,1,1)
        CP.plotConfScore(legend_name)
        
        plt.subplot(3,1,2)
        CP.plotFps(legend_name)
        
        plt.subplot(3,1,3)
        CP.plotLatency(legend_name)
        
        plt.xlabel("Date: From "+str(start_0) + ' to ' + str(end_0))
    
        plt.show()
        
    else:
        CP = cvPlot(df_arr, df, df.shape)
        plt.subplot(4,1,1)
        CP.plotConfScore(legend_name)
        
        plt.subplot(4,1,2)
        CP.plotFps(legend_name)
        
        plt.subplot(4,1,3)
        CP.plotLatency(legend_name)
        
        plt.subplot(4,1,4)
        CP.plot_tegra_info(df)
        
        plt.xlabel("Date: From "+str(start_0) + ' to ' + str(end_0))
    
        plt.show()        
        
        
        


if __name__ == "__main__":
    
    if sys.argv[-1] in ('help', '-help','--help'):
        Help().howToUseThisFile()
    elif len(sys.argv) == 2:
        cv_framework_log_dir = sys.argv[1]
        checkCvFramework(cv_framework_log_dir)
    elif len(sys.argv) == 3:
        cv_framework_log_dir = sys.argv[1]
        tegra_log_dir = sys.argv[2]
        checkCvFramework(cv_framework_log_dir, tegra_log_dir)

'''

filename = "/home/qy/Desktop/project/test/fengwei/cv_framework+tegra_plot/log_2020-01-09-11-27-14/slave_uos/uos_cv_framework.log"
tegra_log = "/home/qy/Desktop/project/test/fengwei/cv_framework+tegra_plot/log_2020-01-09-11-27-14/slave_uos/tegra_stats.log"
filename = "/home/qy/Desktop/uisee/work/redmine/AVP/25117/2+2_vslam_vogm/uos_cv_framework.log"
tegra_log = ''
checkCvFramework(filename, tegra_log)

'''   












