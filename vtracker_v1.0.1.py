#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:34:02 2019

@author: qy
"""

import sys
import pandas as pd
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
        
        
        Useage:
        python VTRACKER_DIR/vtracker_va.b.c.py LOG       
        such as:
        python vtracker_v1.0.0.py vtracker.log
        
        
        vtracker v1.0.0 can support relocate analysis in vtracker log
        THE INPUT ORDER IS NOT NECESSARILY EQUAL TO INPUT TIME ORDER
        
        
        
        
        @author: qy
        """
        return
    def howToUseThisFile(self):
        
        return help(self.using_tutorial)


class Relocate:
    
    def __init__(self, df_raw):
        # initialization, save the raw data
        df_raw = df_raw[0].str.split("    ",expand = True)    
        df_raw = df_raw[0].str.split("   ",expand = True)
        df_raw = df_raw[0].str.split("  ",expand = True)
        # some logs somehow have double space split 
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
        df_reloc_bow = df_tracker[df_tracker[4] == "reloc_bow:"]
        df_reloc_bow[5] = df_reloc_bow[5].astype("float")
        df_reloc_pnp = df_tracker[df_tracker[4] == "reloc_pnp:"]
        df_reloc_pnp[5] = df_reloc_pnp[5].astype("float")
        df_reloc_opt = df_tracker[df_tracker[4] == "reloc_opt:"]
        df_reloc_opt[5] = df_reloc_opt[5].astype("float")
        df_reloc_proj = df_tracker[df_tracker[4] == "reloc_proj:"]
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
        
        plt.figure("tracker.cc relocate analysis ") #figure 1
        plt.subplot(2,2,1)    
        plot_reloc_bow(df_reloc_bow)
        plt.subplot(2,2,2)
        plot_reloc_pnp(df_reloc_pnp)
        plt.subplot(2,2,3)
        plot_reloc_opt(df_reloc_opt)
        plt.subplot(2,2,4)
        plot_reloc_proj(df_reloc_proj)

        plt.show()
        
        
    
    
if __name__ == "__main__":
    filename = sys.argv[1]
    if sys.argv[-1] in ('help', '-help','--help'):
        Help().howToUseThisFile()
    else:
        try:
            df_raw = pd.read_csv(filename,"\t",header = None, error_bad_lines=False)
        except:
            print("can not read log file, please check log dir")
            Help().howToUseThisFile()
        else:
            relocateCheck = Relocate(df_raw)
            relocateCheck.trackerAnalysis()
            
