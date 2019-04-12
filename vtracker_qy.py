#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:03:03 2019

@author: qy
"""

import pandas as pd
import matplotlib.pyplot as plt
import time


filename = "vtracker.log"
df_raw = pd.read_csv(filename,"\t",header = None, error_bad_lines=False)
df_raw = df_raw[0].str.split(" ", expand = True)
df_raw[3] = df_raw[3].str.split(":", expand = True)


df_common = df_raw[df_raw[3]=="common.h"]  # log from common.h
df_tracker = df_raw[df_raw[3]=="tracker.cc"]  # log from tracker.cc


## relocate
## tracker analysis

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
    plotName = ["reloc_opt", "thr=10","thr=50"]
    plt.legend(tuple(plotName), loc='upper right')
    n_pass = len(df[df[5] >= 10])
    plt.title("Times for solving pose opt: "+ str(df.shape[0])+
              " valid number:" + str(n_pass))   
 
def plot_reloc_proj(df):
    thr = [50] * df.shape[0]
    plt.plot(df[5].values)
    plt.plot(thr,"r--")
    plotName = ["Proj", "thr=50"]
    plt.legend(tuple(plotName), loc='upper right')
    n_pass = len(df[df[5] >= 50])
    plt.title("Times for searching by proj: "+ str(df.shape[0])+
              " valid number:" + str(n_pass))  
    
    
plt.subplot(2,2,1)    
plot_reloc_bow(df_reloc_bow)
plt.subplot(2,2,2)
plot_reloc_pnp(df_reloc_pnp)
plt.subplot(2,2,3)
plot_reloc_opt(df_reloc_opt)
plt.subplot(2,2,4)
plot_reloc_proj(df_reloc_proj)
plt.show()

