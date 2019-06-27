#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:39:10 2019

@author: qy
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import re


#%% get gps data frame
gps_dir = "cube_gps.txt"


try:
    df_gps = pd.read_csv(gps_dir,"\t",header = None, error_bad_lines=False)
except:
    print("can not read gps file, please check log dir")
    
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




#%% get vtracker data frame

vtracker_dir = "vtracker_test.log"
try:
    df_vtracker = pd.read_csv(vtracker_dir,"\t",header = None, error_bad_lines=False)
except:
    print("can not read vtracker file, please check log dir")
# initialization, save the raw data
# some logs somehow have double space split naive solution
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

# Get untracked images
#untrackedImgList = df_locate_res[df_locate_res["State"]==0]["Image_id"]




#%%
# Tracking rate
plt.subplot(2,1,1)
df_tracked = df_locate_res[df_locate_res["State"] == 1]
tracking_rate = df_tracked.shape[0]/df_locate_res.shape[0]
print("Total Frame Number: " + str(df_locate_res.shape[0]))
print("Tracking Rate: " + str(tracking_rate) )
#plot tracking state
plt.plot(df_locate_res["State"].values)
plt.title("Frame Number: " + str(df_locate_res.shape[0]) + "\n" +
          "Tracking Rate: " + str(tracking_rate))
#plt.xlabel("Frame" )
plt.ylabel("Tracking States"  + "\n" + 
          "1 : OK, 0 : LOST")


# Matches number

plt.subplot(2,1,2)
plt.plot(df_locate_res["Matches"].values)
plt.title("Frame Number: " + str(df_locate_res.shape[0]) + '\n'
          +"mean = " + "%.2f"%df_locate_res["Matches"].mean())
#plt.xlabel("Frame")
plt.ylabel("Matches Number")
plt.show()


#%% regression test

df_regression = pd.merge(df_locate_res, df_gps, on="Image_id").copy()
df_regression.columns = ["Image_id", "State", "X_loc", "Y_loc","Z_loc",
                         "Roll_loc", "Pitch_loc", "Yaw_loc", "Matches",
                         "X_gps", "Y_gps", "Z_gps", "Roll_gps", "Pitch_gps",
                         "Yaw_gps", "Conf_Score"]

# set unit from meter to centimeter
df_regression["X_diff"] = abs(df_regression["X_loc"] - df_regression["X_gps"])
df_regression["Y_diff"] = abs(df_regression["Y_loc"] - df_regression["Y_gps"])
df_regression["Z_diff"] = abs(df_regression["Z_loc"] - df_regression["Z_gps"])

# set lost diff to -1
#df_regression["X_diff"][df_regression["State"] == 0] = -0.1
#df_regression["Y_diff"][df_regression["State"] == 0] = -0.1
#df_regression["Z_diff"][df_regression["State"] == 0] = -0.1
df_regression.loc[df_regression.State == 0, "X_diff"] = -0.1
df_regression.loc[df_regression.State == 0, "Y_diff"] = -0.1
df_regression.loc[df_regression.State == 0, "Z_diff"] = -0.1   




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



# what if no Z?





plt.show()
