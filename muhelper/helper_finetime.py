import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy.signal

def finetime_calib(times, low=None, high=None):
    if low is None:
        times = np.array(times)
        mask = (times > 4)& (times < 1000)
        low = np.min(times[mask])
        high=np.max(times[mask])
        
    total_uncalib = high-low
    finetime_calib = (times-low)/total_uncalib*25
    
    return finetime_calib


def load_finetime_comp(filename, ch ="0", ftime_min=None, ftime_max=None, finetime_roll=0):
    
    if type(ch) is str:
        data  = pd.read_csv(filename, sep=';', on_bad_lines  = "skip", usecols=["CHARGE_"+ch, "FINE_"+ch,"RUN_EventTimecode_ns"])
        data[f"FINE_calib_{ch}"] = finetime_calib(data[f"FINE_{ch}"], low=ftime_min, high=ftime_max)

        if finetime_roll!=0:
            data[f"FINE_calib_{ch}"]= (data[f"FINE_calib_{ch}"]+finetime_roll)%25    
            
    return data


def load_finetime(filename, chs =["0","1"], ftime_min=None, ftime_max=None, finetime_roll=0):
    
    if type(chs) is list:
        keys_to_read = ['RUN_EventTimeCodeLSB', 'RUN_EventTimecode_ns', 'T0_to_Event_Timecode', 'T0_to_Event_Timecode_ns']
        for ch in chs:
            keys_to_read.append("CHARGE_"+ch)
            keys_to_read.append("FINE_"+ch)
            keys_to_read.append("COARSE_"+ch)
            keys_to_read.append("HIT_"+ch)

        data  = pd.read_csv(filename, sep=';', on_bad_lines  = "skip", usecols=keys_to_read)

        for ch in chs:
            data[f"FINE_calib_{ch}"] = finetime_calib(data[f"FINE_{ch}"], low=ftime_min, high=ftime_max)
            data[f"TIMESTAMP_{ch}"] = (data["COARSE_"+ch]+1)*25 - data[f"FINE_calib_{ch}"]

        return data
    
    elif type(chs) is str:
        return load_finetime_comp(filename, ch =chs, ftime_min=ftime_min, ftime_max=ftime_max, finetime_roll=finetime_roll)
    