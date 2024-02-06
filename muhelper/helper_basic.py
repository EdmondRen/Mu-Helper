
# Python standard library
import inspect
import os, sys
import importlib
import time
from importlib import reload
from tqdm import tqdm

# Other libraries
import scipy
import scipy.optimize
import numpy as np

from pylab import *


    
def Uniform(x,A):
    if type(x) not in [np.ndarray,list]:
        return A
    else:
        return A*np.ones_like(x)
def Exp(x,A,t):
    return A*np.exp(-x/t)
def Gauss(x, A, mean, sigma):
    return A * np.exp(-(x - mean)**2 / (2 * sigma**2)) 
def Gauss_sideband(x, A, mean, sigma, a1,a2):
    # a1 for left, a2 for right
    return Utils.Gauss(x, A, mean, sigma) + sqrt(2*np.pi)*sigma/2*(a1*scipy.special.erfc((x-mean)/sqrt(2)/sigma) + a2*(2-scipy.special.erfc((x-mean)/sqrt(2)/sigma))) 
def Poisson(k, Lambda,A):
    # Lambda: mean, A: amplitude
    return A*(Lambda**k/scipy.special.factorial(k)) * np.exp(-Lambda)
def Poly(x, *P):
    '''
    Compute polynomial P(x) where P is a vector of coefficients
    Lowest order coefficient at P[0].  Uses Horner's Method.
    '''
    result = 0
    for coeff in P[::-1]:
        result = x * result + coeff
    return result

def Pulse(x, A, x0 = 0, tau1=2, tau2=20):
    # Fs: samples per s
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    kernel = (np.exp(-dx/tau1)-np.exp(-dx/tau2))/(tau1-tau2)*np.heaviside(dx,1)
    kernel_normed = kernel/np.max(kernel)
    return kernel_normed*A

def Pulse2(x, tau_r, tau_f1, tau_f2, A1, A2, t0, A0):
    # Test: plot(np.linspace(-50,50,200), Pulse3(np.linspace(-50,50,200), 2,4,10,20,3,2,1,10.5,5))
    times=x-t0
    mask =times>0
    pulse = (A1     *     (np.exp(-times[mask] / tau_f1))
            +A2     *     (np.exp(-times[mask] / tau_f2))
            -(A1+A2) * (np.exp(-times[mask] / tau_r))
            )
    pulse = np.concatenate((np.zeros(sum(~mask)), pulse))
    pulse/=max(pulse)
    pulse*=A0
    return pulse

def Pulse3(x, tau_r, tau_f1, tau_f2, tau_f3, A1, A2, A3, t0, A0):
    # Test: plot(np.linspace(-50,50,200), Pulse3(np.linspace(-50,50,200), 2,4,10,20,3,2,1,10.5,5))
    times=x-t0
    mask =times>0
    pulse = (A1     *     (np.exp(-times[mask] / tau_f1))
            +A2     *     (np.exp(-times[mask] / tau_f2))
            +A3     *     (np.exp(-times[mask] / tau_f3))
            -(A1+A2+A3) * (np.exp(-times[mask] / tau_r))
            )
    pulse = np.concatenate((np.zeros(sum(~mask)), pulse))
    pulse/=max(pulse)
    pulse*=A0
    return pulse

def Chi2(x, dof, A):
    return scipy.stats.chi2.pdf(x,dof)*A 


def pulse_2pole(t_rise_ns, t_fall_ns, samples_per_ns, total_samples=8000, pre_trig_samples=0):
    total_time = total_samples/samples_per_ns
    x = np.linspace(0,total_time, total_samples)
    x0 = pre_trig_samples/samples_per_ns
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    kernel = (np.exp(-dx/t_fall_ns)-np.exp(-dx/t_rise_ns))/np.heaviside(dx,1)
    kernel_normed = kernel/max(kernel)#(np.dot(kernel,kernel/max(kernel)))
    
    return kernel_normed


def pulse_2pole_old(pre_trig, post_trig, tau1=2,tau2=20, Fs = 100):
    # Fs: samples per s
    x = np.arange(0,(pre_trig+post_trig))/Fs
    x0 = pre_trig/Fs
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    kernel = (np.exp(-dx/tau1)-np.exp(-dx/tau2))/(tau1-tau2)*np.heaviside(dx,1)
    # kernel_normed = kernel/(np.dot(kernel,kernel/max(kernel)))
    kernel_normed = kernel/np.max(kernel)
    return kernel_normed

def roll_zeropad(a, shift):
    y = np.roll(a,shift)
    y[:shift]=0
    return y

def slope(y, x=None):
    x = np.arange(len(y)) if x is None else x
    slope,*_ = scipy.stats.linregress(x, y)
    return slope









def fstr(template, scope):
    """
    Evaluate the f string later at a given scope
    """
    return eval(f"f'{template}'", scope)  

def fit_curve(f, xdata, ydata, makeplot = True, label=None, fit_range=None, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-np.inf, np.inf), method=None, jac=None, **kwargs):
   
    if type(f) is str:
        if f in ["Gauss","gauss","gaus"]:
            f=Gauss
            mean = np.sum(xdata*ydata)/np.sum(ydata)
            p0 = [np.max(ydata), mean, np.sqrt(np.sum(ydata*(xdata-mean)**2)/(np.sum(ydata)-1))] if p0 is None else p0
        elif f in ["Exp", "exp"]:
            f=Exp
            
    
    # Fit range
    fit_range = [np.min(xdata), np.max(xdata)] if fit_range is None else fit_range
    mask = (xdata>=fit_range[0]) &(xdata<=fit_range[1])
    xdata = xdata[mask]
    ydata = ydata[mask]   
    
    # Get keyword arguments:
    curvefit_kwargs = {}
    for key, value in kwargs.items() :
        if key in inspect.getfullargspec(scipy.optimize.curve_fit)[0] or key in ["maxfev"] :
            curvefit_kwargs[key] = value
    # for key, value in kwargs.items():    
            
    popt, pcov, info, *_ = scipy.optimize.curve_fit(f, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, check_finite=check_finite, bounds=bounds, method=method, jac=jac, full_output=True, **curvefit_kwargs)   
    
    if makeplot:
        xdata_plot = np.linspace(*fit_range, 200)
        ydata_plot = f(xdata_plot, *popt)
        
        scope = locals()
        func_parameter_names = inspect.getfullargspec(f)[0][1:]
        label = fstr(label,scope) if label is not None else "Fit"
        
        # Get keyword arguments:
        plotkwargs = {}
        for key, value in kwargs.items() :
            if key in ["color", "marker", "linestyle"] :
                plotkwargs[key] = value
        # for key, value in kwargs.items():
        plot(xdata_plot, ydata_plot, label=label, **plotkwargs)    
    
    return popt, pcov, info, f
    
    
def fit_hist(f, h, makeplot = True, label=None, fit_range=None, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-np.inf, np.inf), method=None, jac=None, full_output=False, nan_policy=None, **kwargs):

    
    # Limit fit range
    xdata = 0.5*(h[1][:-1]+h[1][1:])
    ydata = h[0]
    fit_range = [xdata[0], xdata[-1]] if fit_range is None else fit_range
    mask = (xdata>=fit_range[0]) &(xdata<=fit_range[1])
    xdata = xdata[mask]
    ydata = ydata[mask]
    
    # Calculate uncertainty
    sigma = np.sqrt(ydata) if sigma is None else sigma; sigma[sigma==0] =1
    
    # Fit
    popt, pcov, info, f = fit_curve(f, xdata, ydata, makeplot = makeplot, label=label, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, check_finite=check_finite, bounds=bounds, method=method, jac=jac, **kwargs)

    return popt, pcov, info, f





from typing import List, Tuple
import scipy.ndimage

def constant_fraction_discriminator(waveform: List[float], baseline: float, threshold: float, fraction: float, gauss_filter = None) -> List[Tuple[float, int]]:
    """
    This function takes in a waveform, baseline, threshold, and fraction as input and returns a list of tuples containing
    the amplitude and sample number of the leading edge of each pulse.
    """
    leading_edges = []
    triggered = False
    
    if gauss_filter is not None:
        waveform = scipy.ndimage.gaussian_filter(waveform,sigma=gauss_filter,)
    
    for i in range(1, len(waveform)):
        if waveform[i] > baseline + threshold:
            if triggered:
                continue
                
            for j in range(i - 1, -1, -1):
                if waveform[j] < baseline + fraction * (waveform[i] - baseline):
                    leading_edges.append((waveform[i], i))
                    triggered=True
                    break
        else:
            triggered = False
    return leading_edges



# trace = data_save[1][0]
# trace-=np.mean(trace[:1600])
# trace = -trace
# trace/=15.8
# leading_edges = constant_fraction_discriminator(trace, 0, 0.02, 0.5, gauss_filter=4)
# plot(trace)
# print(leading_edges)
# axvline(leading_edges[0][1])


    
def float_to_ADU(waveform, bits=14):
    # SCPIcmd = ":DATA1 VOLATILE, "
    # strData = ""
    # for i in waveform:
    #     strData+=f"{i:.7f},"
    # strData=strData[:-1]   
    # SCPIcmd = SCPIcmd + strData
    waveform_new=waveform/np.max(np.abs(waveform))
    waveform_new=np.round(waveform_new* 2**(bits-1))
    return waveform_new    





# Generate white noise
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    """
    Generate band limited noise
    The returned noise density is 1/rtHz
    """
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)*np.sqrt(samples*samplerate/2)