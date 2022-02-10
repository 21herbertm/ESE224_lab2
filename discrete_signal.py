# Melanie Herbert, ALina Ho
#ESE 224
import numpy as np

class sqpulse(object):
 # discrete square pulse signal of width M, with signal length N.
 # Arguments:
 #       T0: width of square pulse (int)
 #       T: duration of the signal (int)
 #       fs: sampling frequncy (int)

    def __init__(self, T0, T, fs):
        self.N=np.int(np.floor(T*fs))
        self.M=np.int(np.floor(T0*fs))
        # using the equation T0= MTs where M is the total duration n<M
        # Create the square pulse of width M
        self.pulse=np.concatenate((np.ones(self.M),np.zeros(self.N-self.M)))
        self.pulse*=1/np.sqrt(self.M)

        # Create the time array
        self.t=np.arange(0,T,1/fs)

# square pulse is less smooth than the triangular pulse
# Looking at the square pulse graph, there is a sharp transition which signifies cosines of higher frequencies.
# DFT becomes more concentrated for wider pulses


class tripulse(object):
# this class generates a unit-energy triangular pulse
# tripulse(T0,T,fs) generates a unit-energy triangular pulse x(t) of
# duration T sampled at frequency fs and with active duration T0.
# Vector t contains the time indices (x axis) and vector x contains
# the values of the signal (y axis).

 # T0: width of triangular pulse (int)
 # T: duration of the signal (int)
 # fs: sampling frequncy (int)

    def __init__(self, T0, T, fs):
        self.N=np.int(np.floor(T*fs))
        self.M=np.int(np.floor(T0*fs))

        # Create the active part
        ascendent_part = np.arange(0,np.ceil(T0/2*fs))
        descendent_part = np.arange(np.ceil(T0/2*fs)-1,0,-1)

        # Construct the triangular pulse
        self.pulse=np.concatenate((ascendent_part,descendent_part,np.zeros(self.N-len(ascendent_part)-len(descendent_part))))

        # Normalize
        self.pulse*=1/np.linalg.norm(self.pulse)

        # Create the time array
        self.t=np.arange(0,T,1/fs)


class kaiser_window(object):
# Generates a kaiser window signal
# Arguments:
    # beta: beta parameter of the Kaiser window
    # T: duration of the signal (int)
    # fs: sampling frequncy (int)

    def __init__(self, beta, T, fs):
        self.N = np.int(np.floor(T * fs))

        # Create the active part
        self.signal=np.kaiser(self.N,beta)

        # Normalize
        self.signal *= 1 / np.linalg.norm(self.signal)

        # Create the time array
        self.t = np.arange(0, T, 1 / fs)

#We compared the Kaiser Window to spectra of square and triangular pulses.
# The Kaiser signal graph shows a smoother graph than both the square and triangular pulses.
# This in turn is shown by less components of high frequency.
