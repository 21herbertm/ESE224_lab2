# Melanie Herbert, ALina Ho
# ESE 224
# NEW class for project 2


import numpy as np
import cmath

# IN PART 3, ASKED TO USE CLASS DFT PROVIDED ON THE COURSE WEBSITE
# TYPE HELP DFT IN THE CONSOLE IN ORDER TO SEE HOW THIS WORKS
class dft():
    def __init__(self, x, fs, K=None):

        # :param x: Input vector x contains the discrete signal
        # :param fs: Input integer fs contains the sample frequency
        # :param K: Input positive integer that determines the number of coeffients
        # used to calculate the DFT. If K is not provided, K=length(x).

    # START: SANITY CHECK OF INPUTS.
        if (type(fs) != int) or (fs<=0):
            raise NameError('The frequency fs should be a positive integer.')
        if not isinstance(x, np. ndarray):
            raise NameError('The input signal x must be a numpy array.')
        if isinstance(x, np. ndarray):
            if x.ndim!=1:
                raise NameError('The input signal x must be a numpy vector array.')
        self.x=x
        self.fs=fs
        self.N=len(x)
        if K == None:
            K = len(self.x)
        # START: SANITY CHECK OF INPUTS.
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
    def changeK(self,K):
        """
        :param K: Input positive integer that determines the number of coeffients
        used to calculate the DFT. This function changes the attribute K of the class.
        """
        if (type(K) != int) or (K <= 0) or (K <  0):
            raise NameError('K should be a positive integer.')
        old_K=self.K
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
        # This accounts for the frequencies
        # centered at zero. I want to be guaranteed that k=0 is always a
        # possible k. Then, I also have to account for both even and odd choices
        # of K, and that's why the floor() function appears to round down the
        # numbers.
        print('The value of K was succefully change from %d to %d'%(old_K,self.K))
        pass

        matrix_k=np.transpose(np.tile(np.arange(self.K),(self.N,1)))
        matrix_n=np.tile(np.transpose(np.arange(self.N)),(self.K,1))
        indices=np.multiply(matrix_k,matrix_n)
        WKN=1/np.sqrt(self.N)*np.exp(-1j*2*np.pi*indices/self.K)
        X=WKN@self.x

        X_c=np.roll(X,np.int(np.ceil(self.K/2)-1)) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]

    def solveLoops(self):
        #DFT using for loop
        X = np.zeros(self.K, dtype=np.complex_)
        for k in range(self.K):  # For each time index k=0,1,...,K;
            for n in range(self.N):  # For each frequency n=0,1,...,N-1:
                X[k] = X[k] + 1 / np.sqrt(self.N) * self.x[n] * np.exp(-1j * 2 * cmath.pi * k * n / self.K)

        X_c = np.roll(X, np.int(np.ceil(self.K / 2) - 1))  # Circularly shift X to get it centered in f_c==0
        return [self.f, X, self.f_c, X_c]


# calling solve3 in main, most efficient method
    def solve3(self):
        X=np.fft.fft(self.x,self.K)/np.sqrt(self.N);
        # \\\\\ CENTER FFT.
        X_c=np.roll(X,np.int(np.ceil(self.K/2-1))) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]
