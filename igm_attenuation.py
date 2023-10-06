import numpy as np
import astropy.units as u














m0=0.3; l0=0.7; h=0.7
H0fac=3.241e-18 #multiply by h to get H0 in per second
c_light=2.99792e8

def getE(z): #redshift evolution of the Hubble parameter
    return np.sqrt(m0*(1.+z)**3 + l0)

def getDM(z): # Hogg eqns 15 and 16
    intlen=int(1e4)
    zs=np.linspace(0,z,intlen)
    return c_light/(H0fac*h)*np.trapz(1./getE(zs),zs) #in meters
    
#rather than computing the conversion factor between observed and absolute UV mag for each object
#just create an array with finely spaced redshift binning to save some computation time
dz = 0.001
zarr = np.arange(7.0,13.+dz/2.,dz)
DLarr = np.array([getDM(z) for z in zarr])*(1.+zarr)
magfacarr = -5.*np.log10(DLarr/3.086e17)+2.5*np.log10(1.+zarr) #Muv = mag + magfac

##############################################################

#code for the Inoue IGM transmission

lambda_j=np.array([1215.67,1025.72,972.537,949.743,937.803,930.748,926.226,923.150,920.963,
                   919.352,918.129,917.181,916.429,915.824,915.329,914.919,914.576,914.286,914.039,
                   913.826,913.641,913.480,913.339,913.215,913.104,913.006,912.918,912.839,912.768,
                   912.703,912.645,912.592,912.543,912.499,912.458,912.420,912.385,912.353,912.324])
A_jLAF1=np.array([1.690e-02,4.692e-03,2.239e-03,1.319e-03,8.707e-04,6.178e-04,4.609e-04,
                  3.569e-04,2.843e-04,2.318e-04,1.923e-04,1.622e-04,1.385e-04,1.196e-04,1.043e-04,
                  9.174e-05,8.128e-05,7.251e-05,6.505e-05,5.868e-05,5.319e-05,4.843e-05,4.427e-05,
                  4.063e-05,3.738e-05,3.454e-05,3.199e-05,2.971e-05,2.766e-05,2.582e-05,2.415e-05,
                  2.263e-05,2.126e-05,2.000e-05,1.885e-05,1.779e-05,1.682e-05,1.593e-05,1.510e-05])
A_jLAF2=np.array([2.354e-03,6.536e-04,3.119e-04,1.837e-04,1.213e-04,8.606e-05,6.421e-05,
                  4.971e-05,3.960e-05,3.229e-05,2.679e-05,2.259e-05,1.929e-05,1.666e-05,1.453e-05,
                  1.278e-05,1.132e-05,1.010e-05,9.062e-06,8.174e-06,7.409e-06,6.746e-06,6.167e-06,
                  5.660e-06,5.207e-06,4.811e-06,4.456e-06,4.139e-06,3.853e-06,3.596e-06,3.364e-06,
                  3.153e-06,2.961e-06,2.785e-06,2.625e-06,2.479e-06,2.343e-06,2.219e-06,2.103e-06])
A_jLAF3=np.array([1.026e-04,2.849e-05,1.360e-05,8.010e-06,5.287e-06,3.752e-06,2.799e-06,
                  2.167e-06,1.726e-06,1.407e-06,1.168e-06,9.847e-07,8.410e-07,7.263e-07,6.334e-07,
                  5.571e-07,4.936e-07,4.403e-07,3.950e-07,3.563e-07,3.230e-07,2.941e-07,2.689e-07,
                  2.467e-07,2.270e-07,2.097e-07,1.943e-07,1.804e-07,1.680e-07,1.568e-07,1.466e-07,
                  1.375e-07,1.291e-07,1.214e-07,1.145e-07,1.080e-07,1.022e-07,9.673e-08,9.169e-08])
A_jDLA1=np.array([1.617e-04,1.545e-04,1.498e-04,1.460e-04,1.429e-04,1.402e-04,1.377e-04,
                  1.355e-04,1.335e-04,1.316e-04,1.298e-04,1.281e-04,1.265e-04,1.250e-04,1.236e-04,
                  1.222e-04,1.209e-04,1.197e-04,1.185e-04,1.173e-04,1.162e-04,1.151e-04,1.140e-04,
                  1.130e-04,1.120e-04,1.110e-04,1.101e-04,1.091e-04,1.082e-04,1.073e-04,1.065e-04,
                  1.056e-04,1.048e-04,1.040e-04,1.032e-04,1.024e-04,1.017e-04,1.009e-04,1.002e-04])
A_jDLA2=np.array([5.390e-05,5.151e-05,4.992e-05,4.868e-05,4.763e-05,4.672e-05,4.590e-05,
                  4.516e-05,4.448e-05,4.385e-05,4.326e-05,4.271e-05,4.218e-05,4.168e-05,4.120e-05,
                  4.075e-05,4.031e-05,3.989e-05,3.949e-05,3.910e-05,3.872e-05,3.836e-05,3.800e-05,
                  3.766e-05,3.732e-05,3.700e-05,3.668e-05,3.637e-05,3.607e-05,3.578e-05,3.549e-05,
                  3.521e-05,3.493e-05,3.466e-05,3.440e-05,3.414e-05,3.389e-05,3.364e-05,3.339e-05])

#just doing the z>4.7 cases
def getT(lambda_obs_array,z): #returns an array of transmission values
    T=np.zeros_like(lambda_obs_array)
    lolen = len(lambda_obs_array)
    tauLAFLS,tauDLALS = np.zeros((lolen,39)), np.zeros((lolen,39))
    for j in range(39):
        lj = lambda_j[j]
        ldivs = lambda_obs_array/lj
        doOnes = (lambda_obs_array>lj) & (lambda_obs_array<lj*(1.+z))
        doOnes1 = doOnes & (lambda_obs_array<2.2*lj)
        tauLAFLS[doOnes1,j] = A_jLAF1[j]*ldivs[doOnes1]**1.2
        doOnes2 = doOnes & (lambda_obs_array>2.2*lj) & (lambda_obs_array<5.7*lj)
        tauLAFLS[doOnes2,j] = A_jLAF2[j]*ldivs[doOnes2]**3.7
        doOnes3 = doOnes & (lambda_obs_array>=5.7*lj)
        tauLAFLS[doOnes3,j] = A_jLAF3[j]*ldivs[doOnes3]**5.5
        doOnes4 = doOnes & (lambda_obs_array<3.*lj)
        tauDLALS[doOnes4,j] = A_jDLA1[j]*ldivs[doOnes4]**2.
        doOnes5 = doOnes & (lambda_obs_array>=3.*lj)
        tauDLALS[doOnes5,j] = A_jDLA2[j]*ldivs[doOnes5]**3.

    #now on to the lyman continuum (LC)
    tauLAFLC = np.zeros(lolen); tauDLALC = np.zeros(lolen)

    ldiv2s=lambda_obs_array/911.8
    doOnes = ldiv2s<2.2
    tauLAFLC[doOnes] = 5.22e-4*(1.+z)**3.4*ldiv2s[doOnes]**2.1 + 0.325*ldiv2s[doOnes]**1.2 - 3.14e-2*ldiv2s[doOnes]**2.1
    doOnes2 = (ldiv2s>=2.2) & (ldiv2s<5.7)
    tauLAFLC[doOnes2] = 5.22e-4*(1.+z)**3.4*ldiv2s[doOnes2]**2.1 + 0.218*ldiv2s[doOnes2]**2.1 - 2.55e-2*ldiv2s[doOnes2]**3.7
    doOnes3 = (ldiv2s>=5.7) & (ldiv2s<1.+z)
    tauLAFLC[doOnes3] = 5.22e-4*((1.+z)**3.4*ldiv2s[doOnes3]**2.1 - ldiv2s[doOnes3]**5.5)

    doOnes4 = ldiv2s<3.
    tauDLALC[doOnes4] = 0.634+4.7e-2*(1.+z)**3. - 1.78e-2*(1.+z)**3.3*ldiv2s[doOnes4]**-0.3 - 0.135*ldiv2s[doOnes4]**2. - 0.291*ldiv2s[doOnes4]**-0.3
    doOnes5 = (ldiv2s>=3.) & (ldiv2s<1.+z)
    tauDLALC[doOnes5] = 4.7e-2*(1.+z)**3. - 1.78e-2*(1.+z)**3.3*ldiv2s[doOnes5]**-0.3 - 2.92e-2*ldiv2s[doOnes5]**3.

    tautotal = np.sum(tauLAFLS,axis=1) + np.sum(tauDLALS,axis=1) + tauLAFLC + tauDLALC
    return np.exp(-1.*tautotal)













def get_inoue14_table2(return_j_keys=True):
    """
    """
    table2_j_keys = {
        2: {'wavelength': 1215.67, 'A_LAF1': 1.690e-2, 'A_LAF2': 2.354e-3,
            'A_LAF3': 1.026e-4, 'A_DLA1': 1.617e-4, 'A_DLA2': 5.390e-5},
        3: {'wavelength': 1025.72, 'A_LAF1': 4.692e-3, 'A_LAF2': 6.536e-4,
            'A_LAF3': 2.849e-5, 'A_DLA1': 1.545e-4, 'A_DLA2': 5.151e-5},
        4: {'wavelength': 972.537, 'A_LAF1': 2.239e-3, 'A_LAF2': 3.119e-4,
            'A_LAF3': 1.360e-5, 'A_DLA1': 1.498e-4, 'A_DLA2': 4.992e-5},
        5: {'wavelength': 949.743, 'A_LAF1': 1.319e-3, 'A_LAF2': 1.837e-4,
            'A_LAF3': 8.010e-6, 'A_DLA1': 1.460e-4, 'A_DLA2': 4.868e-5},
        6: {'wavelength': 937.803, 'A_LAF1': 8.707e-4, 'A_LAF2': 1.213e-4,
            'A_LAF3': 5.287e-6, 'A_DLA1': 1.429e-4, 'A_DLA2': 4.763e-5},
        7: {'wavelength': 930.748, 'A_LAF1': 6.178e-4, 'A_LAF2': 8.606e-5,
            'A_LAF3': 3.752e-6, 'A_DLA1': 1.402e-4, 'A_DLA2': 4.672e-5},
        8: {'wavelength': 926.226, 'A_LAF1': 4.609e-4, 'A_LAF2': 6.421e-5,
            'A_LAF3': 2.799e-6, 'A_DLA1': 1.377e-4, 'A_DLA2': 4.590e-5},
        9: {'wavelength': 923.150, 'A_LAF1': 3.569e-4, 'A_LAF2': 4.971e-5,
            'A_LAF3': 2.167e-6, 'A_DLA1': 1.355e-4, 'A_DLA2': 4.516e-5},
        10: {'wavelength': 920.963, 'A_LAF1': 2.843e-4, 'A_LAF2': 3.960e-5,
             'A_LAF3': 1.726e-6, 'A_DLA1': 1.335e-4, 'A_DLA2': 4.448e-5},
        11: {'wavelength': 919.352, 'A_LAF1': 2.318e-4, 'A_LAF2': 3.229e-5,
             'A_LAF3': 1.407e-6, 'A_DLA1': 1.316e-4, 'A_DLA2': 4.385e-5},
        12: {'wavelength': 918.129, 'A_LAF1': 1.923e-4, 'A_LAF2': 2.679e-5,
             'A_LAF3': 1.168e-6, 'A_DLA1': 1.298e-4, 'A_DLA2': 4.326e-5},
        13: {'wavelength': 917.181, 'A_LAF1': 1.622e-4, 'A_LAF2': 2.259e-5,
             'A_LAF3': 9.847e-7, 'A_DLA1': 1.281e-4, 'A_DLA2': 4.271e-5},
        14: {'wavelength': 916.429, 'A_LAF1': 1.385e-4, 'A_LAF2': 1.929e-5,
             'A_LAF3': 8.410e-7, 'A_DLA1': 1.265e-4, 'A_DLA2': 4.218e-5},
        15: {'wavelength': 915.824, 'A_LAF1': 1.196e-4, 'A_LAF2': 1.666e-5,
             'A_LAF3': 7.263e-7, 'A_DLA1': 1.250e-4, 'A_DLA2': 4.168e-5},
        16: {'wavelength': 915.329, 'A_LAF1': 1.043e-4, 'A_LAF2': 1.453e-5,
             'A_LAF3': 6.334e-7, 'A_DLA1': 1.236e-4, 'A_DLA2': 4.120e-5},
        17: {'wavelength': 914.919, 'A_LAF1': 9.174e-5, 'A_LAF2': 1.278e-5,
             'A_LAF3': 5.571e-7, 'A_DLA1': 1.222e-4, 'A_DLA2': 4.075e-5},
        18: {'wavelength': 914.576, 'A_LAF1': 8.128e-5, 'A_LAF2': 1.132e-5,
             'A_LAF3': 4.936e-7, 'A_DLA1': 1.209e-4, 'A_DLA2': 4.031e-5},
        19: {'wavelength': 914.286, 'A_LAF1': 7.251e-5, 'A_LAF2': 1.010e-5,
             'A_LAF3': 4.403e-7, 'A_DLA1': 1.197e-4, 'A_DLA2': 3.989e-5},
        20: {'wavelength': 914.039, 'A_LAF1': 6.505e-5, 'A_LAF2': 9.062e-6,
             'A_LAF3': 3.950e-7, 'A_DLA1': 1.185e-4, 'A_DLA2': 3.949e-5},
        21: {'wavelength': 913.826, 'A_LAF1': 5.868e-5, 'A_LAF2': 8.174e-6,
             'A_LAF3': 3.563e-7, 'A_DLA1': 1.173e-4, 'A_DLA2': 3.910e-5},
        22: {'wavelength': 913.641, 'A_LAF1': 5.319e-5, 'A_LAF2': 7.409e-6,
             'A_LAF3': 3.230e-7, 'A_DLA1': 1.162e-4, 'A_DLA2': 3.872e-5},
        23: {'wavelength': 913.480, 'A_LAF1': 4.843e-5, 'A_LAF2': 6.746e-6,
             'A_LAF3': 2.941e-7, 'A_DLA1': 1.151e-4, 'A_DLA2': 3.836e-5},
        24: {'wavelength': 913.339, 'A_LAF1': 4.427e-5, 'A_LAF2': 6.167e-6,
             'A_LAF3': 2.689e-7, 'A_DLA1': 1.140e-4, 'A_DLA2': 3.800e-5},
        25: {'wavelength': 913.215, 'A_LAF1': 4.063e-5, 'A_LAF2': 5.660e-6,
             'A_LAF3': 2.467e-7, 'A_DLA1': 1.130e-4, 'A_DLA2': 3.766e-5},
        26: {'wavelength': 913.104, 'A_LAF1': 3.738e-5, 'A_LAF2': 5.207e-6,
             'A_LAF3': 2.270e-7, 'A_DLA1': 1.120e-4, 'A_DLA2': 3.732e-5},
        27: {'wavelength': 913.006, 'A_LAF1': 3.454e-5, 'A_LAF2': 4.811e-6,
             'A_LAF3': 2.097e-7, 'A_DLA1': 1.110e-4, 'A_DLA2': 3.700e-5},
        28: {'wavelength': 912.918, 'A_LAF1': 3.199e-5, 'A_LAF2': 4.456e-6,
             'A_LAF3': 1.943e-7, 'A_DLA1': 1.101e-4, 'A_DLA2': 3.668e-5},
        29: {'wavelength': 912.839, 'A_LAF1': 2.971e-5, 'A_LAF2': 4.139e-6,
             'A_LAF3': 1.804e-7, 'A_DLA1': 1.091e-4, 'A_DLA2': 3.637e-5},
        30: {'wavelength': 912.768, 'A_LAF1': 2.766e-5, 'A_LAF2': 3.853e-6,
             'A_LAF3': 1.680e-7, 'A_DLA1': 1.082e-4, 'A_DLA2': 3.607e-5},
        31: {'wavelength': 912.703, 'A_LAF1': 2.582e-5, 'A_LAF2': 3.596e-6,
             'A_LAF3': 1.568e-7, 'A_DLA1': 1.073e-4, 'A_DLA2': 3.578e-5},
        32: {'wavelength': 912.645, 'A_LAF1': 2.415e-5, 'A_LAF2': 3.364e-6,
             'A_LAF3': 1.466e-7, 'A_DLA1': 1.065e-4, 'A_DLA2': 3.549e-5},
        33: {'wavelength': 912.592, 'A_LAF1': 2.263e-5, 'A_LAF2': 3.153e-6,
             'A_LAF3': 1.375e-7, 'A_DLA1': 1.056e-4, 'A_DLA2': 3.521e-5},
        34: {'wavelength': 912.543, 'A_LAF1': 2.126e-5, 'A_LAF2': 2.961e-6,
             'A_LAF3': 1.291e-7, 'A_DLA1': 1.048e-4, 'A_DLA2': 3.493e-5},
        35: {'wavelength': 912.499, 'A_LAF1': 2.000e-5, 'A_LAF2': 2.785e-6,
             'A_LAF3': 1.214e-7, 'A_DLA1': 1.040e-4, 'A_DLA2': 3.466e-5},
        36: {'wavelength': 912.458, 'A_LAF1': 1.885e-5, 'A_LAF2': 2.625e-6,
             'A_LAF3': 1.145e-7, 'A_DLA1': 1.032e-4, 'A_DLA2': 3.440e-5},
        37: {'wavelength': 912.420, 'A_LAF1': 1.779e-5, 'A_LAF2': 2.479e-6,
             'A_LAF3': 1.080e-7, 'A_DLA1': 1.024e-4, 'A_DLA2': 3.414e-5},
        38: {'wavelength': 912.385, 'A_LAF1': 1.682e-5, 'A_LAF2': 2.343e-6,
             'A_LAF3': 1.022e-7, 'A_DLA1': 1.017e-4, 'A_DLA2': 3.389e-5},
        39: {'wavelength': 912.353, 'A_LAF1': 1.593e-5, 'A_LAF2': 2.219e-6,
             'A_LAF3': 9.673e-8, 'A_DLA1': 1.009e-4, 'A_DLA2': 3.364e-5},
        40: {'wavelength': 912.324, 'A_LAF1': 1.510e-5, 'A_LAF2': 2.103e-6,
             'A_LAF3': 9.169e-8, 'A_DLA1': 1.002e-4, 'A_DLA2': 3.339e-5},
    }

    table2_param_keys = {
        'wavelengths': np.array([table2_j_keys[j]['wavelength'] for
                                j in table2_j_keys.keys()]),
        'A_LAF1': np.array([table2_j_keys[j]['A_LAF1'] for
                           j in table2_j_keys.keys()]),
        'A_LAF2': np.array([table2_j_keys[j]['A_LAF2'] for
                           j in table2_j_keys.keys()]),
        'A_LAF3': np.array([table2_j_keys[j]['A_LAF3'] for
                           j in table2_j_keys.keys()]),
        'A_DLA1': np.array([table2_j_keys[j]['A_DLA1'] for
                           j in table2_j_keys.keys()]),
        'A_DLA2': np.array([table2_j_keys[j]['A_DLA2'] for
                           j in table2_j_keys.keys()]),
    }

    if return_j_keys:
        table2 = table2_j_keys
    else:
        table2 = table2_param_keys

    return table2

def calculate_equation21(observed, rest, j, table2):
    """
    """
    tau = np.zeros(observed.shape)

    mask1 = observed < 2.2*rest
    mask2 = (2.2*rest <= observed) & (observed < 5.7*rest)
    mask3 = observed >= 5.7*rest
    tau[mask1] = table2[j]['A_LAF1'] * (observed[mask1] / rest)**1.2
    tau[mask2] = table2[j]['A_LAF2'] * (observed[mask2] / rest)**3.7
    tau[mask3] = table2[j]['A_LAF3'] * (observed[mask3] / rest)**5.5

    return tau

def calculate_equation22(observed, rest, j, table2):
    """
    """
    tau = np.zeros(observed.shape)

    mask1 = observed < 3.0*rest
    mask2 = observed >= 3.0*rest
    tau[mask1] = table2[j]['A_DLA1'] * (observed[mask1] / rest)**2.0
    tau[mask2] = table2[j]['A_DLA2'] * (observed[mask2] / rest)**3.0

    return tau


def calculate_equation25(observed, z):
    """
    """
    tau = np.zeros(observed.shape)
    ll = 911.8 * u.AA  # Lyman limit

    mask1 = observed < ll*(1+z)
    mask2 = observed >= ll*(1+z)
    tau[mask1] = (0.325 * (observed[mask1] / ll)**1.2 -
                  0.325 * (1 + z)**-0.9 * (observed[mask1] / ll)**2.1)
    tau[mask2] = 0

    return tau


def calculate_equation26(observed, z):
    """
    """
    tau = np.zeros(observed.shape)
    ll = 911.8 * u.AA  # Lyman limit

    mask1 = observed < 2.2*ll
    mask2 = (2.2*ll <= observed) & (observed < ll*(1+z))
    mask3 = observed >= ll*(1+z)
    tau[mask1] = (2.55e-2 * (1 + z)**1.6 * (observed[mask1] / ll)**2.1 +
                  0.325 * (observed[mask1] / ll)**1.2 -
                  0.25 * (observed[mask1] / ll)**2.1)
    tau[mask2] = (2.55e-2 * (1 + z)**1.6 * (observed[mask2] / ll)**2.1 -
                  2.55e-2 * (observed[mask2] / ll)**3.7)
    tau[mask3] = 0

    return tau


def calculate_equation27(observed, z):
    """
    """
    tau = np.zeros(observed.shape)
    ll = 911.8 * u.AA  # Lyman limit

    mask1 = observed < 2.2*ll
    mask2 = (2.2*ll <= observed) & (observed < 5.7*ll)
    mask3 = (5.7*ll <= observed) & (observed < ll*(1+z))
    mask4 = observed >= ll*(1+z)
    tau[mask1] = (5.22e-4 * (1 + z)**3.4 * (observed[mask1] / ll)**2.1 +
                  0.325 * (observed[mask1] / ll)**1.2 -
                  3.14e-2 * (observed[mask1] / ll)**2.1)
    tau[mask2] = (5.22e-4 * (1 + z)**3.4 * (observed[mask2] / ll)**2.1 +
                  0.218 * (observed[mask2] / ll)**2.1 -
                  2.55e-2 * (observed[mask2] / ll)**3.7)
    tau[mask3] = (5.22e-4 * (1 + z)**3.4 * (observed[mask3] / ll)**2.1 -
                  5.22e-4 * (observed[mask3] / ll)**5.5)
    tau[mask4] = 0

    return tau


def calculate_equation28(observed, z):
    """
    """
    tau = np.zeros(observed.shape)
    ll = 911.8 * u.AA  # Lyman limit

    mask1 = observed < ll*(1+z)
    mask2 = observed >= ll*(1+z)
    tau[mask1] = (0.211 * (1 + z)**2.0 -
                  7.66e-2 * (1 + z)**2.3 * (observed[mask1] / ll)**-0.3 -
                  0.135 * (observed[mask1] / ll)**2.0)
    tau[mask2] = 0

    return tau


def calculate_equation29(observed, z):
    """
    """
    tau = np.zeros(observed.shape)
    ll = 911.8 * u.AA  # Lyman limit

    mask1 = observed < 3.0*ll
    mask2 = (3.0*ll <= observed) & (observed < ll*(1+z))
    mask3 = observed >= ll*(1+z)
    tau[mask1] = (0.634 + 4.7e-2 * (1 + z)**3.0 -
                  1.78e-2 * (1 + z)**3.3 * (observed[mask1] / ll)**-0.3 -
                  0.135 * (observed[mask1] / ll)**2.0 -
                  0.291 * (observed[mask1] / ll)**-0.3)
    tau[mask2] = (4.7e-2 * (1 + z)**3.0 -
                  1.78e-2 * (1 + z)**3.3 * (observed[mask2] / ll)**-0.3 -
                  2.92e-2 * (observed[mask2] / ll)**3.0)
    tau[mask3] = 0

    return tau

def inoue14_tau_LS(observed, z):
    """
    Equations (21) and (22)

    observed wavelength in angstroms
    """
    # Get all wavelengths and Table 2 from Inoue+2014
    rest_frame = get_inoue14_table2(return_j_keys=False)['wavelengths'] * u.AA
    redshifted = (1 + z) * rest_frame

    table2 = get_inoue14_table2(return_j_keys=True)


    LAF_tau = np.zeros((len(observed), len(rest_frame)))
    DLA_tau = np.zeros((len(observed), len(rest_frame)))
    for i, (rest, shifted) in enumerate(zip(rest_frame, redshifted)):
        j = i + 2  # Since the Lyman series starts at 2
        mask = (rest < observed) & (observed < shifted)
        LAF_tau[mask, i] = calculate_equation21(observed[mask], rest, j, table2)
        DLA_tau[mask, i] = calculate_equation22(observed[mask], rest, j, table2)
    LAF_tau, DLA_tau = np.sum(LAF_tau, axis=1), np.sum(DLA_tau, axis=1)

    return LAF_tau, DLA_tau


def inoue14_tau_LC(observed, z):
    """
    Equations (25)-(29)

    observed wavelength in angstroms
    """
    ll = 911.8 * u.AA
    mask = observed > ll
    LAF_tau = np.zeros((len(observed)))
    DLA_tau = np.zeros((len(observed)))

    # LAF component
    if z < 1.2:
        LAF_tau[mask] = calculate_equation25(observed[mask], z)
    elif 1.2 <= z < 4.7:
        LAF_tau[mask] = calculate_equation26(observed[mask], z)
    else:
        LAF_tau[mask] = calculate_equation27(observed[mask], z)

    # DLA component
    if z < 2.0:
        DLA_tau[mask] = calculate_equation28(observed[mask], z)
    else:
        DLA_tau[mask] = calculate_equation29(observed[mask], z)

    return LAF_tau, DLA_tau

def calc_transmission(observed, z, plot=False):
    """
    """
    tau_LS_LAF, tau_LS_DLA = inoue14_tau_LS(observed, z=z)
    tau_LC_LAF, tau_LC_DLA = inoue14_tau_LC(observed, z=z)
    tau = tau_LS_LAF + tau_LS_DLA + tau_LC_LAF + tau_LC_DLA
    transmission = np.exp(-tau)

    return transmission
