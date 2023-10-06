import numpy as np
from scipy.stats import norm


# randomly sample Muv in a way that has fewer very bright objects
def sample_muv(min_muv, max_muv, loc, scale, Nsamples):
    muvs = []
    while len(muvs) < Nsamples:
        x = np.random.uniform(min_muv, max_muv)
        y = np.random.uniform()
        if y < norm.cdf(x, loc=loc, scale=scale):
            muvs.append(x)
    return muvs




# size-luminosity relationship from Shibuya+2015
def shibuya_rcirc(Muv, z):
    LuvDivL0 = 10**(-0.4*(21.+np.array(Muv)))
    reff_circ = 6.9*(1.+np.array(z))**-1.2 * LuvDivL0**0.27

    return reff_circ



# UV-slope - Muv relationship from Topping+2023
def uv_slope(zarr, muvarr, band):
    if hasattr(zarr, "__len__"):
        betas = []
        for z, muv in zip(zarr, muvarr):
#            if 7 < z < 12:
#                betas.append((muv+19)*-.06 +-2.33)
#            elif 10 < z < 16:
#                betas.append((muv+19)*-.06 +-2.42)
#            else:
#                betas.append(-2)
            if band == 'F115W':
                betas.append((muv+19)*-.06 +-2.33)
            elif band == 'F150W':
                betas.append((muv+19)*-.06 +-2.42)
            else:
                betas.append(-2)

        return betas
                
    else:
        return -2
