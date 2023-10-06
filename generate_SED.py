import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from astropy import units as u
from igm_attenuation import *
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)


filters_bounds = {
        'F070W':[0.624, 0.781],
        'F090W':[0.795, 1.005],
        'F115W':[1.013, 1.282],
        'F150W':[1.331, 1.668],
        'F200W':[1.755, 2.227],
        'F277W':[2.423, 3.132],
        'F335M':[3.177, 3.537],
        'F356W':[3.135, 3.981],
        'F410M':[3.866, 4.302],
        'F444W':[3.881, 4.982]}


def normalize_flux_in_filter(z, Muv, beta, filts):
    obs_lam = np.linspace(6000, 53000, 1000)
    # Cosmology to use
    DL = cosmo.luminosity_distance(z)*1e6

    # calculate the observed UV magnitude
    muv_obs = Muv+5*(np.log10(DL.value)-1) - 2.5*np.log10(1+z)

#    # IGM transmission = exp(-tau), attenuates the redshifted spectrum
#    igm_transmission = (calc_transmission(obs_lam*u.AA, z=z) *
#                        u.dimensionless_unscaled)
    igm_transmission = getT(obs_lam,z)

    # just make top hat filter curves for now
    # can fold in actual transmission later
    filter_curves = {}
    for filt in filts:
        new_curve = np.zeros_like(obs_lam)
        new_curve[(filters_bounds[filt][1]*1e4 > obs_lam) * (filters_bounds[filt][0]*1e4 < obs_lam)] = 1
        filter_curves[filt] = np.array(new_curve)

    # create power-law spectrum
    # add in the IGM attenuation
    # normalize it at 1500AA
    unscaled_fnu = (obs_lam**(beta+2)) * igm_transmission
    ind = np.abs(obs_lam-(1500*(1+z))).argmin()
    unscaled_fnu /= unscaled_fnu[ind]

    # convert to [cgs] and normalize the spectrum
    w = 1500*(1+z)*u.AA
    ab_const = 48.59993437771777  # -2.5 * log(3.631e-20 erg/s/cm^2/Hz)
    normflux = 10**(-0.4*(muv_obs+ab_const))
    obs_fnu = unscaled_fnu * normflux

#    plt.plot(obs_lam, obs_fnu*1e23*1e9, color='black', linewidth=2)
#    plt.xlabel('observed wavelength (AA)')
#    plt.ylabel('flux (njy)')


    # Calculate photometry
    fluxes = {}
    for filt in filter_curves:
        # Integrate lambda * throughputs to normalize and then calculate a
        # weighted average
        int_throughputs = trapz(filter_curves[filt]/obs_lam,
                                      x=obs_lam)

        true_flux = trapz(obs_fnu/obs_lam * filter_curves[filt],
                           x=obs_lam)/int_throughputs
        
#        pivot_sqrd = (trapz(filter_curves[filt] * obs_lam, x=obs_lam) /
#                            trapz(filter_curves[filt] / obs_lam, x=obs_lam))
#        pivot = np.sqrt(pivot_sqrd)
#        plt.scatter(pivot, true_flux*1e9*1e23, color='grey', s=62, zorder=9, linewidth=1, edgecolor='black')
        fluxes[filt] = true_flux*1e9*1e23
    return fluxes


# DEBUG
if __name__ == "__main__":
    z = 9
    Muv = -20
    beta = -2.5
    obs_lam = np.linspace(6000, 53000, 1000)
    filts = ['F090W', 'F150W', 'F277W']
    normalize_flux_in_filter(z, Muv, beta, obs_lam, filts)
























