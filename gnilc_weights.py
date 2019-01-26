import numpy as np
import healpy as hp
from astropy.io import fits as pyfits
import subprocess
import ConfigParser
import os

import gnilc_auxiliary
import misc_functions

import pdb
import time

############################################################################
############################################################################

start_time = time.time()

##### Read parameters.ini

Config = ConfigParser.ConfigParser()
initial_file = "parameters_weights.ini"

##### Read Inputs

ensemble_maps = misc_functions.ConfigSectionMap(Config, initial_file, "General")['ensemble_maps']
bandcenters = misc_functions.ConfigSectionMap(Config, initial_file, "General")['bandcenters']
output_suffix = misc_functions.ConfigSectionMap(Config, initial_file, "General")['output_suffix']

bandcenters_list = misc_functions.getlist(bandcenters)
n_bandcenters = len(bandcenters_list)
bandcenters = np.zeros(n_bandcenters, dtype=np.int32)
for i in range(0, n_bandcenters):
    bandcenters[i] = long(bandcenters_list[i])

### Create output directory

path_in = os.path.realpath(__file__)
directory_in = os.path.dirname(path_in)
directory_out = directory_in + '/output_weights'

if not os.path.exists(directory_out):
    os.makedirs(directory_out)

##### Maps: info

maps = pyfits.getdata('input/' + ensemble_maps + '.fits')  # read input maps

nf = maps[:, 0].size
nside = hp.pixelfunc.npix2nside((maps[0, :]).size)
lmax = 3 * nside - 1

##### Needlet band-pass windows

print "Creating wavelet bands."

# Cosine bands -- later add Gaussian as well
    
bands = gnilc_auxiliary.cosine_ell_bands(bandcenters)

nbands = (bands[0, :]).size   # number of bands
lmax_bands = np.zeros(nbands, dtype=np.int32)   # bands effective ell max 

for j in range(0, nbands):
    lmax_bands[j] = max(np.where(bands[:, j] != 0.0)[0])

############################################################################
############################################################################

##### Channel maps: SHT and wavelet transform

print "Applying wavelets to the observed maps."

relevant_band_max = np.zeros(nf, dtype=np.int32)

for i in range(0, nf):
    alm_map = hp.sphtfunc.map2alm(maps[i, :], lmax=lmax)

    # Wavelet transform channel maps: band-pass filtering in (l,m)-space and transform back to real (pixel) space

    # relevant bands for each channel map
    if lmax <= max(lmax_bands):
        relevant_band_max[i] = min(np.where(lmax_bands[:] >= lmax)[0])
    else:
        relevant_band_max[i] = nbands - 1
        
    if lmax_bands[relevant_band_max[i]] == max(lmax_bands):
        relevant_band_max[i] = nbands - 1
            
    gnilc_auxiliary.alm2wavelets(alm_map, bands[:, 0:relevant_band_max[i] + 1], nside, 'wavelet_ensemble_' + str(i).strip() + '.fits', nside)

maps = 0
alm_map = 0

############################################################################    ############################################################################

##### Apply GNILC weights to the ensemble wavelet maps

print "Applying the GNILC weights to the ensemble wavelet maps."

for i in range(0, nf):
    pyfits.append('wavelet_gnilc_ensemble_' + str(i).strip() + '.fits', bands[:, 0:relevant_band_max[i] + 1])

for j in range(0, nbands):    # loop over needlet bands
    w_target = pyfits.getdata('output/' + 'ilc_weights_' + output_suffix + "_" + str(j).strip() + '.fits')
          
    for i in range(0, nf):   # loop over frequency channels
        needlet_ilc_r = 0.
        for k in range(0, nf):   # loop over frequency channels
            tot_needlet = pyfits.getdata('wavelet_ensemble_' + str(k).strip() + '.fits', j + 1)
            w_map_r = w_target[:, i, k]

            # apply the ILC weight matrix to the channel wavelet maps

            needlet_ilc_r = needlet_ilc_r + w_map_r * tot_needlet   # ILC filtering

        pyfits.append('wavelet_gnilc_ensemble_' + str(i).strip() + '.fits', needlet_ilc_r)

w_target = 0; w_map_r = 0

############################################################################    ############################################################################

##### Synthesize GNILC wavelet maps to GNILC maps

ilc_map = np.zeros((nf, hp.pixelfunc.nside2npix(nside)))

for i in range(0, nf):
   ilc_map[i, :] = gnilc_auxiliary.wavelets2map('wavelet_gnilc_ensemble_' + str(i).strip() + '.fits', nside)

############################################################################    ############################################################################

##### Produce GNILC maps

print "Producing the GNILC maps."

maps_out = np.zeros((nf, hp.pixelfunc.nside2npix(nside)))

for i in range(0, nf):
    # GNILC maps (fits file)
    maps_out[i, :] = ilc_map[i, :]# * galmask

pyfits.writeto('output_weights/ensemble_gnilc_maps_' + output_suffix + '.fits', maps_out, overwrite = True)

##### Clean and save things 

for i in range(0, nf):
    file = 'wavelet_ensemble_' + str(i).strip() + '.fits'
    subprocess.call(["rm", file])
    file = 'wavelet_gnilc_ensemble_' + str(i).strip() + '.fits'
    subprocess.call(["rm", file])

print time.time() - start_time
