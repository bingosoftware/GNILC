import numpy as np
import healpy as hp
from astropy.io import fits as pyfits
import subprocess
import ConfigParser
import os

import gnilc_auxiliary
import misc_functions

import time

############################################################################
############################################################################

start_time = time.time()

##### Read parameters.ini

Config = ConfigParser.ConfigParser()
initial_file = "parameters.ini"

##### Read Inputs

input_maps = misc_functions.ConfigSectionMap(Config, initial_file, "General")['input_maps']
prior_maps = misc_functions.ConfigSectionMap(Config, initial_file, "General")['prior_maps']
mask = misc_functions.ConfigSectionMap(Config, initial_file, "General")['mask']
needlet_bands = misc_functions.ConfigSectionMap(Config, initial_file, "General")['needlet_bands']
bandcenters = misc_functions.ConfigSectionMap(Config, initial_file, "General")['bandcenters']
ilc_bias = misc_functions.ConfigSectionMap(Config, initial_file, "General")['ilc_bias']
output_suffix = misc_functions.ConfigSectionMap(Config, initial_file, "General")['output_suffix']

save_ilc_weights = misc_functions.ConfigGetBoolean(Config, initial_file, "General", 'save_ilc_weights')
save_wavelets_dimensions = misc_functions.ConfigGetBoolean(Config, initial_file, "General", 'save_wavelets_dimensions')

bandcenters_list = misc_functions.getlist(bandcenters)
n_bandcenters = len(bandcenters_list)
bandcenters = np.zeros(n_bandcenters, dtype=np.int32)
for i in range(0, n_bandcenters):
    bandcenters[i] = long(bandcenters_list[i])

ilc_bias = float(ilc_bias)

### Create output directory

path_in = os.path.realpath(__file__)
directory_in = os.path.dirname(path_in)
directory_out = directory_in + '/output'

if not os.path.exists(directory_out):
    os.makedirs(directory_out)

##### Maps: info

maps = pyfits.getdata('input/' + input_maps + '.fits')  # read input maps
maps_target = pyfits.getdata('input/' + prior_maps +'.fits')   # read prior map

nf = maps[:, 0].size
nside = hp.pixelfunc.npix2nside((maps[0, :]).size)
lmax = 3 * nside - 1

##### Needlet band-pass windows

print "Creating wavelet bands."

# Cosine bands -- later add Gaussian as well

if needlet_bands == 'Cosine':    
    bands = gnilc_auxiliary.cosine_ell_bands(bandcenters)

    nbands = (bands[0, :]).size   # number of bands
    lmax_bands = np.zeros(nbands, dtype=np.int32)   # bands effective ell max

    for j in range(0, nbands):
        lmax_bands[j] = max(np.where(bands[:, j] != 0.0)[0])

# Gaussian bands

if needlet_bands == 'Gaussian':
    bands = gnilc_auxiliary.gaussbeam_ell_bands(bandcenters, lmax)

    nbands = (bands[0, :]).size   # number of bands
    lmax_bands = np.zeros(nbands, dtype=np.int32)   # bands effective ell max

    for j in range(0, nbands):
        lmax_bands[j] = max(np.where(bands[:, j] != 0.)[0])

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
            
    gnilc_auxiliary.alm2wavelets(alm_map, bands[:, 0:relevant_band_max[i] + 1], nside, 'wavelet_' + str(i).strip() + '.fits', nside)

maps = 0
alm_map = 0

############################################################################    ############################################################################

##### Prior inputs: SHT and wavelet transform    

print "Applying wavelets to the prior maps."

for i in range(0, nf):
    alm_map = hp.sphtfunc.map2alm(maps_target[i, :], lmax=lmax)

    # Wavelet transform channel maps: band-pass filtering in (l,m)-space and transform back to real (pixel) space

    gnilc_auxiliary.alm2wavelets(alm_map, bands[:, 0:relevant_band_max[i] + 1], nside, 'wavelet_target_' + str(i).strip() + '.fits', nside)

maps_target = 0
alm_map = 0

############################################################################    ############################################################################

##### Compute GNILC weights in each needlet band

print "Computing the GNILC weights in each wavelet band."

galmask = pyfits.getdata('input/' + mask + '.fits')   # read Galactic mask

nmodes_band = np.zeros(nbands)   # number of modes for bands
np_band = np.zeros(nbands)   # npix of each wavelets map
nside_band = np.zeros(nbands)   # nside of each wavelets map
pps = np.zeros(nbands)   # pixel area per needlet scale

# Calculate the statistics locally both in space and angular scale (Olivari el al, 2015)

wavelets_dimensions = np.zeros((nbands, hp.pixelfunc.nside2npix(nside)))

for j in range(0, nbands):   # loop on needlet bands
    
    for l in range(0, lmax_bands[nbands - 1]):
        nmodes_band[j] = nmodes_band[j] + (2.0 * l + 1.0) * bands[l, j]**2

    tot_map1_band = pyfits.getdata('wavelet_' + str(nf - 1).strip() + '.fits', j + 1)
    np_band[j] = tot_map1_band.size
    nside_band[j] = hp.pixelfunc.npix2nside(np_band[j])

    tot_map1_band = 0

    # pps: pixel area per needlet scale for computing local covariance matrices: it is controled to be large enough so that the ILC bias is not larger than ILC bias upper limit fixed by  the user (Delabrouille et al., 2009).

    pps[j] = np.sqrt(np_band[j] * (nf - 1) / (ilc_bias * nmodes_band[j]))
 
    # mask at the correct wavelets maps nside
    
    mask_wav = hp.pixelfunc.ud_grade(galmask, nside_band[j].astype(int), order_in='RING', order_out='RING')  # mask at the correct wavelets maps nside

    ind_pix_relevant = np.where(mask_wav != 0)
    #n_ind_pix_relevant = ind_pix_relevant.size	

    Rq = np.zeros((np_band[j].astype(int), nf, nf))   #total local covariance
    Rq_nuisance = np.zeros((np_band[j].astype(int), nf, nf))   # nuisance local covariance: "nuisance" stands for target signal

    # Compute local covariance matrices for each wavelet j: R_{nu x nu'}(p) (total covariance) and Rn_{nu x nu'}(p) (nuisance covariance)
     
    for i in range(0, nf):
	tot_map1_band = pyfits.getdata('wavelet_' + str(i).strip() + '.fits', j + 1) #* mask_wav
	target_map1_band = pyfits.getdata('wavelet_target_' + str(i).strip() + '.fits', j + 1) #* mask_wav

        for k in range(0, i + 1):
	    tot_map2_band = pyfits.getdata('wavelet_' + str(k).strip() + '.fits', j + 1) #* mask_wav
            target_map2_band = pyfits.getdata('wavelet_target_' + str(k).strip() + '.fits', j + 1) #* mask_wav

            Rq[:, i, k] = gnilc_auxiliary.localcovar(tot_map1_band, tot_map2_band, pps[j])   # total local covariance
            Rq[:, k, i] = Rq[:, i, k]
            Rq_nuisance[:, i, k] = gnilc_auxiliary.localcovar(target_map1_band, target_map2_band, pps[j])   # nuisance local covariance
            Rq_nuisance[:, k, i] = Rq_nuisance[:, i, k]

            tot_map2_band = 0; target_map2_band = 0

        tot_map1_band = 0; target_map1_band = 0

    w_target = np.zeros((np_band[j].astype(int), nf, nf))   # ILC weight

    # Eigenvalue decomposition of the whitened total covariance matrix, PCA-like analysis, and statistical model selection of the optimal rank of the target signal mixing matrix by using the AIC criterion (Olivari et al. 2016).  

    for l in ind_pix_relevant[0]:   # loop on pixels

        # Whitening of the total covariance matrix and eigenvalue decomposition.
      
        inv_sq_root_nuisance = gnilc_auxiliary.whitening_matrix(Rq_nuisance[l, :, :])   # inverse square root nuisance covariance matrix

        white_covar = np.array(np.matrix(inv_sq_root_nuisance) * np.matrix(Rq[l, :, :]) * np.matrix.transpose(np.matrix(inv_sq_root_nuisance)))  # whitened total matrix - eigenvalues ~ 1 corresponds to the HI signal

        evecr, evalr, Vr = np.linalg.svd(white_covar, full_matrices=True)
        evecr_t = np.transpose(evecr)

        # COMPUTE AIC CRITERION    

        AIC = np.zeros(nf)

        fun = evalr - np.log(evalr) - 1.0
        total = np.sum(fun)
        for r in range(1, nf + 1):
            if r < nf:
                total = total - fun[r - 1]
                AIC[r - 1] = 2 * r + total
            else:
                AIC[r - 1] = 2 * r

        n_dim = max(np.where(AIC == np.ndarray.min(AIC))[0]) + 1  # foregrounds plus noise degrees fo freedom

        if np.sum(fun) < np.ndarray.min(AIC):
            n_dim = 0   # allow zero dimension 

	wavelets_dimensions[j, l] = n_dim

        t_evecr = np.transpose(evecr_t)

        if nf - n_dim == 0:
            w_target[l, :, :] = 0.0
        else:
            if n_dim < nf:
                Un = np.zeros((nf, nf - n_dim))
                Un[:, 0:nf - n_dim] = t_evecr[:, n_dim:nf]  # nuisance eigenvectors

                Dn = np.diagflat(evalr[n_dim:nf])   # nuisance eigenvalues

                ortho_r = np.array(np.matrix.transpose(np.linalg.inv(np.matrix((np.sqrt(Dn)))) * np.matrix.transpose(np.matrix(Un)) * np.matrix(t_evecr[:, n_dim:nf])))

                ortho_r = ortho_r / np.linalg.norm(ortho_r, ord=np.inf)

                Ln = np.array(np.linalg.inv(np.matrix(np.sqrt(Dn))) * np.matrix.transpose(np.matrix(Un)) * np.matrix(inv_sq_root_nuisance))
                Ln_minus = np.array(np.linalg.inv(np.matrix(inv_sq_root_nuisance)) * np.matrix(Un) * np.matrix.transpose(np.matrix(np.sqrt(Dn))))

                Pn = np.array(np.matrix(Ln_minus) * np.matrix(Ln))   # Projector on nuisance subspace

                if nf - n_dim == 1:
                    vv = np.sqrt((evalr[n_dim:nf])[0]) * np.array(np.matrix.transpose(np.linalg.inv(np.matrix(inv_sq_root_nuisance)) * np.matrix(t_evecr[:, n_dim:nf])))
                    w_target[l, :, :] = np.array(np.matrix.transpose(np.matrix(vv)) * np.matrix(gnilc_auxiliary.ilc(Rq[l, :, :], vv)))   # 1D ILC    
                else:  
                    w_target[l, :, :] = gnilc_auxiliary.multi_ilc(Ln_minus, Pn, np.transpose(ortho_r), Rq[l, :, :])    # multidimensional ILC

        inv_sq_root_nuisance = 0; white_covar = 0; Un = 0; Dn = 0; ortho_r = 0; Ln = 0; Ln_minus = 0; Pn = 0; vv = 0;

    Rq = 0; Rq_nuisance = 0

    pyfits.append('ilc_weights_' + output_suffix + "_" + str(j).strip() + '.fits', w_target)

    w_target = 0

############################################################################    ############################################################################

##### Apply GNILC weights to wavelet maps

print "Applying the GNILC weights to the observed wavelet maps."

for i in range(0, nf):
    pyfits.append('wavelet_gnilc_target_' + str(i).strip() + '.fits', bands[:, 0:relevant_band_max[i] + 1])

for j in range(0, nbands):    # loop over needlet bands
    w_target = pyfits.getdata('ilc_weights_' + output_suffix + "_" + str(j).strip() + '.fits')
          
    for i in range(0, nf):   # loop over frequency channels
        needlet_ilc_r = 0.
        for k in range(0, nf):   # loop over frequency channels
            tot_needlet = pyfits.getdata('wavelet_' + str(k).strip() + '.fits', j + 1)
            w_map_r = w_target[:, i, k]

            # apply the ILC weight matrix to the channel wavelet maps

            needlet_ilc_r = needlet_ilc_r + w_map_r * tot_needlet   # ILC filtering

        pyfits.append('wavelet_gnilc_target_' + str(i).strip() + '.fits', needlet_ilc_r)

w_target = 0; w_map_r = 0

############################################################################    ############################################################################

##### Synthesize GNILC wavelet maps to GNILC maps

ilc_map = np.zeros((nf, hp.pixelfunc.nside2npix(nside)))

for i in range(0, nf):
   ilc_map[i, :] = gnilc_auxiliary.wavelets2map('wavelet_gnilc_target_' + str(i).strip() + '.fits', nside)

############################################################################    ############################################################################

##### Produce GNILC maps

print "Producing the GNILC maps."

maps_out = np.zeros((nf, hp.pixelfunc.nside2npix(nside)))

for i in range(0, nf):
    # GNILC maps (fits file)
    maps_out[i, :] = ilc_map[i, :]# * galmask

pyfits.writeto('output/reconstructed_maps_' + output_suffix + '.fits', maps_out, overwrite = True)

##### Clean and save things 

if save_wavelets_dimensions:
    pyfits.writeto('output/wavelets_dimensions_' + output_suffix + '.fits', wavelets_dimensions, overwrite = True)

if save_ilc_weights:
    for j in range(0, nbands):
        subprocess.call(["mv " + "ilc_weights_" + output_suffix + "_" + str(j).strip() + ".fits" + " output/"], shell=True)
else:
    for j in range(0, nbands):
	subprocess.call(["rm", "ilc_weights_" + output_suffix + "_" + str(j).strip() + ".fits"])

for i in range(0, nf):
    file = 'wavelet_' + str(i).strip() + '.fits'
    subprocess.call(["rm", file])
    file = 'wavelet_target_' + str(i).strip() + '.fits'
    subprocess.call(["rm", file])
    file = 'wavelet_gnilc_target_' + str(i).strip() + '.fits'
    subprocess.call(["rm", file])

print time.time() - start_time
