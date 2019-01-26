import numpy as np
import healpy as hp

from astropy.io import fits as pyfits

def gaussbeam_ell_bands(beamsizes, lmax):

    """This function generates gaussian-shaped windows in spherical harmonic space, used for wavelet (needlet) analysis of spherical maps.
    """

    sortedbeamsizes = (np.sort(beamsizes))[::-1]

    sortedbeamsizes = sortedbeamsizes * 0.000290888209  # arcmin to rad

    nbands = beamsizes.size + 1
    
    bands = np.zeros((lmax + 1, nbands))

    # First band

    bands[:, 0] = (hp.sphtfunc.gauss_beam(sortedbeamsizes[0], lmax))**2

    # Intermediate bands

    if nbands > 2:
        for i in range(1, nbands - 1):
            bands[:, i] = (hp.sphtfunc.gauss_beam(sortedbeamsizes[i], lmax))**2 - (hp.sphtfunc.gauss_beam(sortedbeamsizes[i - 1], lmax))**2

    # Last band

    bands[:, nbands - 1] = 1.0 - (hp.sphtfunc.gauss_beam(sortedbeamsizes[nbands - 2], lmax))**2

    # Take the square root

    bands = np.sqrt(bands)    

    return bands

def cosine_ell_bands(bandcenters):

    """This function generates cosine-shaped windows in spherical harmonic space, used for wavelet (needlet) analysis of spherical maps.
    """

    lmax = max(bandcenters)
    nbands = bandcenters.size
    bands = np.zeros((lmax + 1, nbands))

    b = bandcenters[0]
    c = bandcenters[1]

    # Left part
    
    ell_left = np.linspace(0, b, b + 1, endpoint=True)
    if b > 0:
        bands[ell_left.astype(int), 0] = 1.0

    # Right part

    ell_right = b + np.linspace(0, c - b, c - b + 1, endpoint=True) 
    bands[ell_right.astype(int), 0] = (np.cos(((ell_right - b) / (c - b)) * (np.pi / 2.0)))

    if nbands >= 3:
        for i in range(1, nbands - 1):
            a = bandcenters[i - 1]
            b = bandcenters[i]
            c = bandcenters[i + 1]
            # Left part
            ell_left = a + np.linspace(0, b - a, b - a + 1, endpoint=True)
            bands[ell_left.astype(int), i] = (np.cos(((b - ell_left) / (b - a)) * (np.pi / 2.0)))
            # Right part
            ell_right = b + np.linspace(0, c - b, c - b + 1, endpoint=True)
            bands[ell_right.astype(int), i] = (np.cos(((ell_right - b) / (c - b)) * (np.pi / 2.0)))

        a = bandcenters[nbands - 2]
        b = bandcenters[nbands - 1]
        # Left part
        ell_left = a + np.linspace(0, b - a, b - a + 1, endpoint=True)
        bands[ell_left.astype(int), nbands - 1] = (np.cos(((b - ell_left) / (b - a)) * (np.pi / 2.0)))

    return bands

def alm2wavelets(alm, bands, nside, waveletsfile, nside_max_w):

    """This function computes wavelets maps for each wavelet band using the input spherical harmonics coefficients."""

    # Get the alm

    nbands = (bands[0, :]).size
    l_max = 3. * nside - 1

    # Write the bands in fits file

    pyfits.append(waveletsfile, bands)

    # Start the band loop 
    
    for i in range(0, nbands):

	# Now filter, restricting the alm and index to the needed ones

        uselmax = max(max(np.where(bands[:, i] != 0)))

        possiblenside = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

        whereok = np.where(possiblenside >= uselmax)

        usenside = min(possiblenside[whereok])

        if usenside > nside_max_w:
            usenside = nside_max_w

        nlm_tot = uselmax * (uselmax + 1.0) / 2.0 + uselmax + 1.0
        
        alm_bands = np.zeros(nlm_tot.astype(int), dtype=np.complex128)
        index_in = np.zeros(nlm_tot.astype(int))
        index_out = np.zeros(nlm_tot.astype(int))

        j = 0
        for l in range(0, uselmax + 1):
            for m in range(0, l + 1):
                index_in[j] = hp.Alm.getidx(l_max, l, m)
                index_out[j] = hp.Alm.getidx(uselmax, l, m)
                j = j + 1

        for k in range(0, nlm_tot.astype(int)):
            alm_bands[index_out[k].astype(int)] = alm[index_in[k].astype(int)]

        alm_write = hp.sphtfunc.almxfl(alm_bands[0:(nlm_tot).astype(int)], bands[0:(uselmax + 1).astype(int), i])
        
        # Transform back to Healpix format

        wavelets_write = hp.sphtfunc.alm2map(alm_write, usenside)
                
        # Write in wavelets fits file
        
        pyfits.append(waveletsfile, wavelets_write)

def wavelets2map(wavelets, nside):

    """This function combines wavelets maps into a single map (procedure reverse of map2wavelets)."""
    
    # Get synthesis windows

    windows = pyfits.getdata(wavelets, 0)
    nw = (windows[0, :]).size

    # Create output alm, alm index and ell values

    lmax = (windows[:, 0]).size - 1
    nalm = hp.sphtfunc.Alm.getsize(lmax)

    alm_out= np.zeros(nalm, dtype=np.complex128)

    for i in range (1, nw + 1):
        map = pyfits.getdata(wavelets, i)
        wh = np.where(windows[:, i - 1] != 0)
        if max(max(wh)) < lmax:
            win_lmax = max(max(wh))
        else:
            win_lmax = lmax

        alm = hp.sphtfunc.map2alm(map, lmax=win_lmax)#, iter=1, use_weights=True)
        alm_win = hp.sphtfunc.almxfl(alm, windows[0:win_lmax + 1, i - 1])

        for l in range(0, win_lmax + 1):
            for m in range(0, l + 1):
                ind_0 = hp.sphtfunc.Alm.getidx(win_lmax, l, m)
                ind_1 = hp.sphtfunc.Alm.getidx(lmax, l, m)
                alm_out[ind_1] = alm_out[ind_1] + alm_win[ind_0]

    # Make map

    map_o = hp.sphtfunc.alm2map(alm_out, nside)

    return map_o

def cross_spectrum(map1, map2):

    """Cross spectrum of two maps."""

    nside = hp.pixelfunc.npix2nside(map1.size)
    lmax = 3 * nside - 1

    alm1 = hp.sphtfunc.map2alm(map1, lmax=lmax)#, iter=1, use_weights=True)

    alm2 = hp.sphtfunc.map2alm(map2, lmax=lmax)#, iter=1, use_weights=True)
    
    cl = hp.sphtfunc.alm2cl(alm1, alm2)

    return cl

def localcovar(map1, map2, pixperscale):

    """Local covariance of two maps in the pixel space."""

    map = map1 * map2
    npix = map.size
    nside = hp.pixelfunc.npix2nside(npix)
    nsidecovar = nside

    # First degrade a bit to speed-up smoothing

    if (nside / 4) > 1:
        nside_out = nside / 4
    else:
        nside_out = 1

    stat = hp.pixelfunc.ud_grade(map, nside_out = nside_out, order_in = 'RING', order_out = 'RING')
    
    # Compute alm

    lmax = 3 * nside_out - 1
    nlm_tot = float(lmax) * float(lmax + 1.0) / 2.0 + lmax + 1.0
    alm = hp.sphtfunc.map2alm(stat, lmax=lmax)#, iter=1, use_weights=True)

    # Find smoothing size

    pixsize = np.sqrt(4.0 * np.pi / npix)
    fwhm = pixperscale * pixsize
    bl = hp.sphtfunc.gauss_beam(fwhm, lmax)

    # Smooth the alm

    alm_s = hp.sphtfunc.almxfl(alm, bl)
    
    # Back to pixel space

    stat_out = hp.sphtfunc.alm2map(alm_s, nsidecovar)

    return stat_out

def whitening_matrix(R):

    """This function computes the whitening matrix W of positive definite square matrix R (the Cholesky decomposition of the invert: R^{-1}=W^t*W so that W*R*W^t=I).
    """

    evec0, eval, V = np.linalg.svd(R, full_matrices=True)
    evec = np.transpose(evec0)
    D = np.diagflat(eval)
    W = np.array(np.linalg.inv(np.matrix(np.sqrt(D))) * np.matrix(evec))

    return W

def ilc(covar, a_keep):
    
    """Computes standard ILC weights."""

    nf = (covar[:, 0]).size

    v = np.array(np.matrix(a_keep) * np.linalg.inv(np.matrix(covar)))

    w = v / np.sum(a_keep * v)

    return w

def multi_ilc(Ls_minus, Ps, whitened_mixing_matrix, covar):

    """Computes multidimensional ILC weights."""

    O = whitened_mixing_matrix

    # ILC weigths matrix

    B = np.array(np.matrix(Ls_minus) * np.matrix(O) * np.linalg.inv(np.transpose(np.matrix(Ls_minus) * np.matrix(O)) * np.transpose(np.matrix(Ps)) * np.linalg.inv(np.matrix(covar)) * np.matrix(Ps) * np.matrix(Ls_minus) * np.matrix(O)) * np.transpose(np.matrix(Ls_minus) * np.matrix(O)) * np.transpose(np.matrix(Ps)) * np.linalg.inv(np.matrix(covar)) * np.matrix(Ps))

    return B

def alm_index(lmin, lmax, lmaximum):

    """Get the indexes corresponding to a range of l in an alm array."""

    ni = ((lmax + 1) * (lmax + 2)) / 2 - ((lmin) * (lmin + 1)) / 2
  
    index = np.zeros(ni, dtype=np.int32)
    i = 0
  
    for l in range(lmin, lmax + 1): 
        for m in range(0, l + 1):
            index[i] = hp.sphtfunc.Alm.getidx(lmaximum, l, m)
            i = i + 1

    return index

def index2l_only(index, lmaximum):

    """Give value of l for a given alm index."""

    nl = index.size
    l = np.zeros(nl)

    for i in range(0, nl):
        l[i] = (hp.sphtfunc.Alm.getlm(lmaximum, index[i]))[0]

    return l




