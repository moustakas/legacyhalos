# These functions recreate the Illustris stellar mass profiles from Pillepich et al. 2018. The paramaters are obtained from Table 3 in the paper.

def get_R200c(M_halo, halo=True):
    if halo:
        a, b = 0.33, 2.99
    else:
        a, b = 0.70, 2.88
       
    m = M_halo - 14
    R_200c = a*m+b
    return R_200c

def get_Mstars_200c(M_halo, halo=True):
    if halo:
        a, b = 0.74, 12.04
    else:
        a, b = 1.56, 11.80
    m = M_halo - 14
    Mstars_200c = a*m+b
    return Mstars_200c

def get_sigmoidal_slope(M_halo, halo=True):
    if halo:
        a, b = -0.25, 2.14
    else:
        a, b = -0.52, 2.22
    m = M_halo - 14
    sigmoidal_slope = a*m+b
    return sigmoidal_slope

def get_x05(M_halo, halo=True):
    if halo:
        a, b = 0.19, -1.42
    else:
        a, b = 0.41, -1.48
    m = M_halo - 14
    x05 = a*m+b
    return x05

def enclosed_mstar(M_halo, halo=True):
    logR200c = get_R200c(M_halo, halo=halo)
    logMstar_norm = get_Mstars_200c(M_halo)
    slope = get_sigmoidal_slope(M_halo)
    x05 = get_x05(M_halo)
    
    lograd = np.linspace(0, 4)
    print(M_halo, logMstar_norm, logR200c)
    xx = lograd - logR200c
    
    Mstar_enclosed  = 10**logMstar_norm / (1+np.exp(-slope*(xx-x05)))
    Mstar_enclosed_frac = 10**(np.log10(Mstar_enclosed) - logMstar_norm)

    return xx, Mstar_enclosed_frac

