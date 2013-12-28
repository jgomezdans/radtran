#!/usr/bin/env python

"""Leaf angle distributions test codes
"""
import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt

def kuusk_lad ( theta_l, epsilon, theta_m ):
    """Kuusk leaf area distribution (LAD) function, introduced by
    Kuusk (1994, 1995, 1995). The distribution for zenith angle `theta_l` is
    controlled by two parameters, `epsilon` and `theta_m`. The former 
    describes the leaf orientation (from 0, spherical, to 1, constant at some
    angle), and `theta_m` controls the actual leaf angle, 0 for planophile and
    $\pi/2$ for erectophile.
    
    Note:
        The distribution isn't actually defined for $\epsilon=0$, as in that case,
        $\eta$ and $\mu$ evaluate to 0, and the calculation results in $\log(1)$ in
        the denominator of $b$
    
    Parameters
    ----------
    theta_l: float
        Zenith angle in radians
    epsilon: float
        Epsilon parameter, must be between 0 (uniform) to 1 ($\delta$-like)
    theta_m: float
        Leaf inclination angle in radians
        
    Returns
    -------
    gl: float
        The leaf area distribution
        
    """
    if not 0 <= epsilon <= 1:
        raise ValueError
    
    eta = np.arcsin ( epsilon * np.cos ( theta_m ) )
    nu = np.arcsin ( epsilon * np.sin ( theta_m ) )
    
    b = np.cos ( theta_m )*np.log ( (np.cos(eta) + np.sin(nu))/
        (np.cos(nu) - np.sin ( eta ) ) ) - np.sin( theta_m )*( eta - nu )
    b = epsilon/b
    
    gl = b/np.sqrt(1-epsilon*epsilon*np.cos(theta_l - theta_m)**2)
    return gl
def elliptical_lad ( theta_l, ala=None, chi=None, rotated=False ):
    """One parameter ellipsoidal leaf area distribution (LAD) based on
    Campbell (1990) (also with extensions from Thomas & Winner (2000))
    This distribution considers leaves as small surface elements distributed
    on the surface of a prolate or oblate ellipsoid. The distribution is then
    controlled by a single parameter, `chi`, or alternatively, by a mean
    leaf angle, `ala`. Further, Thomas and Winner (2000) extend the description
    of the distribution by considering a rotated distribution, on the grounds that
    horizontal leaf distributions are ecologically "optimal", and thus a 
    distribution should be able to cope with entirely planophile canopies.
    
    We select these different parametrisations using optional parameters.
    
    Parameters
    ----------
    theta_l: float
        The leaf angle in radians
    ala: float, optional
        The average leaf angle for the canopy
    chi: float, optional
        The value of $\chi$ for the distribution. If not given, it's calculated
        from the value of `ala` above
    rotated: Boolean
        Whether to use the rotated ellipsoid or the non-rotated one.
        
    Returns
    --------
    gl: float
        The LAD for $\theta_{l}$, assuming a value of $\chi$
        
    """
    if chi is None:
        if ala is not None:
            # This bit calculates chi from ALA using the empirical formula
            if rotated:
                chi = -3 * np.power ( ( (0.5*np.pi*ala)/9.65), -0.6061 )
            else:
                chi = -3  + np.power( ( ala/9.65), -0.6061 )
        else:
            raise ValueError
    if chi < 1:
        epsilon = np.sqrt ( 1 - chi*chi )
        Lambda = chi + np.arcsin(epsilon)/epsilon
    elif chi > 1:
        epsilon = np.sqrt ( 1 - np.power(chi, -2) )
        Lambda = chi + np.log ( ( 1 + epsilon )/( 1 - epsilon ) )/(2*epsilon*chi)
    elif chi == 1:
        Lambda = 2.
        
    if rotated:
        # Basically, we have this transformation
        #theta_l = theta_l - np.pi/2.
        gl = 2*chi*chi*chi*np.cos ( theta_l )
        gl = gl/(Lambda*(np.sin(theta_l)**2 + chi*chi*np.cos(theta_l)**2 )**2 )
    else:
        gl = 2*chi*chi*chi*np.sin ( theta_l )
        gl = gl/(Lambda*(np.cos(theta_l)**2 + chi*chi*np.sin(theta_l)**2 )**2 )
    return gl

def calculate_projection ( theta_l, theta_p ):
    """Calculates the projection of $\mathbf{\Omega}_{l}$ onto
    $\mathbf{\Omega}_{p}$, or $\left|\mathbf{\Omega}_{l}\cdot $\mathbf{\Omega}_{p}\right|$,
    as given in Myneni Eq III.13.
    """
    ang_test = np.abs ( (np.cos ( theta_p )/np.sin( theta_p ))*\
                       (np.cos( theta_l )/np.sin(theta_l)))
    psi = np.cos ( theta_p )*np.cos ( theta_l )
    phi = np. where ( ang_test < 1, np.arccos ( -ang_test ), 0 )
    x = np.where ( ang_test > 1, psi, \
          psi*(2.*phi/np.pi - 1.) + 2./np.pi* \
          np.sqrt(1. - np.cos( theta_l )**2) * \
          np.sqrt(1. - np.cos( theta_p )**2)*np.sin(phi) )
    return x

def calculate_big_g ( theta_p, chi, rotated=False ):
    """Calculates $G(\Omega_{p})$ according to the following assumptions:
    1. The azimuth and zenith LAD are independent
    2. There's azimuthal symmetry
    3. We use an elliptical/rotated elliptical LAD
    
    Assumption 2 allows us to only do a single integral over $\theta_{l}$, as the
    integral in azimuth is just over a constant (and is cancelled out by the
    $1/2\pi$ term in the definition of $G$).

    Parameters
    ----------
    theta_p: float
        The look angle in azimuth in radians
    chi: float
        The value of $\chi$ for the elliptical distribution
    rotated: Boolean
        Whether to use the rotated/unrotated distribution
    """
    def integrand (  theta_l, theta_p, chi ):
        
        return elliptical_lad ( theta_l, chi=chi, \
            rotated=rotated )*\
            calculate_projection ( theta_l, theta_p )
    
    G = quad ( integrand, 0, np.pi/2., \
                 args=(theta_p, chi ))
    return G

def test_big_G ():
    """Graphical test of $G$..."""
    s = np.linspace( 0, np.pi/2., 50 )
    x = np.rad2deg( s )
    for chi in [ 0.5, 1, 1.5]:
        y = []
        for theta_p in s:
            y.append ( calculate_big_g ( theta_p, chi )[0] )
        plt.plot ( x, y, label="$\chi$=%4.2f" % chi )
    plt.legend(loc="best")
    plt.ylabel (r'$G(\Omega)$')
    plt.xlabel(r'$\theta_{p}\;[^{\circ}]$')

def test_campbell( rotated=False ):
    """Graphical test of the Campbell/rotated campbell LAD"""
    s = np.linspace( 0, np.pi/2., 100 )
    if rotated:
        for chi in np.logspace ( -1, 1, 5 ):
            plt.plot ( np.rad2deg (s), elliptical_lad ( s, chi=chi, rotated=True), label="%3g" % chi )
        plt.legend()
        plt.title("Rotated")
        plt.ylabel (r'$g_{l}(\theta_{l})$' )
        plt.xlabel(r'$\theta_{l}\;[^{\circ}]$')

    else:
        for chi in np.logspace ( -1, 1, 5 ):
            plt.plot ( np.rad2deg (s), elliptical_lad ( s, chi=chi, rotated=False), label="%3g" % chi )
        plt.legend()
        plt.title( "Campbell Original")
        plt.ylabel (r'$g_{l}(\theta_{l})$' )
        plt.xlabel(r'$\theta_{l}\;[^{\circ}]$')

def test_kuusk():
    """Graphical test of the Kuusk distribution"""
    s = np.linspace( 0, np.pi/2., 100 )
    plt.plot ( np.rad2deg (s), kuusk_lad ( s, 0.001, np.pi/2. ), label="Sph")
    plt.plot ( np.rad2deg (s), kuusk_lad ( s, 0.9, 0 ), label="Plano")
    plt.plot ( np.rad2deg (s), kuusk_lad ( s, 0.9, np.pi/2 ), label="Erecto")
    plt.plot ( np.rad2deg (s), kuusk_lad ( s, 0.9, np.pi/4 ), label="Plagio")

    plt.title( "Kuusk")
    plt.ylabel (r'$g_{l}(\theta_{l})$' )
    plt.xlabel(r'$\theta_{l}\;[^{\circ}]$')

def test_all ():
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages('leaf_angle_test.pdf')
    test_kuusk()
    pdf.savefig()
    plt.close()
    test_campbell( rotated=False )
    pdf.savefig ()
    plt.close()
    test_campbell( rotated=True )
    pdf.savefig ()
    plt.close()
    test_big_G()
    pdf.savefig()
    plt.close()
    pdf.close()    
