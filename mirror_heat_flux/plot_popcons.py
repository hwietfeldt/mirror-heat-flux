"""
"""

import numpy as np
import matplotlib.pyplot as plt

def get_heat_flux(B0, Bw):
    """
    Returns parallel heat flux on wall in MW/m^2
    """
    # TODO: Add fixed parameters to config.toml file
    eta_b = 0.8
    pb_mw = 25 # [MW]
    q_phy = 0.25
    eb_100keV = 1
    n25 = 1
    heat_flux = eta_b * pb_mw * (1 + q_phy / 5) * B0 * Bw
    heat_flux = heat_flux / (2*np.pi * n25**2 * np.sqrt(eb_100keV))
    return heat_flux