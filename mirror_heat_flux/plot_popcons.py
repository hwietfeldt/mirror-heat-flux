"""
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# TODO: Add fixed parameters to config.toml file
eta_b = 0.8
pb_mw = 25 # [MW]
q_phy = 0.25
eb_100keV = 1
n25 = 1
n_rl_end = 10 # Number of tritium gyroradii at end cell
Bm = 26 # [T]
heat_flux_thresh = 5 # [MW /m^2]
a_expansion_factor_max = 5 # a_w <= 5 a_0

def get_heat_flux(B0, Bw):
    """
    Returns parallel heat flux on wall in MW/m^2
    """
    heat_flux = eta_b * pb_mw * (1 + q_phy / 5) * B0 * Bw
    heat_flux = heat_flux / (2*np.pi * n25**2 * eb_100keV)
    return heat_flux

def get_radius_expansion(B0, Bw):
    a_0 = n25*np.sqrt(eb_100keV)/B0
    a_w = a_0 * np.sqrt(B0/Bw)
    return a_w / a_0

def get_min_Bw_from_end_size(B0):
    """
    Returns the minimum Bw required to satisfy the end-plug
    size requirement
    """
    min_Bw = B0 / a_expansion_factor_max**2
    return min_Bw

# def get_min_B0_from_end_rl():
#     return n_rl_end**2 / 


if __name__=='__main__':
    nsteps = 200
    B0_min = 0.1*Bm
    B0_max = 0.25*Bm

    Bw_min = 0.1
    Bw_max = 0.6

    # Create grid of B0 vs Bw
    B0 = np.linspace(B0_min, B0_max, nsteps)
    Bw = np.linspace(Bw_min, Bw_max, nsteps)
    B0_grid, Bw_grid = np.meshgrid(B0, Bw)

    # Get heat flux as function of B0, Bw
    heat_flux = get_heat_flux(B0_grid, Bw_grid)

    # Mask heat flux above 5 MW/m^2
    ok_heat_flux = np.ma.masked_where(heat_flux > heat_flux_thresh, heat_flux)

    # Get minimum Bw from physical size constraint
    min_Bw_end_size_grid = get_min_Bw_from_end_size(B0_grid)
    print(min_Bw_end_size_grid.shape)

    # POPCON style plot
    im = plt.imshow(
        ok_heat_flux,
        extent=[B0_grid.min(), B0_grid.max(), Bw_grid.min(), Bw_grid.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    cbar = plt.colorbar(im)
    cbar.set_label('$q_{||,w}$ [MW/m²]', fontsize=14)

    # Add a thick red contour line at the threshold ---
    red_contour = plt.contour(
        B0_grid, Bw_grid, heat_flux,
        levels=[heat_flux_thresh],
        colors='red', linewidths=3
    )
    plt.clabel(red_contour, fmt={heat_flux_thresh: f'{heat_flux_thresh:.0f} MW/m²'},
            fontsize=10, colors='red')
    
    # Add grey line where end-plug size-constraint is surpassed
    # plt.plot(B0, min_Bw_end_size_grid[0], linestyle='-', lw=3, c='grey', label=r'$a_w \leq 5a_0$')
    # diff = Bw_grid - min_Bw_end_size_grid
    # plt.contourf(B0_grid, Bw_grid, diff, levels=[diff.min(), 0], colors='white', alpha=1.0)

    # plt.clabel(size_constraint_contour, f'$a_w = {a_expansion_factor_max}a_0',
    #         fontsize=10, colors='grey')

    # Add a contour map of the radial expansion
    expansion_grid = get_radius_expansion(B0_grid, Bw_grid)
    expansion_levels = np.arange(3.0, 8.0, 1.0)
    expansion_contours = plt.contour(B0_grid, Bw_grid, expansion_grid,
                                     levels = expansion_levels, colors='k', linewidths=1.5, linestyles='solid')
    plt.clabel(expansion_contours, fmt=lambda v: f"$a_w/a_0$ = {v:.1f}", inline=True, fontsize=12, colors='k')

    plt.ylabel('$B_w$ [T]', fontsize=14)
    plt.xlabel('$B_0$ [T]', fontsize=14)
    plt.xlim(B0_grid.min(), B0_grid.max())
    plt.ylim(Bw_grid.min(), Bw_grid.max())
    #plt.legend(fontsize=14)
    plt.title("End Plug Heat Flux\n $E_b$ = 100 keV, $B_m$ = 26 T, $a_{FLR}=a_{abs}$, $P_{nbi}=25$ MW, $Q=0.25$")
    plt.savefig("heat_flux_map.png")
    plt.show()

