from pyhermes import *
from pyhermes.units import *
import astropy.units as u
import numpy as np
import healpy
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.ticker as tick
import matplotlib.colors as colors
from matplotlib import cm
import sys

nside = int(sys.argv[1])  #Provide it as input
sun_pos = Vector3QLength(8.*kpc, 0*pc, 0*pc)

MinEnergy = 10  # GeV
MaxEnergy = 1e7  # GeV
E_points = 60

DragonRuns = '/proj/dmsearches/users/x_pedde/Hermes/build/DragonRuns/'
filename = DragonRuns + "/GammaOptimized_model.fits.gz"
print('Using model: ' + filename.replace(DragonRuns + '/', ''))

dragon2D_proton = cosmicrays.Dragon2D(filename, [Proton])
dragon2D_helium = cosmicrays.Dragon2D(filename, [Helium])
cr_list = [dragon2D_proton, dragon2D_helium]

kamae06 = interactions.Kamae06Neutrino()
kelahar = interactions.KelnerAharonianNeutrino()

XCOvalues = np.array([1.8, 1.8, 1.8, 3.5, 3.5, 4., 4.5, 7.5, 7.5, 7.5, 8., 8.])
#Correspond to the bins: {0_kpc, 0.86_kpc, 2_kpc, 3_kpc,  4_kpc,  5_kpc, 6_kpc, 7_kpc, 9_kpc, 12_kpc, 15_kpc, 18_kpc}
neutral_gas_HI = neutralgas.RingModel(neutralgas.GasType.HI)
neutral_gas_H2 = neutralgas.RingModel(neutralgas.GasType.H2, XCOvalues)


def integrate_template(integrator, Nside, name):
    
    integrator.setupCacheTable(120, 120, 24)
    sun_pos = Vector3QLength(8.0*kpc, 0*pc, 0*pc)
    integrator.setObsPosition(sun_pos)

    mask_edges = ([90*deg, -90*deg], [360*deg, 0*deg])  #This is all the sky
    mask = RectangularWindow(*mask_edges)
    
    skymap_range = GammaSkymapRange(Nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)
    skymap_range.setIntegrator(integrator)
    
    skymap_range.compute()
    skymap_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(name)))
    
    return skymap_range


def integrate_neutrino(cosmicrays, gas, crosssection, Name):
    NSIDE = nside
    integrator = PiZeroIntegrator(cosmicrays, gas, crosssection)
    return integrate_template(integrator, NSIDE, Name)



print("starting calculation of skymaps for my model... ")  # -------------------------------------------------------

# Calculation of HI skymap
print("Computing map for HI gas")
nameHI_kam = "Neutrino_Optmodel-kam_HI_nside{}".format(nside)
nameHI_kel = "Neutrino_Optmodel-kel_HI_nside{}".format(nside)
skymap_range_neutrino_HI_kamae06 = integrate_neutrino(cr_list, neutral_gas_HI, kamae06, nameHI_kam)
skymap_range_neutrino_HI_kelahar = integrate_neutrino(cr_list, neutral_gas_HI, kelahar, nameHI_kel)

print("Computing map for H2 gas")
nameH2_kam = "Neutrino_Optmodel-kam_H2_nside{}".format(nside)
nameH2_kel = "Neutrino_Optmodel-kel_H2_nside{}".format(nside)
skymap_range_neutrino_H2_kamae06 = integrate_neutrino(cr_list, neutral_gas_H2, kamae06, nameH2_kam)
skymap_range_neutrino_H2_kelahar = integrate_neutrino(cr_list, neutral_gas_H2, kelahar, nameH2_kel)


print("Done!")
