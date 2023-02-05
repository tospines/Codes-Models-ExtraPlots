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


nside = 64
sun_pos = Vector3QLength(8.*kpc, 0*pc, 0*pc)

MinEnergy = 1  # GeV
MaxEnergy = 5e6  # GeV
E_points = 30 #50

DragonRuns = '/proj/dmsearches/users/x_pedde/Hermes/build/DragonRuns/'
filename = DragonRuns + "/OptModel.fits.gz"
print('Using model: ' + filename.replace(DragonRuns + '/', ''))

dragon2D_proton = cosmicrays.Dragon2D(filename, [Proton])
dragon2D_helium = cosmicrays.Dragon2D(filename, [Helium])
cr_list = [dragon2D_proton, dragon2D_helium]
dragon2D_leptons = cosmicrays.Dragon2D([Electron, Positron])

kelner_crossections = interactions.KelnerAharonianGamma()
kn_crosssection = interactions.KleinNishina()
brems_crosssection = interactions.BremsstrahlungTsai74()

CMB_photons = photonfields.CMB()
ISRF_photons = photonfields.ISRF()

XCOvalues = np.array([1.8, 1.8, 1.8, 3.5, 3.5, 4., 4.5, 7.5, 7.5, 7.5, 8., 8.])
#Correspond to the bins: {0_kpc, 0.86_kpc, 2_kpc, 3_kpc,  4_kpc,  5_kpc, 6_kpc, 7_kpc, 9_kpc, 12_kpc, 15_kpc, 18_kpc}
neutral_gas_HI = neutralgas.RingModel(neutralgas.GasType.HI)
neutral_gas_H2 = neutralgas.RingModel(neutralgas.GasType.H2, XCOvalues)


# Deiniing integrator for the model               
integratorHI = PiZeroAbsorptionIntegrator(cr_list, neutral_gas_HI, kelner_crossections)
integratorH2 = PiZeroAbsorptionIntegrator(cr_list, neutral_gas_H2, kelner_crossections)
#integratorHI = PiZeroIntegrator(cr_list, neutral_gas_HI, kelner_crossections)
#integratorH2 = PiZeroIntegrator(cr_list, neutral_gas_H2, kelner_crossections)
integratorIC_cmb  = InverseComptonIntegrator(dragon2D_leptons, CMB_photons, kn_crosssection)
integratorIC_isrf = InverseComptonIntegrator(dragon2D_leptons, ISRF_photons, kn_crosssection)
integratorBremsHI = BremsstrahlungIntegrator(dragon2D_leptons, neutral_gas_HI, brems_crosssection)
integratorBremsH2 = BremsstrahlungIntegrator(dragon2D_leptons, neutral_gas_H2, brems_crosssection)

integratorHI.setupCacheTable(120, 120, 24) # nx,ny,nz
integratorHI.setObsPosition(sun_pos)
integratorH2.setupCacheTable(120, 120, 24)
integratorH2.setObsPosition(sun_pos)
integratorH2.setupCacheTable(120, 120, 24)

integratorIC_cmb.setupCacheTable(30, 30, 12)
integratorIC_cmb.setObsPosition(sun_pos)
integratorIC_isrf.setupCacheTable(30, 30, 12)
integratorIC_isrf.setObsPosition(sun_pos)

integratorBremsHI.setupCacheTable(120, 120, 24) # nx,ny,nz
integratorBremsHI.setObsPosition(sun_pos)
integratorBremsH2.setupCacheTable(120, 120, 24)
integratorBremsH2.setObsPosition(sun_pos)


print("building SkyMapRange object for pi0")
skymapHI_range = GammaSkymapRange(nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)
skymapH2_range = GammaSkymapRange(nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)
print("building SkyMapRange object for IC")
skymapIC_cmb_range = GammaSkymapRange(nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)
skymapIC_isrf_range = GammaSkymapRange(nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)
print("building SkyMapRange object for Brems")
skymapBremsHI_range = GammaSkymapRange(nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)
skymapBremsH2_range = GammaSkymapRange(nside, MinEnergy*GeV, MaxEnergy*GeV, E_points)

#top_left_edge = [30*deg, -30*deg]
#bottom_right_edge = [-180*deg, 180*deg]
#mask = RectangularWindow(top_left_edge, bottom_right_edge)
mask_edges = ([90*deg, -90*deg], [360*deg, 0*deg])  #All sky!!
mask = RectangularWindow(*mask_edges)

skymapHI_range.setMask(mask)
skymapH2_range.setMask(mask)
skymapIC_cmb_range.setMask(mask)
skymapIC_isrf_range.setMask(mask)
skymapBremsHI_range.setMask(mask)
skymapBremsH2_range.setMask(mask)



print("starting calculation of the maps... ")  # -------------------------------------------------------

# Calculation of HI skymap
skymapHI_range.setIntegrator(integratorHI)
skymapHI_range.compute()
nameHI = "Opt-Abs_HI_nside{}".format(nside)
skymapHI_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(nameHI)))
print("pi0 HI done")

# Calculation of H2 skymap
skymapH2_range.setIntegrator(integratorH2)
skymapH2_range.compute()
nameH2 = "Opt-Abs_H2_nside{}".format(nside)
skymapH2_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(nameH2)))
print("pi0 H2 done")




# Calculation of IC skymap (CMB)
skymapIC_cmb_range.setIntegrator(integratorIC_cmb)
skymapIC_cmb_range.compute()
nameIC1 = "Opt-IC_cmb_nside{}".format(nside)
skymapIC_cmb_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(nameIC1)))
print("IC on CMB done")

# Calculation of IC skymap (ISRF)
skymapIC_isrf_range.setIntegrator(integratorIC_isrf)
skymapIC_isrf_range.compute()
nameIC2 = "Opt-IC_isrf_nside{}".format(nside)
skymapIC_isrf_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(nameIC2)))
print("IC on ISRF done")





# Calculation of Brems H2 skymap
skymapBremsH2_range.setIntegrator(integratorBremsH2)
skymapBremsH2_range.compute()
nameBremsH2 = "Opt-BremsHI_nside{}".format(nside)
skymapBremsH2_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(nameBremsH2)))
print("Brems on HI done")

# Calculation of Brems HI skymap
skymapBremsHI_range.setIntegrator(integratorBremsHI)
skymapBremsHI_range.compute()
nameBremsHI = "Opt-BremsH2_nside{}".format(nside)
skymapBremsHI_range.save(outputs.HEALPixFormat("!{}.fits.gz".format(nameBremsHI)))
print("Brems on H2 done")




print("\nAllright!!\n")
