import os
import sys
sys.path.append('/home/tospines/Hermes/build/')
from pyhermes import *
from pyhermes.units import *
           
import numpy as np
import healpy
import matplotlib.pyplot as plt


DragonRun = '/proj/dmsearches/users/x_pedde/Hermes/build/DragonRuns/run_3D.fits.gz'
ID_run = 'run3D_H4kpc'
B_field = 'JF12' #'Sun08' #'Pshirkov2011' #'JF12'


nside = 64
mfield = magneticfields.JF12()     #mfield = magneticfields.PT11()     #mfield = magneticfields.Sun08()
### mfield = magneticfields.WMAP07() --> does not work

#mfield.randomTurbulent(1)  ## Only if FFTW3 is installed
#mfield.randomStriated(1)    ## It does not work!
print('Magnetic field model set')
#dragon3D_leptons = cosmicrays.Dragon3D(DragonRun, [Electron, Positron])
dragon3D_leptons = cosmicrays.Dragon2D([Electron, Positron]) 

integrator = SynchroIntegrator(mfield, dragon3D_leptons)
sun_pos = Vector3QLength(8.5*kpc, 0*pc, 0*pc)
integrator.setObsPosition(sun_pos)


freq = np.logspace(1, np.log10(5e5), 40)*MHz
#print(freq)
for ifreq in freq:
    skymap = RadioSkymap(nside, ifreq)
    skymap.setIntegrator(integrator)
    skymap.compute()
    name = "synchro-{}-{}MHz-{}".format(B_field, freq/MHz, ID_run)
    skymap.save(outputs.HEALPixFormat("{}.fits.gz".format(name)))
    skymap.delete()	    
