import healpy
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
#import ROOT as rt
import numpy as np
from scipy.signal import savgol_filter
#from matplotlib import rcParams

inpfile_map = 'fermi_healpix_NSIDE256.fits'
with pyfits.open(inpfile_map) as hdulist: 
    Ebins = hdulist[1].header['TFIELDS']
    Evec = hdulist[2].data  #KeV
    NSIDE = hdulist[1].header['NSIDE']
    #mymap = print(hdulist[1].data)

maps = healpy.read_map(inpfile_map, field=range(Ebins)) 
npix = 12*(NSIDE**2)
PASS8_counts = np.array(maps)

data = "Catalog-4FGL_DR2.dat"
Gal_LON_LAT = np.loadtxt(data, skiprows = 0, usecols = (1, 2))  #Column 1 is the Galactic longitude (l) and 2 is the latitude (b)
Source_mask = np.ones(npix)
for i in range(len(Gal_LON_LAT)):
    pixx = healpy.pixelfunc.ang2pix(nside=NSIDE, theta=np.pi/2. - Gal_LON_LAT.T[1][:]*np.pi/180, phi=Gal_LON_LAT.T[0][:]*np.pi/180, nest=False, lonlat=False)
    Source_mask[pixx] = 0.
PASS8_Background = PASS8_counts*Source_mask


for i in range(0, 60, 5): #Fermi data
    s = "Fermi-LAT PASS8 CLEAN - Count map No Sources - " + str(round(0.5*(Evec[i][2]+Evec[i][1])*1e-6, 2)) + " GeV"
    plt.rcParams.update({'font.size':14.5})
    healpy.mollview( PASS8_Background[i], title = s, unit=r'counts', norm = "hist", cmap = plt.cm.rainbow)
    healpy.graticule()
    s2 = "Fermi-LAT PASS8 CLEAN-Count map-No Sources " + str(round(0.5*(Evec[i][2]+Evec[i][1])*1e-6, 2)) + "GeV"
    #plt.show()
    plt.savefig("/mnt/c/Users/pedro/OneDrive/Escritorio/" + s2.replace(' ', '_') + "NSIDE{}.png".format(NSIDE))
    #plt.savefig("/mnt/c/Users/pedro/OneDrive/Escritorio/" + s2.replace(' ', '_') + "NSIDE{}.pdf".format(NSIDE))
    plt.close()


