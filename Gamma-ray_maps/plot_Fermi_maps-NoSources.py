import healpy
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
#import ROOT as rt
import numpy as np
from scipy.signal import savgol_filter
#from matplotlib import rcParams

  
inpfile_exp = "fermi_exposure-z105_healpix_NSIDE256.fits"
with pyfits.open(inpfile_exp) as hdulist:
    Ebins = hdulist[1].header['TFIELDS']
    Evec = hdulist[2].data  #MeV
    NSIDE = hdulist[1].header['NSIDE']
    #mymap = print(hdulist[1].data)
                       
inpfile_map = 'fermi_healpix_NSIDE256.fits'
with pyfits.open(inpfile_map) as hdulist: 
    Bins = np.array(list(hdulist[2].data)) # KeV
    #mymap = print(hdulist[1].data)
    
low = Bins[:,1]*1e-3 
high = Bins[:,2]*1e-3 
DeltaE = np.array(high) - np.array(low) #in MeV

maps_exp = healpy.read_map(inpfile_exp, field=range(Ebins)) 
maps = healpy.read_map(inpfile_map, field=range(Ebins)) 
npix = 12*(NSIDE**2)
degree_per_pixel = 4*np.pi/npix
Flux = (maps/(maps_exp+1.e-14))/degree_per_pixel  # 'counts $cm^{-2}$ $s^{-1}$ $sr^{-1}$ '
PASS8_intensity = [np.array(Flux[i])/(DeltaE[i]*1e-3) for i in range(Ebins)] # 'counts cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$'
PASS8_intensity = np.array(PASS8_intensity)

data = "Catalog-4FGL_DR2.dat"
Gal_LON_LAT = np.loadtxt(data, skiprows = 0, usecols = (1, 2))  #Column 1 is the Galactic longitude (l) and 2 is the latitude (b)
Source_mask = np.ones(npix)

for i in range(len(Gal_LON_LAT)):
    pixx = healpy.pixelfunc.ang2pix(nside=NSIDE, theta=np.pi/2. - Gal_LON_LAT.T[1][:]*np.pi/180, phi=Gal_LON_LAT.T[0][:]*np.pi/180, nest=False, lonlat=False)
    Source_mask[pixx] = 0.
PASS8_Background = PASS8_intensity*Source_mask



for i in range(0, 60, 5): #Fermi data
    s = "Fermi-LAT PASS8 CLEAN - No Sources - " + str(round(Evec[i][0]*1e-3, 2)) + " GeV"
    print(s)
    plt.rcParams.update({'font.size':14.5})
    healpy.mollview( PASS8_Background[i] *(Evec[i][0]*1e-3)**2, title = s, unit=r'counts $GeV$ $cm^{-2}$ $s^{-1}$ $sr^{-1}$', norm = "hist", cmap = plt.cm.rainbow, min = np.max( (PASS8_Background[i] *(Evec[i][0]*1e-3)**2)/100000.))
    healpy.graticule()
    s2 = "Fermi-LAT PASS8 CLEAN-No Sources " + str(round(Evec[i][0]*1e-3, 2)) + "GeV"
    #plt.show()
    plt.savefig("/mnt/c/Users/pedro/OneDrive/Escritorio/" + s2.replace(' ', '_') + "NSIDE{}.png".format(NSIDE))
    plt.savefig("/mnt/c/Users/pedro/OneDrive/Escritorio/" + s2.replace(' ', '_') + "NSIDE{}.pdf".format(NSIDE))
    plt.close()


