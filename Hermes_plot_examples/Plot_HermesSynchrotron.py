import healpy
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
#import ROOT as rt
import numpy as np


Desktop = '/mnt/c/Users/pedro/OneDrive/Escritorio/'

def pixnumrectangle_symmetric(nside, lowb_North, highb_North, lowb_South, highb_South, lowl, highl):
    # moving to radiants
    lowb_North = lowb_North/180.*np.pi
    lowb_South = lowb_South/180.*np.pi
    lowl= lowl/180.*np.pi
    highb_North = highb_North/180.*np.pi
    highb_South = highb_South/180.*np.pi
    highl= highl/180.*np.pi
    npix=12*nside**2
    listpix = np.arange(npix)
    theta,phi = healpy.pixelfunc.pix2ang(nside,listpix, nest=False, lonlat=False)
    b = np.pi/2.-theta
    l = phi
    mask = []
    for i in np.arange(npix):
        if(l[i]>np.pi):
            l[i]-=2.*np.pi
        if(( b[i] <= highb_North and b[i] >= lowb_South)  and ((l[i] >= lowl) and (l[i] <= highl) or (l[i] >= -highl) and (l[i] <= -lowl)) ):
        #if((((b[i] >= lowb_North) and (b[i] <= highb_North) ) or ( (b[i] >= lowb_South) and (b[i] <= highb_South))) and ((l[i] >= lowl) and (l[i] <= highl) or (l[i] >= -highl) and (l[i] <= -lowl))):
            mask.append(1)
        else:
            mask.append(0)
    return mask


### MODELS
print('Opening synchrotron maps\n')
HermesRuns = '/home/tospines/PROJECTS/Positrons/Synchrotron_Runs/'
freqs = np.array([100, 1423, 2000, 2317, 10000, 22762, 32808, 40606, 60337, 100000])   #MHz
model_dat = []
for Comp in freqs:
    Comp_ = HermesRuns + 'synchro-JF12-' + str(Comp*1e-3) + 'GHz-window.fits.gz'
    #print(Comp_)
    hdul1 = pyfits.open(Comp_)
    model_dat.append(hdul1[1].data.astype(float)) # Units in T_b (brighness temperature)

    s = "Synchrotron emission - " + str(Comp*1e-3) + " GHz"
    plt.rcParams.update({'font.size':14.5})
    healpy.mollview(model_dat[-1], title = s, unit=r' $T_b$ [K]', norm = "hist", cmap = plt.cm.gnuplot, min = np.max(model_dat[-1])/500)
    healpy.graticule()
    plt.savefig(Desktop + "Synchrotron_maps/synchro-JF12-" + str(Comp*1e-3) + "GHz.png")
    plt.close()
    #exit()
    
NSIDE_model = hdul1[1].header['NSIDE']



### Some data to compare with
freq_dat = np.array([22, 45, 408, 1420, 2326, 23000, 33000, 41000, 61000, 94000])
Temp_freq52 = np.array([57963939.53, 65431891.3, 61584821.11, 46887020.05, 36795056.46, 11993539.46, 10951501.93, 10624678.31, 11635618.51, 2.6e7])
Err_up = np.array([90794972.15, 103271242, 96832378.57, 72477966.37, 56023572.38, 18753019.08, 18158994.43, 18158994.43, 20654248.4, 57856239])
Err_low = np.array([47695103.77, 54248957.7, 50866586.94, 38073078.77, 29429494.11, 6913688.254, 5885898.822, 5518918.646, 4130849.68, 7373413.555])


########################################  AVERAGE DIFFUSE EMISSION SPECTRA  ###########################################################

print('\nAveraging over sky positions')
Ub = 45. 
Lb = 10. 
Ul, Ll = 340., 40. 
low =   [-180] 
high =  [180]
Model_av, Model_avH4, Model_avP11, Model_avS8 = [], [], [], []
GS_av = []
for ll, hh in zip(low, high):  
    ipix_diskM = pixnumrectangle_symmetric(nside=NSIDE_model, lowb_North = 0., highb_North = Ub, lowb_South= Lb, highb_South = 0., lowl = ll, highl = hh)
    [Model_av.append(np.sum(model_dat[ifreq]*ipix_diskM)/np.sum(ipix_diskM)) for ifreq, myfreq in enumerate(freqs)]

Omega = np.sum(ipix_diskM)/len(ipix_diskM)*4*np.pi


######################################################  PLOTTING SECTION  ##################################################################################

KB = 8.617e-5 * 1e-9 #eV/K --> GeV/K
c = 3.e10 #cm/s
T_GS =( np.array(GS_av)*c**2)/(2*KB*(Nu_GS*1e+6)**2) *624.151 * Omega # erg/cm^2/s/sr/Hz --> GeV/cm^2/s/sr/Hz --> K/s/Hz == K

print('\nplotting... ')
ei = 0.8
slop = 2.5
plt.style.use('seaborn-poster') #seaborn-poster, seaborn-darkgrid, seaborn-talk
fig1, ax = plt.subplots(figsize = (12, 7.5))
#fig1.suptitle("Synchrotron spectrum", fontsize = 20, y = 0.95) #Galactic plane #Mid latitud
plt.yscale('log')
plt.xscale('log')
plt.errorbar(freq_dat, Temp_freq52, [Temp_freq52-Err_low, Err_up-Temp_freq52], fmt='ko', ms = 7, label = 'data') 
plt.semilogx(freqs, np.array(Model_av)*(freqs**slop), 'k-', linewidth = 2, label = 'DRAGON JF12 H4') ### k:
plt.legend(fontsize = 18, ncol = 3, frameon = False, loc = 'lower left') #ncol=3
plt.text(2.5e4, 1.8e8, str(Lb) + r'$^{\circ} <$ b < ' + str(Ub) r'$^{\circ}$', fontsize = 19)
plt.text(2.5e4, 1.1e8, str(Ll) + r'$^{\circ} <$ l < ' + str(Ul) r'${\circ}$', fontsize = 19)
plt.grid(color= '0.9', linestyle='dashed', axis = 'both')
ax.tick_params(bottom=True, top=False, left=True, right=True)
ax.tick_params(labeltop=False, labelright=True)
plt.xlim(left=100)
plt.ylabel(r'$\nu^{2.5}\,T \,\, [K * MHz^{-2.5}] $ ', fontsize = 22)
plt.xlabel('Frequency [MHz]', fontsize = 22)
plt.yticks(fontsize= 20)
plt.xticks(fontsize= 20)
plt.tight_layout()
plt.savefig(Desktop + "Synchrotron_test.png")
plt.savefig(Desktop + "Synchrotron_test.pdf")
#plt.show()
plt.close()

print('Alright!')
