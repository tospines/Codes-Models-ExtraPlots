import healpy
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
import numpy as np

def pixnumrectangle_symmetric(nside,lowb_North,highb_North,lowb_South,highb_South,lowl,highl):
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
        if((((b[i] >= lowb_North) and (b[i] <= highb_North) ) or ( (b[i] >= lowb_South) and (b[i] <= highb_South))) and ((l[i] >= lowl) and (l[i] <= highl) or (l[i] >= -highl) and (l[i] <= -lowl))):
            mask.append(1)
        else:
            mask.append(0)
    return mask


Run_fold = '/Projects/DRAGON2/Rvariable_DRAGON2/Neutrinos/Neutrino_runs/' 
Desktop = '/mnt/c/Users/pedro/OneDrive/Escritorio/'

CompMinH2 = Desktop + Run_fold + '/Neutrino_VariableMin-kel_H2_nside64.fits.gz'
hdulMinH2 = pyfits.open(CompMinH2)
print('Energy bins: ', len(hdulMinH2)-1, '\n')
print(hdulMinH2[0].header)
print(hdulMinH2[1].header)
#exit()

GeV_units = 1/(1.60217648e-10) # From Joules to GeV
SI_units = 1.e-4 # m^-2 to cm^-2
## Extracting data
nu_dataMin, E_model = [], []
for hdulMin2 in hdulMinH2[1:]:
    nu_dataMin.append(2*hdulMin2.data.astype(float)*SI_units)  # GeV^-1 m^-2 s^-1 sr^-1
    ## The factor 2 needed to account for nu and anti-nu with the Kelner-Aharonian cross sections
    E_model.append(hdulMin2.header['ENERGY']*GeV_units)
#print(E_model) # Energy in GeV
NSIDE_model = hdulMinH2[1].header['NSIDE']
#print(NSIDE_model)
#print(np.array(nu_dataMin).shape, '\n\n')
E_model = np.array(E_model)
#exit()


########################### mapping on the zone #######################
# Full sky:
Ub = 90. 
Ul, Ll = 180., -180. 

ipixT = pixnumrectangle_symmetric(nside=NSIDE_model, lowb_North = 0, highb_North = Ub, lowb_South= -1*Ub, highb_South = 0, lowl = Ll, highl = Ul)
NuMin_av = [((np.sum(m*ipixT)/np.sum(ipixT))) for m in nu_dataMin]  

dOmega = np.sum(ipixT)/(12*NSIDE_model**2) * (4*np.pi)
#print(dOmega)



ei = 2.0
fig1 = plt.figure(figsize = (12, 7.5))
plt.yscale('log')
plt.semilogx(E_model, np.array(NuMin_av)*(E_model**ei)*dOmega, 'g-', label = r'$\gamma$-optimized - H2 - Min model', lw=5)
            
plt.legend(fontsize = 20, ncol = 3, frameon = False, loc = 2)

plt.grid(color= '0.9', linestyle='dashed', axis = 'both')
plt.text(60, 1.5e-9, '|b| < {} - |l| < {}'.format(Ub, Ul), fontsize = 22, fontweight='bold')
if (ei == 2.0):
    plt.ylabel(r'E$_{\nu}^{2}$ $\times$ d$\phi_{\nu}$/dE$_{\nu}$ [$GeV\, cm^{-2}s^{-1}]$', fontsize = 22)
    plt.ylim(bottom = 1.e-9, top = 1.e-5)
else:
    print('you must set your ylabel!')
    plt.ylim(bottom = 1.e-9, top=1.e-5)
plt.xlabel(r'$\nu$ Energy [GeV]', fontsize = 22)
plt.yticks(fontsize= 20)
plt.xticks(fontsize= 18)
plt.xlim(left = 50., right = 2.e7)
plt.savefig(Desktop + "Neutrino_test_Full.png")
plt.savefig(Desktop + "Neutrino_test_Full.pdf")
#plt.show()
plt.close()
