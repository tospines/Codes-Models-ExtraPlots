import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
import numpy as np
from scipy.optimize import curve_fit
import emcee
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D
import corner
from multiprocessing import Pool
import matplotlib

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
    theta,phi = hp.pixelfunc.pix2ang(nside,listpix, nest=False, lonlat=False)
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



datapath = "/proj/dmsearches/users/x_pedde/GammaLineSearch/"
CountMap = "PASS8_HEALPIX_Z90-210Bins.fits"     #"PASS7_HEALPIX_Z100-210Bins.fits"
ExposureMap = 'Fermi_Exposure_Z90-210Bins.fits'  #'PASS7_Exposure_Z100-210Bins.fits'

Exposure = 0

### Taking the Fermi data!
with pyfits.open(ExposureMap) as hdulist:
    Ebins_PASS8 = hdulist[1].header['TFIELDS']
    E_PASS8 = np.array(list(hdulist[2].data)).flatten()  #MeV
    nside = hdulist[1].header['NSIDE']

with pyfits.open(CountMap) as hdulist:
    Bins_PASS8 = np.array(list(hdulist[2].data)) # KeV
    #print(hdulist[1].header['ORDERING'])

maps_exp_PASS8 = hp.read_map(ExposureMap, field=range(Ebins_PASS8), dtype=np.float64, nest=False)
maps_PASS8 = hp.read_map(CountMap, field=range(len(Bins_PASS8)), dtype=np.float64, nest=False) 
npix_PASS8 = 12*(nside**2)
degree_per_pixel_PASS8 = 4*np.pi/npix_PASS8
if Exposure == 1:
    Flux_PASS8 = (maps_PASS8/(maps_exp_PASS8))/degree_per_pixel_PASS8  # 'counts $cm^{-2}$ $s^{-1}$ $sr^{-1}$ '
    Flux_PASS8[maps_exp_PASS8 == 0] = 0
else:
    Flux_PASS8 = maps_PASS8  # 'counts '

    
Evec_PASS8 = np.array(E_PASS8)*1e-3   ## This is MeV->GeV
DeltaE_PASS8 = [Bins_PASS8[i,2]*1e-6 - Bins_PASS8[i,1]*1e-6 for i in range(0, len(E_PASS8))]  # to pass from KeV to GeV
if Exposure == 1:
    PASS8_intensity = [np.array(Flux_PASS8[i])/(DeltaE_PASS8[i]) for i in range(Ebins_PASS8)] # 'counts cm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$'
    PASS8_intensity = np.array(PASS8_intensity)
else:
    PASS8_intensity = Flux_PASS8    # Counts
PASS8_Background = PASS8_intensity
#print(PASS8_intensity.shape)


### Substraction of the sources
#data = datapath + "Catalog-4FGL_DR2.dat"
data = datapath + "Catalog-2FGL.dat"
Gal_LON_LAT = np.loadtxt(data, skiprows = 0, usecols = (1, 2))  #Column 1 is the Galactic longitude (l) and 2 is the latitude (b)
c_0 = 0.881
beta = 0.817
c_1 = 0.2016
sigma68 = np.sqrt((c_0**2)*(np.array(Evec_PASS8)**(-2*beta)) + c_1**2)  # Energy must be in GeV! Reference: Ackerman et al. ApJS 203, 4 (2012), arXiv:1206.1896
#print(list(sigma68)) ## This is in degrees!
#print(len(Gal_LON_LAT), len(Gal_LON_LAT[3]), PASS8_counts.shape)

for iE in range(len(Evec_PASS8)):
    Source_mask = np.ones(npix_PASS8)
    for ps in range(len(Gal_LON_LAT)):
        vec = hp.ang2vec(Gal_LON_LAT.T[0][ps], Gal_LON_LAT.T[1][ps], lonlat=True) ## In degrees
        ## `lonlat=True` switches `ang2vec` from requiring colatitude $\theta$ and longitude $\phi$ in radians to longitude and latitude in degrees (notice that also the order changes)
        pixx = hp.query_disc(nside=nside, vec=vec, radius=2*sigma68[iE]*np.pi/180) 
        Source_mask[pixx] = 0.
    PASS8_Background[iE] *= Source_mask

### Second mask!
ipix = pixnumrectangle_symmetric(nside=nside, lowb_North = 0., highb_North = 5., lowb_South= -5., highb_South = 0., lowl = 6, highl = 360)
Ipix = ipix
ipix = np.array(ipix)
ipix[np.array(Ipix)==1] = 0
ipix[np.array(Ipix)==0] = 1
PASS8_Background *= ipix 
PASS8_intensity *= ipix

## Defining ROIS
lon = 0
lat = 0
vec = hp.ang2vec(lon, lat, lonlat=True) # Center of the Galaxy
R180 = hp.query_disc(nside, vec, radius=np.radians(180))
R90 = hp.query_disc(nside, vec, radius=np.radians(90))
R41 = hp.query_disc(nside, vec, radius=np.radians(41))
R16 = hp.query_disc(nside, vec, radius=np.radians(16))
R3 = hp.query_disc(nside, vec, radius=np.radians(3))


##################################### AVERAGE EMISSION IN EACH ROI ###########################################################

ROI3 = np.zeros(npix_PASS8)
ROI3[R3] = 1
ROI16 = np.zeros(npix_PASS8)
ROI16[R16] = 1
ROI41 = np.zeros(npix_PASS8)
ROI41[R41] = 1
ROI90 = np.zeros(npix_PASS8)
ROI90[R90] = 1
ROI180 = np.zeros(npix_PASS8)
ROI180[R180] = 1


min_fitE, max_fitE = 5., 300.
Ecut = (np.array(Evec_PASS8) <= max_fitE) * (np.array(Evec_PASS8) >= min_fitE)
print(list(np.array(Evec_PASS8)[Ecut]))
#exit()

Regions = [ROI3, ROI16, ROI41, ROI90, ROI180]
Flux_Regions = []
for ri in range(len(Regions)):
    Flux_ = []
    for Ei in range(len(np.array(Evec_PASS8)[Ecut])):
        if ri == 0:
            Flux_.append(np.sum(PASS8_intensity[Ecut][Ei]*Regions[ri]))  ## For ROI3 there is no source substraction in PASS7 analysis paper!
        else:
            Flux_.append(np.sum(PASS8_Background[Ecut][Ei]*Regions[ri])) 
    Flux_Regions.append(Flux_)



## Taking Edisp matrix
myEdisp = []
with pyfits.open(datapath + "Edisp_Matrix_210Bins.fits") as hdulEdisp:  #"Edisp_Matrix_210Bins.fits"

    for idata in hdulEdisp[3].data: #this is E_real (the bins in which dispersion is calculated)
        myEdisp.append(idata[-1]) # idata[-1] is always of size E_measured (Same number of energy bis than in the HEALPIX file)
# np.array(myEdisp)[iE_real] would give you the array of size E_measured that would result from the energy dispersion at the energy E_real[iE_real]



## Background functions!
def Doubroken_PL(R, A, gamma, rho_b, Dgamma, rho_b2, Dgamma2):
    return A*(R**(-gamma))/((1 + (R/rho_b)**(Dgamma/0.01))**0.01) * (1/(1 + ((R/rho_b2)**(Dgamma2/0.01))**0.01))

def broken_PL(R, A, gamma, rho_b, Dgamma):
    return A*((R/1.)**(-gamma))/((1 + (R/rho_b)**(Dgamma/0.004))**0.004)

def broken_PL2(R, A, gamma, rho_b, Dgamma):
    f = SmoothlyBrokenPowerLaw1D(A, rho_b, gamma, gamma + Dgamma)
    return f(R)

def simple_PL(R, A, gamma):
    return A*(R**(-gamma))

def PL_smooth(R, A, gamma, rho_b, Dgamma):
    K = 1e22
    return A*(R**(-gamma))*(1-((1 + np.tanh(K*(R-rho_b)))/2.)) + ((1 + np.tanh(K*(R-rho_b)))/2.) * (A*rho_b**(-gamma))*((R/rho_b)**(-(gamma+Dgamma)))


### Minimization block
myPool = 20

SlidingEnergy = np.logspace(np.log10(min_fitE), np.log10(max_fitE), 88) #88 points
E_Lbound = (Bins_PASS8.T[1]*1e-6)    #[Ecut]
E_Fermi = np.array(Evec_PASS8)[Ecut]
E_window = 1.5 ## 3 is the best for PASS7 data!
#print('\n', list(sigma68[Ecut])) ## This is in degrees!
def lnprob(rens, Region, E_gamma):
    A, gamma = rens
    if (A < 0) or (gamma < -0.5) or (gamma > 4.):
        return -np.inf
    s = np.round(Flux_Regions[Region])[(E_gamma/E_window <E_Fermi) * (E_window*E_gamma>E_Fermi)] # measured events
    nu = np.round(simple_PL(E_Fermi[(E_gamma/E_window<E_Fermi) * (E_window*E_gamma>E_Fermi)], A*1e2, gamma))  # expected events
    nu[nu==0] = 1
    mysum = []
    for si in range(len(s)):
        summatory = [np.log(i) for i in range(si, 0, -1)]
        mysum.append(np.sum(summatory))
        del summatory
        
    return np.sum(-nu + s*np.log(nu) - np.array(mysum) ) 


def lnprob_S(rens, Region, E_gamma):
    A, gamma, n_s = rens
    if (A < 0) or (gamma < -0.5) or (gamma > 4.) or (n_s < 0):
        return -np.inf
    if E_gamma <= E_Lbound[0]:
        Line_ind = 0
    else:
        Line_ind = np.where(E_Lbound==(E_Lbound[E_gamma > E_Lbound])[-1] )
    S = n_s*np.array(myEdisp)[Line_ind][0][Ecut]
    s = np.round(Flux_Regions[Region])[(E_gamma/E_window <E_Fermi) * (E_window*E_gamma>E_Fermi)] # measured events
    nu = np.round(simple_PL(E_Fermi[(E_gamma/E_window <E_Fermi) * (E_window*E_gamma>E_Fermi)], A*1e2, gamma) + S[(E_gamma/E_window <E_Fermi)*(E_window*E_gamma>E_Fermi)])  # expected events
    nu[nu==0] = 1
    mysum = []
    for si in range(len(s)):
        summatory = [np.log(i) for i in range(si, 0, -1)]
        mysum.append(np.sum(summatory))
        del summatory
    return np.sum(-nu + s*np.log(nu) - np.array(mysum) )


Amps = np.array([23.96, 155.98, 520.75, 1150.83, 2041])*100
ndim, nwalkers = 3, 100
MCMC_fits_S, MCMC_q_S, Bestlnprob_S = [], [], []
MCMC_fits, MCMC_q, Bestlnprob = [], [], []
Samples_NoS, Samples = [], []
print('\nStarting sliding window analysis...\n')
for Reg in range(len(Regions)):
    print("Starting analysis for region", ["ROI3", "ROI16", "ROI41", "ROI90", "ROI180"][Reg])
    fitsS_, BestlnprobS_, QuantilesS_ = [], [], []
    fits_, Bestlnprob_, Quantiles_ = [], [], []
    Samples_NoS_, Samples_ = [], []
    for E_g in SlidingEnergy:
        print("E_gamma:", E_g, "GeV")
        pos = [[Amps[Reg], 2, 0] + 1e-10*np.random.randn(ndim) for i in range(nwalkers)]     ## This is just to inizialize the MonteCarlo in a reasonable point
        pos_NoS = [[Amps[Reg], 2] + 1e-10*np.random.randn(ndim-1) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_S, args=[Reg, E_g], pool = Pool(myPool))
        pos, prob, state = sampler.run_mcmc(pos, 400)
        sampler.reset()
        sampler.run_mcmc(pos, 800)
        samples = sampler.flatchain

        fitsS_.append(np.median(samples, axis = 0))
        [QuantilesS_.append(corner.quantile(samples.T[id], [0.025, 0.16, 0.50, 0.84, 0.975])) for id in range(ndim)]
        BestlnprobS_.append(lnprob_S(np.median(samples, axis = 0), Reg, E_g) )
        
        sampler_NoS = emcee.EnsembleSampler(nwalkers, ndim-1, lnprob, args=[Reg, E_g], pool = Pool(myPool))
        pos_NoS, prob, state = sampler_NoS.run_mcmc(pos_NoS, 400)
        sampler_NoS.reset()
        sampler_NoS.run_mcmc(pos_NoS, 800)        
        samples_NoS = sampler_NoS.flatchain
        fits_.append(np.median(samples_NoS, axis = 0))
        [Quantiles_.append(corner.quantile(samples_NoS.T[id], [0.025, 0.16, 0.50, 0.84, 0.975])) for id in range(ndim-1)]
        Bestlnprob_.append(lnprob(np.median(samples_NoS, axis = 0), Reg, E_g) )
        
        sampler.reset()
        sampler_NoS.reset()
        Samples_.append(samples)
        Samples_NoS_.append(samples_NoS)
        
    MCMC_fits_S.append(fitsS_)
    MCMC_q_S.append(QuantilesS_)
    Bestlnprob_S.append(BestlnprobS_)
    
    MCMC_fits.append(fits_)
    MCMC_q.append(Quantiles_)
    Bestlnprob.append(Bestlnprob_)

    Samples.append(Samples_)
    Samples_NoS.append(Samples_NoS_)


print(MCMC_fits_S, '\n\n')
print(Bestlnprob_S, '\n\n')
print(Bestlnprob)

### Figures block
plt.style.use('seaborn-poster') #seaborn-poster, seaborn-darkgrid, seaborn-talk

for ir, iReg in enumerate(["ROI3", "ROI16", "ROI41", "ROI90", "ROI180"]):
    for iE, E_g in enumerate(SlidingEnergy):
        with open(datapath + '/CornerFiles/Corner_{}_ELine{}GeV.txt'.format(iReg, SlidingEnergy[iE]),"w+") as f:
            for s_lines in Samples[ir][iE]:
                f.write(str(s_lines) + '\n')
        f.close()

        with open(datapath + '/CornerFiles/CornerNoS_{}_ELine{}GeV.txt'.format(iReg, SlidingEnergy[iE]),"w+") as f:
            for s_lines in Samples_NoS[ir][iE]:
                f.write(str(s_lines) + '\n')
        f.close()


for ir, iReg in enumerate(["ROI3", "ROI16", "ROI41", "ROI90", "ROI180"]):
    fig = plt.figure(figsize = (12, 7.5))
    #plt.suptitle("log probability", fontsize = 20, y = 0.95)
    plt.xscale('log')
    plt.plot(SlidingEnergy, Bestlnprob_S[ir], label='Line fit - ROI 3 deg.')
    plt.plot(SlidingEnergy, Bestlnprob[ir], label='No line- ROI 3 deg.')
    #plt.plot(SlidingEnergy, -2*(np.array(Bestlnprob[ir])-np.array(Bestlnprob_S[ir])), label='TS')
    plt.axvline(x=133, label='133 GeV', c='black', linestyle='dashed')
    plt.legend(fontsize = 20)
    plt.xlabel('Energy (GeV)', fontsize = 20)
    plt.ylabel('log likelihood', fontsize = 20)
    plt.savefig(datapath + '/LogProb/Logprob_{}.png'.format(iReg))
    #plt.show()
    plt.close()
    
    fig = plt.figure(figsize = (12, 7.5))
    #plt.suptitle("log probability", fontsize = 20, y = 0.95)
    plt.xscale('log')
    s_local = np.sqrt(-2*(np.array(Bestlnprob[ir])-np.array(Bestlnprob_S[ir])))
    s_local[-2*(np.array(Bestlnprob[ir])-np.array(Bestlnprob_S[ir])) < 0] = 0.
    plt.plot(SlidingEnergy, s_local, label='sqrt(TS)')
    plt.axvline(x=133, label='133 GeV', c='black', linestyle='dashed')
    plt.legend(fontsize = 20)
    plt.xlabel('Energy (GeV)', fontsize = 20)
    plt.ylabel('s local', fontsize = 20)
    plt.savefig(datapath + '/LogProb/TS_{}.png'.format(iReg))
    #plt.show()
    plt.close()


print('\n\n Allright!!')


