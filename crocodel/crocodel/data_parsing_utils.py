import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy import interpolate
import scipy
from astropy import units as un
from astropy import constants as con
from scipy.interpolate import splev, splrep
from astropy.modeling import models
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from tqdm import tqdm
import glob
import astropy.io.fits as fits
import copy
############### ############### ############### ############### ############### 
######## Parsing data modules ########### ############### ############### 


############### ############### ############### ############### ############### 
################# ################# ################# ################# 
################ Parsing CRIRES+ data (reduced from pyCRIRES) ################# 
################# ################# ################# ################# 
############### ############### ############### ############### ############### 

def get_data_BJD_phase_Vbary_crires_plus(datadir = None,  T0 = None, Porb = None, save = False, savedir = None, retres = False, infostring = None):
    
    




################## IGRINS ############### ############### ############### 
### Loop through each file, first for H and then for K band; read in the time, and  
############### ############### ############### ############### ############### 
############### ############### ############### ############### ############### 

def get_data_BJD_phase_Vbary_igrins(datadir = None,  T0 = None, Porb = None, save = False, savedir = None, retres = False, infostring = None):
    
    fitsfiles_H = sorted(glob.glob(datadir + 'SDCH*.fits'))
    fitsfiles_K = sorted(glob.glob(datadir + 'SDCK*.fits'))
    
    ## First parse the wavelength data 
    spdatacube_dd = {}
    
    
    wlfits=fits.open(fitsfiles_H[0])
    wlgridH=wlfits[1].data
    print('H wav', wlgridH.shape)
    wlfits=fits.open(fitsfiles_K[0])
    wlgridK=wlfits[1].data
    print('K wav', wlgridK.shape)
    #concatenating wavelength grid
    wlgrid=np.concatenate([wlgridK, wlgridH])
    
    #### Create a wavelength mask to exclude orders heavily contaminated by tellurics or low instrumental throughput
    ### From Matteo's WASP-18b paper : < 1.44 µm, 1.79-1.95 µm, and > 2.42 µm
    wlgrid_ord_mask = np.ones(wlgrid.shape[0], dtype = bool)
    for io in range(wlgrid_ord_mask.shape[0]):
        minw, maxw = np.min(wlgrid[io,:]), np.max(wlgrid[io,:]) 
        print(io, minw, maxw)
        if minw < 1.45 or (1.79 < minw and maxw < 1.97) or (minw > 2.42 or maxw > 2.42):
            wlgrid_ord_mask[io] = False
    wlgrid_masked = wlgrid[wlgrid_ord_mask,:]
    ##### Alternatively can also select these (from Matteo's wavelength recal code): 
    #     ''' Selecting orders that we want to include (some are heavilty telluric 
    # contaminated so they are removed). Orders are removed generally when median 
    # telluric transmittance is ~<0.7, or when wavelength realingment fails because 
    # of too many / too little tellurics. '''
    # whichorders=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,28,29,30,31,32,33,
    #             34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
    
    #Define lengths and dimensions 
    Nphi=len(fitsfiles_H)
    Ndet, Npix=wlgrid.shape #dimensions of an H+K spectrum (orders x pixels..sorry Ndet = Norders)
    
    ## Define data array 
    data_RAW=np.zeros((Ndet, Nphi,Npix))
    
    ### fitsfiles : list of fitsfiles 
    times_BJD_TDB, Vbary, phases, airmass = np.ones(Nphi), np.ones(Nphi), np.ones(Nphi), np.ones(Nphi)
    
    for it, file in enumerate(fitsfiles_H): ## Assuming the number of files for H and K band is same always 
        hdul = fits.open(fitsfiles_H[it], memmap=False)
        hdul_K = fits.open(fitsfiles_K[it], memmap=False)
        
        image_data_H = hdul[0].data
        image_data_K = hdul_K[0].data
        
        ## Checked for one time instance and the time stamps seem to be always same for H and K bands 
        hdr = hdul[0].header
        
        ## Get the airmass
        airmass[it] = 0.5*(hdr['AMSTART']+hdr['AMEND'])
        
        ## Time stuff 
        radec = hdr['USERRA'] + '\t' + hdr['USERDEC']
        gemini = EarthLocation.of_site(hdr['TELESCOP'])
        t_stamp =  Time(0.5*(hdr['JD-END'] + hdr['JD-OBS']), format = 'jd', location = gemini, scale = 'utc')
        star_coord = SkyCoord([radec], unit=(un.hourangle, un.deg), frame = 'icrs')
        
        ## Calculate the light correction time for the t_stamp 
        timecorrection_delta = t_stamp.light_travel_time(star_coord,'barycentric')
        time_corrected = t_stamp.tdb + timecorrection_delta
        
        ## Calculate BERV
        barycorr = star_coord.radial_velocity_correction(obstime=time_corrected)
        Vbary_ = -barycorr.to(un.km/un.s).value
        
        ## Calculate phase 
        phase = (time_corrected[0].tdb.value-T0) / Porb
        
        ## Populate arrays 
        times_BJD_TDB[it], Vbary[it], phases[it] = time_corrected[0].tdb.value, Vbary_, phase
        data=np.concatenate([image_data_K,image_data_H]) ## concatenating in the same order as the wavelength grid was concatenated 
        data_RAW[:,it,:]=data
        
    data_RAW_nan_removed = np.nan_to_num(data_RAW, nan = -1e30)
    # pdb.set_trace()
        
    #### Mask bad orders 
    data_RAW_masked = data_RAW_nan_removed[wlgrid_ord_mask,:,:]
    
    ## Extract the time of obs : 
    datestr = Time(times_BJD_TDB, format = 'jd')[0].isot.split("T")[0]
    
    spdatacube_dd['spdatacube'] = data_RAW_masked
    file_name = infostring + '_' + datestr + '_spdd.npy'
    spdatacube_dd['file_name'] = file_name
    spdatacube_dd['airmass'] = airmass
    spdatacube_dd['phases'] = phases
    spdatacube_dd['wavsoln'] = wlgrid_masked*1000. ### convert to nm
    spdatacube_dd['time'] = times_BJD_TDB
    spdatacube_dd['bary_RV'] = Vbary
    
    if save:
        np.save(savedir + file_name, spdatacube_dd)
    
    if retres:
        return data_RAW_masked, wlgrid_masked, times_BJD_TDB, Vbary, phases, airmass

############### ############### ############### ############### ############### 
################# ################# ################# ################# 
################ IGRINS Wavelength recalibration ################# 
################# ################# ################# ################# 
############### ############### ############### ############### ############### 
''' Function that applies a linear streching+shift to a spectrum.
curve_fit will fit aa and bb (intercept--shift & slope--strech) to shift and 
stretch each spectrum in a sequence to match a "reference" spectrum. '''
def fitting_func_data_wrap(cs_data):
    def fitting_func_data(x,  aa, bb):
        xx=aa+x*bb
        data_int=splev(xx,cs_data,der=0)	
        return data_int
    return fitting_func_data

def wavelength_recalib_Matteo(spdd_path = None, model_telluric_path = None, savedir = None):
    
    ##### Load the observed data 
    spdd = np.load(spdd_path, allow_pickle = True).item()
    spdd_wave_recalib = copy.deepcopy(spdd)
    
    wl_data = spdd['wavsoln']   # Initial wavelengths; in nm.
    data = spdd['spdatacube']   # Corresponding fluxes; telluric saturated orders have already been removed here. 

    #### Load telluric model data 
    model_tell = fits.getdata(model_telluric_path)
    wl_tell = model_tell['lam']   # Extracting the wavelengths from the FITS file; in nm.
    trans = model_tell['trans']   # Extracting atmospheric transmission from FITS file
    cs_tell = splrep(wl_tell, trans,s=0.0)
    
    ''' Doing some preliminary work on the fluxes to eliminate borders (low-SNR)
    and not-a-number (NaN) / negative values '''
    wl_data = wl_data[:,100:-100]  # Cropping edges
    data = data[:,:,100:-100]   # Cropping edges
    data[np.isnan(data)]=0. #pruning NaN's
    data[data <0.]=0. # purging negative values
    Ndet, Nphi, Npix = data.shape
    
    ''' Preliminary selection of the spectrum with the highest SNR '''
    order = 12   #select order index for "visual" inspection
    avgSpec = np.median(data[order,],axis=1)  # Median across wavelengths
    idxRef = avgSpec == avgSpec.max()

    plt.figure()
    plt.plot(avgSpec)
    plt.show()
    
    data_corrected=np.zeros(data.shape)
    for order in range(Ndet):    # Order loop
        print('- Correcting order {:02}'.format(order))
        wl0 = wl_data[order,:]   # Starting wavelengths
        data_to_fit = data[order,idxRef,:].flatten() # Reference spectrum 
        for phi in range(Nphi):  # phase/time/frame loop
            data_to_correct=data[order,phi,:]    #current spectrum to re-align (raw spectrum)
            # import scipy
            # print(scipy.__version__)
            cs_data_ = splrep(wl0,data_to_correct/data_to_correct.max(),s=0.0) #splining
            # extra_kwargs = {'cs_data':cs_data_}
            popt, pconv=curve_fit(fitting_func_data_wrap(cs_data=cs_data_), wl0, data_to_fit/data_to_fit.max(), 
                                  p0=np.array([0.0,1.0])) #curve fitting

            data_refit=fitting_func_data_wrap(cs_data=cs_data_)(wl0, *popt) #generating "realigned" spectrum
            data_corrected[order,phi,]=data_refit #packing into another fun "Norders x Nphi x Nwavelengths" array/cube
        ''' Diagnostic written during 2024-03-04 meeting '''
        plt.figure(figsize=(20,3))
        plt.plot(wl0,np.mean(data_corrected[order,],axis=0),lw=0.7, label = 'data corrected')
        plt.plot(wl0,interpolate.splev(wl0,cs_tell),lw=0.7, label = 'telluric')
        plt.title('Order {:02}'.format(order))
        plt.legend()
        plt.show()

    spdd_wave_recalib['wavsoln'] = wl_data
    spdd_wave_recalib['spdatacube'] = data_corrected
    spdd_wave_recalib['file_name'] = 'wave_recal_' + spdd['file_name']
    
    np.save(savedir + spdd_wave_recalib['file_name'] , spdd_wave_recalib)
    
############### ############### ############### ############### ############### ############### 
################## CARMENES ############### ############### ############### ############### ###
############### ############### ############### ############### ############### ###############
def carmenes_parse1(ddir = '/Users/vatsalpanwar/source/work/astro/projects/Warwick/v1298tau/data/red/', obj_name = 'V1298 Tau', date = None, 
                         savedir = '/Users/vatsalpanwar/source/work/astro/projects/Warwick/v1298tau/data/parsed/', savestring = 'V1298-Tau-b_CARMENES_2020-01-04'):
    ### Get all the file names first
    nir_scilist_A = glob.glob(ddir + "car*sci*todk*nir_A*")   ### Fibre A NIR files.
    nir_scilist_B = glob.glob(ddir + "car*sci*todk*nir_B*")   ### Fibre B VIS files.
    vis_scilist_A = glob.glob(ddir + "car*sci*todk*vis_A*")   ### Fibre A VIS files.
    vis_scilist_B = glob.glob(ddir + "car*sci*todk*vis_B*")   ### Fibre B VIS files.

    ### Sort them by the time of their creation
    # nir_scilist_A.sort(key=os.path.getmtime)
    # nir_scilist_B.sort(key=os.path.getmtime)
    # vis_scilist_A.sort(key=os.path.getmtime)
    # vis_scilist_B.sort(key=os.path.getmtime)


    #### Select the files that are for th object of interest first before moving forward :
    nir_scilist_A_obj = [k for k in nir_scilist_A if fits.open(k, memmap=False)[0].header['OBJECT'] == obj_name]
    nir_scilist_B_obj = [k for k in nir_scilist_B if fits.open(k, memmap=False)[0].header['OBJECT'] == obj_name]
    vis_scilist_A_obj = [k for k in vis_scilist_A if fits.open(k, memmap=False)[0].header['OBJECT'] == obj_name]
    vis_scilist_B_obj = [k for k in vis_scilist_B if fits.open(k, memmap=False)[0].header['OBJECT'] == obj_name]

    def sort_func(ll):
        return fits.open(ll, memmap=False)[0].header['HIERARCH CARACAL BJD']

    #### Sort them by the time mid point of the exposure instead
    nir_scilist_A_obj.sort(key=sort_func)
    nir_scilist_B_obj.sort(key=sort_func)
    vis_scilist_A_obj.sort(key=sort_func)
    vis_scilist_B_obj.sort(key=sort_func)
    print("Found ", len(nir_scilist_A), " files")
    
    obs_dd = {}
    obs_dd["savestring"] = savestring
    obs_dd["star_name"] = 'V1298 Tau'
    obs_dd["planet_ext"] = 'b'
    obs_dd["planet_name"] = 'V1298 Tau b'

    obs_dd["NIR"] = {}   ### Info for all the data from fibre A.
    obs_dd["VIS"] = {}   ### Info for all the data from fibre B.

    obs_dd["NIR"]["A"] = {}
    obs_dd["NIR"]["B"] = {}
    obs_dd["VIS"]["A"] = {}
    obs_dd["VIS"]["B"] = {}

    obs_dd["NIR"]["A"]["filelist"] = nir_scilist_A_obj
    obs_dd["NIR"]["B"]["filelist"] = nir_scilist_B_obj
    obs_dd["VIS"]["A"]["filelist"] = vis_scilist_A_obj
    obs_dd["VIS"]["B"]["filelist"] = vis_scilist_B_obj


    ### Parse all the header info first and store it as key words inthe obs_dd :

    cardlist = [i[0] for i in list(fits.open(obs_dd["NIR"]["A"]["filelist"][0], memmap=False)[0].header.cards)]
    obs_dd["cardlist"] = cardlist

    print("Getting the header info...")

    for odd in tqdm([obs_dd["NIR"]["A"],obs_dd["NIR"]["B"],obs_dd["VIS"]["A"],obs_dd["VIS"]["B"]]):

        odd["hdrinfo"] = {}
        fitsfilelist = odd["filelist"]

        for crd in cardlist:
            temp_list = []

            for fitsfile in fitsfilelist:

                hdu_exp = fits.open( fitsfile , memmap=False)
                try:
                    temp_list.append(hdu_exp[0].header[crd])
                except KeyError:
                    continue

                hdu_exp.close()

            odd["hdrinfo"][crd] = np.array(temp_list)
    
    
    print("Getting the data and making dictionaries...")

    for odd in tqdm([ obs_dd["NIR"]["A"],obs_dd["NIR"]["B"],obs_dd["VIS"]["A"],obs_dd["VIS"]["B"] ]):

        spec_list = []
        spec_list_norm = []
        
        cont_list = []
        sig_list = []
        wave_list = []
        time_list = []

        fitsfilelist = odd["filelist"]

        for fitsfile in tqdm(fitsfilelist):
            hdu_exp = fits.open(fitsfile, memmap=False)

            time_list.append( hdu_exp[0].header['HIERARCH CARACAL BJD'] )



            spec = hdu_exp[1].data
            cont = hdu_exp[2].data
            sig = hdu_exp[3].data
            wave = hdu_exp[4].data

            hdu_exp.close()

            spec_temp = []
            spec_temp_norm = []
            cont_temp = []
            sig_temp = []
            wave_temp = []

            
            for i in range(wave.shape[0]):
                # spec_temp.extend(spec[i,:])
                # cont_temp.extend(cont[i,:])
                # sig_temp.extend(sig[i,:])
                # wave_temp.extend(wave[i,:])
                spec_temp.append(spec[i,:])
                spec_temp_norm.append(spec[i,:]/np.nanmedian(spec[i,:]))
                cont_temp.append(cont[i,:])
                sig_temp.append(sig[i,:])
                wave_temp.append(wave[i,:])
            # import pdb
            # pdb.set_trace()
            spec_list.append(np.array(spec_temp))
            spec_list_norm.append(np.array(spec_temp_norm))
            cont_list.append(np.array(cont_temp))
            sig_list.append(np.array(sig_temp))
            wave_list.append(np.array(wave_temp))

        odd["spec"] = np.array(spec_list)
        odd["spec_norm"] = np.array(spec_list_norm)
        odd["cont"] = np.array(cont_list)
        odd["sig"] = np.array(sig_list)
        odd["wave"] =  np.array(wave_list)    #### save the wavelength solution for individual exposures as this will be useful for when shifting frame of references.
        odd["time_bjd"] = np.array(time_list)
        
    np.save(savedir + savestring + '_parse1.npy', obs_dd)

def carmenes_parse2(parse1_path = None, savedir = None):
    
    obs_dd = np.load(parse1_path, allow_pickle = True).item()
    savestring = obs_dd['savestring']
    
    # date = '04-01-2020'
    spdatacube_dd_NIR_A = {}
    spdatacube_dd_NIR_B = {}
    spdatacube_dd_VIS_A = {}
    spdatacube_dd_VIS_B = {}

    spdatacube_dd_NIR_A['SPEC'] = np.swapaxes(obs_dd['NIR']['A']['spec'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_NIR_A['NOISE'] = np.swapaxes(obs_dd['NIR']['A']['sig'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_NIR_A['AIRM'] = obs_dd['NIR']['A']['hdrinfo']['AIRMASS']
    spdatacube_dd_NIR_A['PH'] = obs_dd['NIR']['A']['hdrinfo']['MJD-OBS'] # convert it to phase later 
    spdatacube_dd_NIR_A['WLEN'] = 0.1*obs_dd['NIR']['A']['wave'][0,:,:] # shape is (ndet, nwav) , assuming wavelength solution doesn't change throughout the night, in nm
    spdatacube_dd_NIR_A['MJD'] = obs_dd['NIR']['A']['hdrinfo']['MJD-OBS'] # shape is (nspec,)

    spdatacube_dd_NIR_B['SPEC'] = np.swapaxes(obs_dd['NIR']['B']['spec'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_NIR_B['NOISE'] = np.swapaxes(obs_dd['NIR']['B']['sig'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_NIR_B['AIRM'] = obs_dd['NIR']['B']['hdrinfo']['AIRMASS']
    spdatacube_dd_NIR_B['PH'] = obs_dd['NIR']['B']['hdrinfo']['MJD-OBS'] # convert it to phase later 
    spdatacube_dd_NIR_B['WLEN'] = 0.1*obs_dd['NIR']['B']['wave'][0,:,:] # shape is (ndet, nwav) , assuming wavelength solution doesn't change throughout the night 
    spdatacube_dd_NIR_B['MJD'] = obs_dd['NIR']['B']['hdrinfo']['MJD-OBS'] # shape is (nspec,)

    spdatacube_dd_VIS_A['SPEC'] = np.swapaxes(obs_dd['VIS']['A']['spec'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_VIS_A['NOISE'] = np.swapaxes(obs_dd['VIS']['A']['sig'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_VIS_A['AIRM'] = obs_dd['VIS']['A']['hdrinfo']['AIRMASS']
    spdatacube_dd_VIS_A['PH'] = obs_dd['VIS']['A']['hdrinfo']['MJD-OBS'] # convert it to phase later 
    spdatacube_dd_VIS_A['WLEN'] = 0.1*obs_dd['VIS']['A']['wave'][0,:,:] # shape is (ndet, nwav) , assuming wavelength solution doesn't change throughout the night 
    spdatacube_dd_VIS_A['MJD'] = obs_dd['VIS']['A']['hdrinfo']['MJD-OBS'] # shape is (nspec,)

    spdatacube_dd_VIS_B['SPEC'] = np.swapaxes(obs_dd['VIS']['B']['spec'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_VIS_B['NOISE'] = np.swapaxes(obs_dd['VIS']['B']['sig'], 0, 1)# shape should be (ndet, nspec, nwav)
    spdatacube_dd_VIS_B['AIRM'] = obs_dd['VIS']['B']['hdrinfo']['AIRMASS']
    spdatacube_dd_VIS_B['PH'] = obs_dd['VIS']['B']['hdrinfo']['MJD-OBS'] # convert it to phase later 
    spdatacube_dd_VIS_B['WLEN'] = 0.1*obs_dd['VIS']['B']['wave'][0,:,:] # shape is (ndet, nwav) , assuming wavelength solution doesn't change throughout the night 
    spdatacube_dd_VIS_B['MJD'] = obs_dd['VIS']['B']['hdrinfo']['MJD-OBS'] # shape is (nspec,)

    spdatacube_dd = {}
    
    spdatacube_dd['RA'] = obs_dd['NIR']['A']['hdrinfo']['RA']
    spdatacube_dd['DEC'] = obs_dd['NIR']['A']['hdrinfo']['DEC']
    
    spdatacube_dd["savestring"] = savestring 
    spdatacube_dd['NIR'] = {}
    spdatacube_dd['VIS'] = {}

    spdatacube_dd['NIR']['A'] = spdatacube_dd_NIR_A
    spdatacube_dd['NIR']['B'] = spdatacube_dd_NIR_B
    spdatacube_dd['VIS']['A'] = spdatacube_dd_VIS_A
    spdatacube_dd['VIS']['B'] = spdatacube_dd_VIS_B

    np.save(savedir + savestring + '_parse2.npy', spdatacube_dd)

def carmenes_parse3(parse2_path = None, savedir = None, 
                    bandpass = 'NIR', fibre = 'A', T0 = None, Porb = None, radec = '04 05 19.5909996648 +20 09 25.563233736'):
    
    spdd_load = np.load(parse2_path, allow_pickle = True).item()
    
    spdd = spdd_load[bandpass][fibre]
    
    ##### Compute phaes and BERVs
    Nphi = len(spdd['MJD'])
    times_BJD_TDB, Vbary, phases = np.ones(Nphi), np.ones(Nphi), np.ones(Nphi)
    
    for it in range(Nphi):
        if radec is None:
            radec = str(spdd_load['RA'][0]) + ' ' + str(spdd_load['DEC'][0])
        else:
            radec = radec
        caha = EarthLocation.from_geodetic(lat = 37.220953 * un.deg, 
                                                      lon = -2.546778 * un.deg, 
                                                      height=2158 * un.m)
        t_stamp =  Time(spdd['MJD'][it]+2400000.5, 
                        format = 'jd', location = caha, scale = 'utc')
        # import pdb
        # pdb.set_trace()
        star_coord = SkyCoord([radec], unit=(un.hourangle, un.deg), frame = 'icrs')
        
        ## Calculate the light correction time for the t_stamp 
        timecorrection_delta = t_stamp.light_travel_time(star_coord,'barycentric')
        time_corrected = t_stamp.tdb + timecorrection_delta
        
        ## Calculate BERV
        barycorr = star_coord.radial_velocity_correction(obstime=time_corrected)
        Vbary_ = -barycorr.to(un.km/un.s).value
        
        ## Calculate phase 
        # import pdb
        # pdb.set_trace()
        phase = (time_corrected[0].tdb.value-T0) / Porb
        # import pdb
        # pdb.set_trace()
        times_BJD_TDB[it], Vbary[it], phases[it] = time_corrected[0].tdb.value, Vbary_, phase

    spdatacube_dd = {}
    spdatacube_dd['spdatacube'] = spdd['SPEC']
    spdatacube_dd['noisecube'] = spdd['NOISE']
    spdatacube_dd['file_name'] = spdd_load['savestring'] + '_' + bandpass
    spdatacube_dd['airmass'] = spdd['AIRM']
    spdatacube_dd['phases'] = phases
    spdatacube_dd['wavsoln'] = spdd['WLEN']
    spdatacube_dd['time'] = times_BJD_TDB
    spdatacube_dd['bary_RV'] = Vbary
    
    np.save(savedir + spdd_load['savestring'] + '_' + bandpass + '_parse3.npy', spdatacube_dd)

    
    