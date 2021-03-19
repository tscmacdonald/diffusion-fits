'''Module for global fitting of (time-dependent) diffusion NMR data.
This module provides the following functions:

GlobalDiff(data)
    Global nonlinear fitting of the S-T equation for an arbitrary number of gradients and chemical shift environments. 
    Returns an lmfit parameters object containing fitted I0 intensities for all peaks, as well as a single globally fitted D value. 
DiffPerSection(data,sectionlength=16,slicetime=1/8):
    Globally fitted time-resolved diffusion NMR processing. 
    This function takes integrals and b-values for a number of peaks and returns a *single* time-resolved diffusion dataframe fitted globally to all input data.
    Distinct I0 data will be returned for each peak.
    To be used for peaks corresponding to the same molecule.
SeparateDiffPerSection(data,sectionlength=16,slicetime=1/8):
    Diffusion processing for multiple separate chemical species. Fits individual sections of data, of length 'sectionlength'.
    Returns time-resolved diffusion and intensity data for *each* integrated peak provided.  
SeparateDiffPerSection_csv(fname,sectionlength=16,slicetime=1/8)
    A wrapper for SeparateDiffPerSection to act on a similarly formatted .csv file.
MovingDiff(data,slicelength=10)
    Global moving average fit for diffusion, using GlobalDiff to obtain a D value for each time point. S
    Takes as input a pandas dataframe with the first column containing B values, and subsequent columns containing integrals for the peaks of interest. 
    Slicelength sets the number of experiments used for each D(time) point. Slicetime sets the time (in minutes) taken to acquire each gradient slice.
    Returns (Dpoints, I0points,Derr,Ierr): four pandas dataframes, each with indices corresponding to time. I0points and Ierr contain a column for each fitted NMR peak, while Dpoint and Der contain a single column with globally fitted diffusion data.
    For systems involving multiple chemical species, use SeparateMovingDiffusion to obtain individual (non-globally fitted) diffusion coefficients.
    Dpoints, I0points, Derr, and Ierr contain diffusion points, extrapolated intensities, and respective errors for each from fitting.
MovingDiff_csv(fname,slicelength=10,slicetime=2/3)
    A wrapper for MovingDiff to act on a similarly formatted .csv file.
SeparateMovingDiffusion(data,slicelength=10,slicetime=2/3)
    Returns (Dpoints, I0points,Derr,Ierr): four pandas dataframes, each with indexes corresponding to time and a column for each peak.
    Dpoints, I0points, Derr, and Ierr contain diffusion points, extrapolated intensities, and respective errors for each from fitting.
SeparateMovingDiffusion_csv(fname,slicelength=10,slicetime=2/3)
    Generates a pair of pandas dataframes [D,I] containing calculation time-dependent diffusion coefficients and unattenuated integrals.
    Acts on a .csv file with the first column containing B-values, and each subsequent column containing the corresponding integrals for a particular chemical shift.
MeOHTemp(dDelta)
    Calculates temperature from methanol OH-CH3 chemical shift separation (in ppm)
MeOHDiff(dDelta)
    Calculates expected diffusion coefficient from methanol OH-CH3 chemical shift separation (in ppm)
'''

import numpy as np
import pandas as pd

def GlobalDiff(data):
    '''Function to globally fit a single diffusion coefficient to data from a list of peaks
    Input: a pandas dataframe consisting of:
    
    B_0 I0_0 I1_0 ... In_0
    B_1 I0_1 I1_1 ... In_1
    ...
    B_m I0_m I1_m ... In_m
    where B is the list of B-parameters for all experiments, and each column In_ contains the integrals measured for a single peak.
    The function returns a single lmfit Parameters object.  
    '''
    import numpy as np
    import pandas as pd
    from lmfit import minimize, Parameters, report_fit

    def STExp(B,I0,D):
        I0,B,D = np.asarray(I0), np.asarray(B), np.asarray(D)
        return I0*np.exp(-B*D)
    def STExp_dataset(B,params,i):
        I0 = params['I0_%i' % (i+1)].value
        D = params['D_%i' % (i+1)].value
        return STExp(B,I0,D)
    def objective(params,B,data):
        dataT = np.array(data.T[1:])
        ndata, nx = dataT.shape
        resid = 0.0*dataT[:]
        #Residual per data set:
        for i in range(ndata):
            resid[i,:] = dataT[i,:] - STExp_dataset(B,params,i)
        #Flatten to a 1d array:
        return resid.flatten()
    B = data.iloc[:,0]
    dataT = np.array(data.T[1:])
    fit_params = Parameters()
    I0guesses = data.max()[1:]
    for iy, y in enumerate(dataT):
        fit_params.add('D_%i' % (iy+1), value = 1e-9, min = 1e-12, max = 1e-8)
        fit_params.add('I0_%i' % (iy+1), value = I0guesses[iy], min = 1, max = 100*I0guesses[iy]) #Give each I0 parameter a unique guessed I0
    for iy in range(2,len(dataT)+1):
        fit_params['D_%i' % iy].expr='D_1'
    return minimize(objective,fit_params,args=(B,data))


def DiffPerSection(data,sectionlength=16,slicetime=1/8):
    '''
    Globally fitted time-resolved diffusion NMR processing. 
    This function takes integrals and b-values for a number of peaks and returns a *single* time-resolved diffusion dataframe fitted globally to all input data.
    Distinct I0 data will be returned for each peak.
    To be used for peaks corresponding to the same molecule.
    
    Parameters
    ----------
    data : DataFrame
        Pandas dataframe containing b-values and peak integrals for each gradient slice of a diffusion experiment.
        eg:
        B_0 I0_0 I1_0 ... In_0
        B_1 I0_1 I1_1 ... In_1
        ...
        B_m I0_m I1_m ... In_m
        where B is the list of B-parameters for all experiments, and each column In_ contains the integrals measured for a single peak. All peaks should correspond to the same molecule.
    sectionlength : int, optional
        Number of consecutive gradient slices over which to apply the Stejskal-Tanner fitting. The default is 16.
    slicetime : float, optional
        Time taken to acquire each gradient slice in minutes. The default is 1/8 (minutes, eg 7.5 s seconds)

    Returns
    -------
    D : DataFrame
        Fitted time-resolved diffusion coefficients. A single D value is globally fitted across all integrated peaks at each timepoint.
        eg:
        t0 D0_0 D1_0 ... Dn_0
        t1 D0_1 D1_1 ... Dn_1
        ...
        tm D0_m D1_m ... Dn_m
        where t is the time points, and each column Dn_ contains the diffusion coefficent calculated for each timepoint.
    I0 : DataFrame
        Fitted time-resolved intensity coefficients. Individual I0 values are fitted for each integrated peak at each timepoint.
        eg:
        t0 I00_0 I01_0 ... I0n_0
        t1 I00_1 I01_1 ... I0n_1
        ...
        tm I00_m I01_m ... I0n_m
        where t is the time points, and each column I0n_ contains the I0 calculated for each timepoint.
    Derr : DataFrame
        Estimated error on D, obtained from the lmfit object calculated by GlobalDiff. A single Derr value is globally fitted across all integrated peaks at each timepoint.
        t0 Derr0_0 Derr1_0 ... Derrn_0
        t1 Derr0_1 Derr1_1 ... Derrn_1
        ...
        tm Derr0_m Derr1_m ... Derrn_m
        where t is the time points, and each column Derrn_ contains the error associated with the diffusion coefficent calculated for each timepoint.
    I0err : DataFrame
        Estimated error on I0, obtained from the lmfit object calculated by GlobalDiff. Individual I0err values are fitted for each integrated peak at each timepoint.
        t0 I0err0_0 I0err1_0 ... I0errn_0
        t1 I0err0_1 I0err1_1 ... I0errn_1
        ...
        tm I0err0_m I0err1_m ... I0errn_m
        where t is the time points, and each column I0errn_ contains the error associated with the I0 value calculated for each timepoint.

    '''
    npoints = int(data.shape[0]/sectionlength) # number of points (should be the number of diffusion experiments run individually)
    ipoints = np.arange(0,npoints) # make array of the number of points
    tpoints = ipoints*slicetime*sectionlength # use array of points to create time index for each point, from sectionlength and slicetime
    col = data.columns[1:] # exclude difflist and consider only integrals
    # create dataframes of the correct size, with the right timepoints to collect the results
    D = pd.DataFrame(index = tpoints,columns = [col[0]])
    Derr = pd.DataFrame(index = tpoints,columns = [col[0]])
    I0 = pd.DataFrame(index = tpoints,columns = col)
    I0err = pd.DataFrame(index = tpoints,columns = col)
    # loop over the total length of the number of points, fitting individual sections of data
    for i in range(npoints):   
        params = GlobalDiff(data.iloc[i*sectionlength:(i+1)*sectionlength])
        # loop over the number of peaks input to produce individual I0 values for each 
        I0slice,I0errslice = [],[]
        for Ival in range(0,len(col)):
            ParamName = 'I0_{}'.format(Ival+1)
            I0slice.append(params.params[ParamName].value)
            I0errslice.append(params.params[ParamName].stderr)        
        # assign globally fitted D and individual I0 values to the correct positions in the pre-made dataframes
        D.loc[tpoints[i],col[0]] = params.params['D_1'].value
        Derr.loc[tpoints[i],col[0]] = params.params['D_1'].stderr
        I0.loc[tpoints[i],col] = I0slice       
        I0err.loc[tpoints[i],col] = I0errslice  
    return D,I0,Derr,I0err 

def SeparateDiffPerSection(data,sectionlength=16,slicetime=1/8):
    '''  
    Diffusion processing for multiple separate chemical species. Fits individual sections of data, of length 'sectionlength'.
    Returns time-resolved diffusion and intensity data for *each* integrated peak provided.  
    
    Parameters
    ----------
    data : DataFrame
        Pandas dataframe containing b-values and peak integrals for each gradient slice of a diffusion experiment.
        eg:
        B_0 I0_0 I1_0 ... In_0
        B_1 I0_1 I1_1 ... In_1
        ...
        B_m I0_m I1_m ... In_m
        where B is the list of B-parameters for all experiments, and each column In_ contains the integrals measured for a single peak.
    sectionlength : int, optional
        Number of consecutive gradient slices over which to apply the Stejskal-Tanner fitting. The default is 16.
    slicetime : float, optional
        Time taken to acquire each gradient slice in minutes. The default is 1/8 (minutes, eg 7.5 s seconds)

    Returns
    -------
    D : DataFrame
        Fitted time-resolved diffusion coefficients. Individual D values are fitted for each integrated peak at each timepoint.
        eg:
        t0 D0_0 D1_0 ... Dn_0
        t1 D0_1 D1_1 ... Dn_1
        ...
        tm D0_m D1_m ... Dn_m
        where t is the time points, and each column Dn_ contains the diffusion coefficent calculated for each timepoint.
    I0 : DataFrame
        Fitted time-resolved intensity coefficients. Individual I0 values are fitted for each integrated peak at each timepoint.
        eg:
        t0 I00_0 I01_0 ... I0n_0
        t1 I00_1 I01_1 ... I0n_1
        ...
        tm I00_m I01_m ... I0n_m
        where t is the time points, and each column I0n_ contains the I0 calculated for each timepoint.
    Derr : DataFrame
        Estimated error on D, obtained from the lmfit object calculated by GlobalDiff. Individual Derr values are fitted for each integrated peak at each timepoint.
        t0 Derr0_0 Derr1_0 ... Derrn_0
        t1 Derr0_1 Derr1_1 ... Derrn_1
        ...
        tm Derr0_m Derr1_m ... Derrn_m
        where t is the time points, and each column Derrn_ contains the error associated with the diffusion coefficent calculated for each timepoint.
    I0err : DataFrame
        Estimated error on I0, obtained from the lmfit object calculated by GlobalDiff. Individual I0err values are fitted for each integrated peak at each timepoint.
        t0 I0err0_0 I0err1_0 ... I0errn_0
        t1 I0err0_1 I0err1_1 ... I0errn_1
        ...
        tm I0err0_m I0err1_m ... I0errn_m
        where t is the time points, and each column I0errn_ contains the error associated with the I0 value calculated for each timepoint.
    '''
    npoints = int(data.shape[0]/sectionlength) # number of points (should be the number of diffusion experiments run individually)
    ipoints = np.arange(0,npoints) # make array of the number of points
    tpoints = ipoints*slicetime*sectionlength # use array of points to create time index for each point, from sectionlength and slicetime
    cols = data.columns[1:] # exclude difflist and consider only integrals to use as y-data in Stejskal-Tanner fitting
    # set the correct sizes for dataframes t0 collect data for all peaks
    D = pd.DataFrame(index = tpoints, columns=cols)
    I = pd.DataFrame(index = tpoints, columns=cols)
    Derr = pd.DataFrame(index = tpoints, columns=cols)
    Ierr = pd.DataFrame(index = tpoints, columns=cols)
    # loop over the number of peaks provided in the dataframe and run the DiffPerSection function for each
    for peak in data.columns[1:]:
        td,ti,tderr,tierr = DiffPerSection(pd.concat([data.iloc[:,0],data[peak]],axis=1),sectionlength=sectionlength,slicetime=slicetime)    
        # assign the fitted D,I,Derr,Ierr values to the correct position in the pre-made dataframe
        D[peak] = td
        I[peak] = ti
        Derr[peak] = tderr
        Ierr[peak] = tierr
    return D, I, Derr, Ierr


def SeparateDiffPerSection_csv(fname,sectionlength=16,slicetime=1/8):
    '''A wrapper of SeparateDiffPerSection to act on .csv files'''
    return  SeparateDiffPerSection(pd.read_csv(fname),sectionlength,slicetime)


def MovingDiff(data,slicelength=10,slicetime=2/3):
    '''Fitting for time-dependent diffusion + concentration data.
    Input: pandas dataframe formatted as
    B_0 I0_0 I1_0 ... In_0
    B_1 I0_1 I1_1 ... In_1
    ...
    B_m I0_m I1_m ... In_m
      
    '''
    import numpy as np
    import pandas as pd
    from tqdm import tqdm_notebook as tqdm
    from lmfit import minimize, Parameters, report_fit

    npoints = data.shape[0]-slicelength
    ipoints = np.arange(0,npoints)
    tpoints = ipoints*slicetime+slicetime*slicelength/2
    
    cols = data.columns[1:]
    D = pd.DataFrame(index = tpoints,columns = [cols[0]])
    Derr = pd.DataFrame(index = tpoints,columns = [cols[0]])
    I0 = pd.DataFrame(index = tpoints,columns = cols)
    I0err = pd.DataFrame(index = tpoints,columns = cols)

    for i in tqdm(range(npoints),desc='Progress:',position=1,leave=False):      
        params = GlobalDiff(data.iloc[i:i+slicelength])
        I0slice,I0errslice = [],[]
        for Ival in range(0,len(cols)):
            ParamName = 'I0_{}'.format(Ival+1)
            #I0.insert(params.params[ParamName].value,index=i,col)
            I0slice.append(params.params[ParamName].value)
            I0errslice.append(params.params[ParamName].stderr)        
        D.loc[tpoints[i],cols[0]] = params.params['D_1'].value
        Derr.loc[tpoints[i],cols[0]] = params.params['D_1'].stderr
        I0.loc[tpoints[i],cols] = I0slice       
        I0err.loc[tpoints[i],cols] = I0errslice  
    return D,I0,Derr,I0err

    
def MovingDiff_csv(fname,slicelength=10,slicetime=2/3):
    '''A simple wrapper of  MovingDiff() to act on .csv files'''
    import pandas as pd
    return MovingDiff(pd.read_csv(fname),slicelength,slicetime)

def SeparateMovingDiffusion(data,slicelength=10,slicetime=2/3):
    '''Moving average diffusion processing for multiple separate chemical species. 
    Acts on a pandas dataframe containing a list of B-values in teh first column, and corresponding peak integrals in subsequent columns.
    Returns a pair of pandas dataframes [D,I] containing the calculated diffusion coefficients 
    and concentrations for each peak present in the input array. '''
    import pandas as pd
    import numpy as np
    from tqdm import tqdm_notebook as tqdm
    npoints = data.shape[0]-slicelength
    ipoints = np.arange(0,npoints)
    tpoints = ipoints*slicetime+slicetime*slicelength/2
    D = pd.DataFrame(index = tpoints,columns=data.columns[1:])
    I = pd.DataFrame(index = tpoints,columns=data.columns[1:])
    Derr = pd.DataFrame(index = tpoints,columns=data.columns[1:])
    Ierr = pd.DataFrame(index = tpoints,columns=data.columns[1:])

    for peak in tqdm(data.columns[1:],desc='Peak-by-peak progress',position=2):
        td,ti,tderr,tierr = MovingDiff(pd.concat([data.iloc[:,0],data[peak]],axis=1),slicelength=slicelength,slicetime=slicetime)    
        D[peak] = td
        I[peak] = ti
        Derr[peak] = tderr
        Ierr[peak] = tierr
    return D, I, Derr, Ierr
def SeparateMovingDiffusion_csv(fname,slicelength=10,slicetime=2/3):
    '''A wrapper of SeparateMovingDiffusion to act on .csv files'''
    import pandas as pd
    return SeparateMovingDiffusion(pd.read_csv(fname),slicelength=slicelength,slicetime=slicetime)

def MeOHTemp(dDelta):
    '''Converts a methanol CH3-OH chemical shift separation (in ppm) to a temperature (in K). 
    See J. Magn. Reson. 1982, 46, 319-321'''
    return 409-36.54*dDelta-21.85*dDelta**2
def MeOHDiff(dDelta):
    '''Calculates the expected self-diffusion coefficient of methanol for a given OH-CH3 peak chemical shift separation.
    See MacDonald et al, ChemPhysChem 2019, 20, 926–930'''
    import numpy as np
    return 5.124e-7 * np.exp((-1601)/(MeOHTemp(dDelta)))
