'''Module for global fitting of (time-dependent) diffusion NMR data.
This module provides the following functions:

GlobalDiff(data)
    Global nonlinear fitting of the S-T equation for an arbitrary number of gradients and chemical shift environments. 
    Returns an lmfit parameters object containing fitted I0 intensities for all peaks, as well as a single globally fitted D value. 
MovingDiff(data,slicelength=10)
    Moving average fit for diffusion, using GlobalDiff to obtain a D value for each time point. S
    Takes as input a pandas dataframe with the first column containing B values, and subsequent columns containing integrals for the peaks of interest. Slicelength sets the number of experiments used for each D(time) point.
    Returns (Dpoints, I0points): two numpy arrays.
MovingDiff_csv(fname,slicelength=10)
    A wrapper for MovingDiff to act on a similarly formatted .csv file.
SeparateMovingDiffusion(data,slicelength=10)
    Generates a pair of pandas dataframes [D,I] containing calculation time-dependent diffusion coefficients and unattenuated integrals.
    Acts on a pandas dataframe.
SeparateMovingDiffusion_csv(fname,slicelength=10)
    Generates a pair of pandas dataframes [D,I] containing calculation time-dependent diffusion coefficients and unattenuated integrals.
    Acts on a .csv file with the first column containing B-values, and each subsequent column containing the corresponding integrals for a particular chemical shift.
MeOHTemp(dDelta)
    Calculates temperature from methanol OH-CH3 chemical shift separation (in ppm)
MeOHDiff(dDelta)
    Calculates expected diffusion coefficient from methanol OH-CH3 chemical shift separation (in ppm)
'''

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

    
def MovingDiff_csv(fname,slicelength=10):
    '''A simple wrapper of  MovingDiff() to act on .csv files'''
    import pandas as pd
    return MovingDiff(pd.read_csv(fname),slicelength)

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
    (This work)'''
    import numpy as np
    return 5.124e-7 * np.exp((-1601)/(MeOHTemp(dDelta)))
