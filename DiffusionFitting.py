'''Module for global fitting of diffusion NMR data.
This module provides two functions:
GlobalDiff(data)	
	Global fitting for the diffusion coefficient of a single chemical species, obtained from integrals over an arbitrary number of peaks.
	Takes as input a pandas dataframe with the first column containing B values, and subsequent columns containing integrals for the peaks of interest.
	Returns a scipy.optimize Parameters object. Parameters can be viewed with eg GlobalDiff(data).params
MovingDiff(data,slicelength=10)
	Moving average fit for diffusion, using GlobalDiff to obtain a D value for each time point. S
	Takes as input a pandas dataframe with the first column containing B values, and subsequent columns containing integrals for the peaks of interest. Slicelength sets the number of experiments used for each D(time) point.
	Returns (Dpoints, I0points): two numpy arrays.
 '''

def GlobalDiff(data):
    '''Function to globally fit a single diffusion coefficient to data from a list of peaks
    Input: a pandas dataframe consisting of:
    [B, I_0, I_1, I_2 ... I_n]
    where B is the list of B-parameters for all experiments, and I_0 through I_n are the measured intensities.
    The function returns a single scipy.optimize Parameters object.  
    '''
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
    for iy, y in enumerate(dataT):
        fit_params.add('D_%i' % (iy+1), value = 1e-8, min = 0, max = 1e-5)
        fit_params.add('I0_%i' % (iy+1), value = 100, min = 0, max = 1e5)
    for iy in range(2,len(dataT)+1):
        fit_params['D_%i' % iy].expr='D_1'
    return minimize(objective,fit_params,args=(B,data))

def MovingDiff(data,slicelength=10):
    '''Fitting for time-dependent diffusion + concentration data.
    Input: pandas dataframe formatted as
    B_0 I0_0 I1_0 ... In_0
    B_1 I0_1 I1_1 ... In_1
    ...
    B_m I0_m I1_m ... In_m
      
    '''
    npoints = data.shape[0]-slicelength-1
    Dpoints = np.zeros((npoints))
    I0points = np.zeros((npoints,data.shape[1]-1))
    I0range = range(I0points.shape[1])
    for i in range(npoints-1):
        params = GlobalDiff(data.iloc[i:i+slicelength])
        Dpoints[i] = params.params['D_1'].value
        for j in I0range:
            I0points[i,j] = params.params['I0_%i' % (j+1)].value
    return Dpoints,I0points