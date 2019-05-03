def GlobalT1(data):
    '''
	Input: a pandas dataframe consisting of:
    
    d_0 I0_0 I1_0 ... In_0
    d_1 I0_1 I1_1 ... In_1
    ...
    d_m I0_m I1_m ... In_m
    where d is the list of delays for all experiments, and each column In_ contains the integrals measured for a single peak.
    The function returns a single lmfit Parameters object.  
    '''
    import numpy as np
    import pandas as pd
    from lmfit import minimize, Parameters, report_fit

    def Invrec(d,I0,T1,Ii):
        I0,T1,d,Ii = np.asarray(I0), np.asarray(T1), np.asarray(d),np.asarray(Ii)
        return I0 - 2*Ii*np.exp(-d/T1)
    def InvRec_dataset(d,params,i):
        I0 = params['I0_%i' % (i+1)].value
        T1 = params['T1_%i' % (i+1)].value
        Ii = params['Ii_%i' % (i+1)].value
        return Invrec(d,I0,T1,Ii)
    def objective(params,d,data):
        dataT = np.array(data.T[1:])
        ndata, nx = dataT.shape
        resid = 0.0*dataT[:]
        #Residual per data set:
        for i in range(ndata):
            resid[i,:] = dataT[i,:] - InvRec_dataset(d,params,i)
        #Flatten to a 1d array:
        return resid.flatten()
    d = data.iloc[:,0]
    dataT = np.array(data.T[1:])
    fit_params = Parameters()
    for iy, y in enumerate(dataT):
        fit_params.add('T1_%i' % (iy+1), value = 1.5, min = 1e-3, max = 100)
        fit_params.add('I0_%i' % (iy+1), value = 1e5, min = 1, max = 1e8)
        fit_params.add('Ii_%i' % (iy+1), value = 1e5, min = 1, max = 1e8)
    #for iy in range(2,len(dataT)+1):
        #fit_params['p1_%i' % iy].expr='p1_1'
    return minimize(objective,fit_params,args=(d,data))

def MovingT1(data,slicelength=10):

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

    cols = data.columns[1:]
    T1 = pd.DataFrame(columns =cols)
    T1err = pd.DataFrame(columns =cols)
    I0 = pd.DataFrame(columns =cols)
    I0err = pd.DataFrame(columns =cols)
    Ii = pd.DataFrame(columns=cols)
    Iierr = pd.DataFrame(columns=cols)

    I0points,I0errpoints,T1points,T1errpoints,Iipoints,Iierrpoints = np.zeros(len(cols)),np.zeros(len(cols)),np.zeros(len(cols)),np.zeros(len(cols)),np.zeros(len(cols)),np.zeros(len(cols))
    for i in tqdm(range(0,npoints),desc='Progress:',position=1,leave=False):
        params = GlobalT1(data.iloc[i:i+slicelength])
        #Ii = pd.concat([Ii,pd.DataFrame([params.params['Ii_1'].value],index=[i],columns=['p1'])])
        #Iierr = pd.concat([p1err,pd.DataFrame([params.params['p1_1'].stderr],index=[i],columns=['p1 error'])])       
        for j in range(0,len(cols)):
            I0points[j] = params.params['I0_%i' % (j+1)].value
            I0errpoints[j] = params.params['I0_%i' % (j+1)].stderr
            Iipoints[j] = params.params['Ii_%i' % (j+1)].value
            Iierrpoints[j] = params.params['Ii_%i' % (j+1)].stderr
            T1points[j] = params.params['T1_%i' % (j+1)].value
            T1errpoints[j] = params.params['T1_%i' % (j+1)].stderr
        I0 = pd.concat([I0,pd.DataFrame([I0points],index=[i],columns=cols)])
        I0err = pd.concat([I0err,pd.DataFrame([I0errpoints],index=[i],columns=cols)])
        Ii = pd.concat([Ii,pd.DataFrame([Iipoints],index=[i],columns=cols)])
        Iierr = pd.concat([Iierr,pd.DataFrame([Iierrpoints],index=[i],columns=cols)])
        T1 = pd.concat([T1,pd.DataFrame([T1points],index=[i],columns=cols)])
        T1err = pd.concat([T1err,pd.DataFrame([T1errpoints],index=[i],columns=cols)])
    return T1,I0,T1err,I0err,Ii,Iierr
    
def MovingT1_csv(fname,slicelength=10):
    '''A simple wrapper of  MovingDiff() to act on .csv files'''
    import pandas as pd
    return MovingT1(pd.read_csv(fname),slicelength)

def SeparateMovingT1(data,slicelength=10):
    '''Moving average InvRec processing for multiple separate chemical species. 
    Acts on a pandas dataframe containing a list of d-values in the first column, and corresponding peak integrals in subsequent columns.
    Returns a pair of pandas dataframes [T1,I] containing the calculated diffusion coefficients 
    and concentrations for each peak present in the input array. '''
    import pandas as pd
    from tqdm import tqdm_notebook as tqdm
    #D = pd.DataFrame()
    T1,I,T1err,Ierr,Ii,Iierr = MovingT1(data.iloc[:,[0,1]],slicelength=slicelength)
    T1 = pd.DataFrame(T1)
    T1.columns = [data.columns[1]]
    I = pd.DataFrame(I)
    I.columns = [data.columns[1]]
    T1err = pd.DataFrame(T1err)
    T1err.columns = [data.columns[1]]
    Ierr = pd.DataFrame(Ierr)
    Ierr.columns = [data.columns[1]]
    Ii = pd.DataFrame(Ii)
    Iierr = pd.DataFrame(Iierr)
    Ii.columns = [data.columns[1]]
    Iierr.columns = [data.columns[1]]
    for i in tqdm(range(2,data.shape[1]),desc='Overall progress',position=2):
        tT1,ti,tT1err,tierr,tIi,tIierr = MovingT1(data.iloc[:,[0,i]],slicelength=slicelength)
        tT1 = pd.DataFrame(tT1)
        ti = pd.DataFrame(ti)
        tT1err = pd.DataFrame(tT1err)
        tierr = pd.DataFrame(tierr)
        tIi = pd.DataFrame(tIi)
        tIierr = pd.DataFrame(tIierr)
        
        tT1.columns = [data.columns[i]]
        ti.columns = [data.columns[i]]
        tT1err.columns = [data.columns[i]]
        tierr.columns = [data.columns[i]]
        tIi.columns = [data.columns[i]]
        tIierr.columns = [data.columns[i]]
        
        T1 = pd.concat([T1,tT1],axis=1)
        I = pd.concat([I,ti],axis=1)
        T1err = pd.concat([T1err,tT1err],axis=1)
        Ierr = pd.concat([Ierr,tierr],axis=1)
        Ii = pd.concat([Ii,tIi],axis=1)
        Iierr = pd.concat([Iierr,tIierr],axis=1)
    return T1, I, T1err, Ierr,Ii,Iierr

def GetTimeArray(dlist,fixed,nscans):
    '''Generate an array of time indices for moving invrec experiments (where each delay on the delay list corresponds to a different slice time)
    Usage: GetTimeArray(vdlist,fixed,nscans)
    where: 
    vdlist      List of delays, 
    fixed       Additional time taken per experiment (acquisition time + recycle delay)
    nscans      Total number of scans per time increment (ns + ds)'''
    tints = nscans*(fixed+dlist)
    tvals = np.zeros(len(tints))
    for i in range(0,len(tints)):
        tvals[i] = sum(tints[0:i])
    return tvals