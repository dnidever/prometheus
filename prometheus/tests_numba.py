import os
import numpy as np
from prometheus import utils_numba as utils, models_numba as mnb, getpsf_numba as gnm


def tests():
    utils_tests()
    models_tests()
    getpsf_tests()

def utils_tests():

    ###############
    # UTILS
    ###############

    sm = utils.nansum(np.random.rand(100))
    sm = utils.nansum(np.random.rand(10,10))
    arr = np.random.rand(100)
    arr[10:20] = np.nan
    sm = utils.nansum(arr)
    print('utils_numba.nansum() okay')

    sm = utils.sum(np.random.rand(100))
    sm = utils.sum(np.random.rand(10,10))
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    sm = utils.sum(arr,ignore_nan=True)
    print('utils_numba.sum() okay')

    sm = utils.sum2d(np.random.rand(10,10),axis=0)
    sm = utils.sum2d(np.random.rand(10,10),axis=1)
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    sm = utils.sum2d(arr,axis=0,ignore_nan=True)
    print('utils_numba.sum2d() okay')

    sm = utils.sum3d(np.random.rand(10,10,20),axis=0)
    sm = utils.sum3d(np.random.rand(10,10,20),axis=1)
    sm = utils.sum3d(np.random.rand(10,10,20),axis=2)
    arr = np.random.rand(10,10,20)
    arr[0,1,2] = np.nan
    sm = utils.sum3d(arr,axis=0,ignore_nan=True)
    print('utils_numba.sum3d() okay')


    mn = utils.nanmean(np.random.rand(100))
    mn = utils.nanmean(np.random.rand(10,10))
    arr = np.random.rand(100)
    arr[10:20] = np.nan
    mn = utils.nanmean(arr)
    print('utils_numba.nanmean() okay')

    mn = utils.mean(np.random.rand(100))
    mn = utils.mean(np.random.rand(10,10))
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    mn = utils.mean(arr,ignore_nan=True)
    print('utils_numba.mean() okay')

    mn = utils.mean2d(np.random.rand(10,10),axis=0)
    mn = utils.mean2d(np.random.rand(10,10),axis=1)
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    mn = utils.mean2d(arr,axis=0,ignore_nan=True)
    print('utils_numba.mean2d() okay')

    mn = utils.mean3d(np.random.rand(10,10,20),axis=0)
    mn = utils.mean3d(np.random.rand(10,10,20),axis=1)
    mn = utils.mean3d(np.random.rand(10,10,20),axis=2)
    arr = np.random.rand(10,10,20)
    arr[0,1,2] = np.nan
    mn = utils.mean3d(arr,axis=0,ignore_nan=True)
    print('utils_numba.mean3d() okay')


    med = utils.nanmedian(np.random.rand(100))
    med = utils.nanmedian(np.random.rand(10,10))
    arr = np.random.rand(100)
    arr[10:20] = np.nan
    med = utils.nanmedian(arr)
    print('utils_numba.nanmedian() okay')

    med = utils.median(np.random.rand(100))
    med = utils.median(np.random.rand(10,10))
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    med = utils.median(arr,ignore_nan=True)
    print('utils_numba.median() okay')

    med = utils.median2d(np.random.rand(10,10),axis=0)
    med = utils.median2d(np.random.rand(10,10),axis=1)
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    med = utils.median2d(arr,axis=0,ignore_nan=True)
    print('utils_numba.median2d() okay')

    med = utils.median3d(np.random.rand(10,10,20),axis=0)
    med = utils.median3d(np.random.rand(10,10,20),axis=1)
    med = utils.median3d(np.random.rand(10,10,20),axis=2)
    arr = np.random.rand(10,10,20)
    arr[0,1,2] = np.nan
    med = utils.median3d(arr,axis=0,ignore_nan=True)
    print('utils_numba.median3d() okay')
    
    
    sig = utils.mad(np.random.rand(100))
    sig = utils.mad(np.random.rand(10,10))
    arr = np.random.rand(10,10,20)
    arr[0,1,2] = np.nan
    sig = utils.mad(arr,ignore_nan=True)
    print('utils_numba.mad() okay')

    sig = utils.mad2d(np.random.rand(10,10),axis=0)
    sig = utils.mad2d(np.random.rand(10,10),axis=1)
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    sig = utils.mad2d(arr,axis=0,ignore_nan=True)
    print('utils_numba.mad2d() okay')

    sig = utils.mad3d(np.random.rand(10,10,20),axis=0)
    sig = utils.mad3d(np.random.rand(10,10,20),axis=1)
    sig = utils.mad3d(np.random.rand(10,10,20),axis=2)
    arr = np.random.rand(10,10,20)
    arr[0,1,2] = np.nan
    sig = utils.mad3d(arr,axis=0,ignore_nan=True)
    print('utils_numba.mad3d() okay')

    x = np.arange(1,4,1.0)
    y = 1.0+0.5*x-0.33*x**2
    out = utils.quadratic_bisector(x,y)
    print('utils_numba.quadratic_bisector() okay')

    x,y = utils.meshgrid(np.arange(10),np.arange(20))
    print('utils_numba.meshgrid() okay')
    
    out = utils.aclip(np.random.rand(100),-0.5,0.5)
    print('utils_numba.aclip() okay')
    
    out = utils.clip(-0.2,-0.5,0.5)
    print('utils_numba.clip() okay')
    
    out = utils.gamma(0.5)
    print('utils_numba.gamma() okay')
    
    out = utils.gammaincinv05(1.5)
    print('utils_numba.gammaincinv05() okay')

    data = np.random.rand(10,20)
    out = utils.linearinterp(data,1.4,2.5)
    print('utils_numba.linearinterp() okay')

    data = np.random.rand(10,20)
    x = np.array([1.4,2.5])
    y = np.array([2.6,3.4])
    out = utils.alinearinterp(data,x,y)
    print('utils_numba.alinearinterp() okay')

    arr = np.random.rand(20,20)
    out = utils.inverse(arr)
    print('utils_numba.inverse() okay')

    jac = np.random.rand(200,5)
    resid = np.random.rand(10,20).ravel()
    weight = np.ones((10,20),float).ravel()
    out = utils.qr_jac_solve(jac,resid,weight=weight)
    print('utils_numba.qr_jac_solve() okay')

    jac = np.random.rand(200,5)
    resid = np.random.rand(10,20).ravel()
    weight = np.ones((10,20),float).ravel()
    out = utils.jac_covariance(jac,resid,weight)
    print('utils_numba.jac_covariance() okay')

    pars = np.array([1.0,2.0,3.0])
    bounds = np.zeros((3,2),float)
    bounds[:,0] = [0.0,0.0,0.0]
    bounds[:,1] = [4.0,2.0,3.5]
    out = utils.checkbounds(pars,bounds)
    print('utils_numba.checkbounds() okay')

    pars = np.array([1.0,2.0,3.0])
    bounds = np.zeros((3,2),float)
    bounds[:,0] = [0.0,0.0,0.0]
    bounds[:,1] = [4.0,2.0,3.5]
    out = utils.limbounds(pars,bounds)
    print('utils_numba.limbounds() okay')

    steps = np.array([0.5,0.1,0.2])
    maxsteps = np.array([0.6,0.2,0.1])
    out = utils.limsteps(steps,maxsteps)
    print('utils_numba.limsteps() okay')

    pars = np.array([1.0,2.0,3.0])
    bounds = np.zeros((3,2),float)
    bounds[:,0] = [0.0,0.0,0.0]
    bounds[:,1] = [4.0,2.0,3.5]
    steps = np.array([0.5,0.1,0.2])
    maxsteps = np.array([0.6,0.2,0.1])
    out = utils.newpars(pars,steps,bounds=bounds,maxsteps=maxsteps)
    print('utils_numba.newpars() okay')

    xdata = np.zeros((10,2),float)
    xdata[:,0] = np.arange(10)
    xdata[:,1] = np.arange(10)
    pars = np.array([1.0,2.0,3.0,4.0])
    out = utils.poly2d(xdata,pars)
    print('utils_numba.poly2d() okay')
    
    out = utils.jacpoly2d(xdata,pars)
    print('utils_numba.jacpoly2d() okay')

    x,y = np.meshgrid(np.arange(10),np.arange(10))
    xdata = np.zeros((100,2),float)
    xdata[:,0] = x.ravel()
    xdata[:,1] = y.ravel()
    pars = np.array([1.0,2.0,3.0,4.0])
    data = utils.poly2d(xdata,pars)
    error = data*0+1
    out = utils.poly2dfit(xdata[:,0],xdata[:,1],data,error,maxiter=2,minpercdiff=0.5,verbose=False)
    print('utils_numba.poly2dfit() okay')
    

def models_tests():
    
    ###############
    # MODELS
    ###############
    
    # run individual model functions and check that they work without crashing

    out = mnb.gaussian2d()
    print('models_numba.gaussian2d() okay')
    
def getpsf_tests():
    
    ###############
    # GETPSF
    ###############

    out = gnb.starbbox()
    print('getpsf_numba.starbbox() okay')
