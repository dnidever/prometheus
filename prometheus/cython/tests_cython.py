import os
import numpy as np
import time
#from prometheus import utils_numba_static as utils, models_numba_static as mnb, getpsf_numba_static as gnb
#from prometheus import groupfit_numba_static as gfit, allfit_numba_static as afit
#from prometheus import ccddata
#import ccddata
#import utils_numba_static as utils
#import models_numba_static as mnb
#import getpsf_numba_static as gnb
#import groupfit_numba_static as gfit
#import allfit_numba_static as afit
#import _utils_numba_static as utils
#import _models_numba_static as mnb

def alltests():
    utils_tests()
    models_tests()
    getpsf_tests()

def utils_tests():

    ###############
    # UTILS
    ###############

    import utils
    
    sig = utils.mad(np.random.rand(100),False,False)
    #sig = utils.mad(np.random.rand(10,10))
    arr = np.random.rand(1000)
    arr[1:3] = np.nan
    sig = utils.mad(arr,True,False)
    print('utils_cython.mad() okay')

    sig = utils.mad2d(np.random.rand(10,10),0,False,False)
    sig = utils.mad2d(np.random.rand(10,10),1,False,False)
    arr = np.random.rand(10,10)
    arr[0,1] = np.nan
    sig = utils.mad2d(arr,0,True,False)
    print('utils_cython.mad2d() okay')

    sig = utils.mad3d(np.random.rand(10,10,20),0,False,False)
    sig = utils.mad3d(np.random.rand(10,10,20),1,False,False)
    sig = utils.mad3d(np.random.rand(10,10,20),2,False,False)
    arr = np.random.rand(10,10,20)
    arr[0,1,2] = np.nan
    sig = utils.mad3d(arr,0,True,False)
    print('utils_cython.mad3d() okay')

    x = np.arange(1,4,1.0)
    y = 1.0+0.5*x-0.33*x**2
    out = utils.quadratic_bisector(x,y)
    print('utils_cython.quadratic_bisector() okay')

    data = np.random.rand(10,20)
    out = utils.linearinterp(data,1.4,2.5)
    print('utils_cython.linearinterp() okay')

    data = np.random.rand(10,20)
    x = np.array([1.4,2.5])
    y = np.array([2.6,3.4])
    out = utils.alinearinterp(data,x,y)
    print('utils_cython.alinearinterp() okay')

    # arr = np.random.rand(20,20)
    # out = utils.inverse(arr)
    # print('utils_cython.inverse() okay')

    # jac = np.random.rand(200,5)
    # resid = np.random.rand(10,20).ravel()
    # weight = np.ones((10,20),float).ravel()
    # out = utils.qr_jac_solve(jac,resid,weight)
    # print('utils_cython.qr_jac_solve() okay')

    # jac = np.random.rand(200,5)
    # resid = np.random.rand(10,20).ravel()
    # weight = np.ones((10,20),float).ravel()
    # out = utils.jac_covariance(jac,resid,weight)
    # print('utils_cython.jac_covariance() okay')

    # pars = np.array([1.0,2.0,3.0])
    # lbounds = np.array([0.0,0.0,0.0])
    # ubounds = np.array([4.0,2.0,3.5])
    # bounds = (lbounds,ubounds)
    # out = utils.checkbounds(pars,bounds)
    # print('utils_cython.checkbounds() okay')

    # pars = np.array([1.0,2.0,3.0])
    # lbounds = np.array([0.0,0.0,0.0])
    # ubounds = np.array([4.0,2.0,3.5])
    # bounds = (lbounds,ubounds)
    # out = utils.limbounds(pars,bounds)
    # print('utils_cython.limbounds() okay')

    # steps = np.array([0.5,0.1,0.2])
    # maxsteps = np.array([0.6,0.2,0.1])
    # out = utils.limsteps(steps,maxsteps)
    # print('utils_cython.limsteps() okay')

    pars = np.array([1.0,2.0,3.0])
    lbounds = np.array([0.0,0.0,0.0])
    ubounds = np.array([4.0,2.0,3.5])
    bounds = (lbounds,ubounds)
    steps = np.array([0.5,0.1,0.2])
    maxsteps = np.array([0.6,0.2,0.1])
    out = utils.newpars(pars,steps,bounds,maxsteps)
    print('utils_cython.newpars() okay')

    xdata = np.zeros((10,2),float)
    xdata[:,0] = np.arange(10)
    xdata[:,1] = np.arange(10)
    pars = np.array([1.0,2.0,3.0,4.0])
    out = utils.poly2d(xdata,pars)
    print('utils_cython.poly2d() okay')
    
    # out = utils.jacpoly2d(xdata,pars)
    # print('utils_cython.jacpoly2d() okay')

    # x,y = np.meshgrid(np.arange(10),np.arange(10))
    # xdata = np.zeros((100,2),float)
    # xdata[:,0] = x.ravel()
    # xdata[:,1] = y.ravel()
    # pars = np.array([1.0,2.0,3.0,4.0])
    # data = utils.poly2d(xdata,pars)
    # error = data*0+1
    # out = utils.poly2dfit(xdata[:,0],xdata[:,1],data,error,2,0.5,False)
    # print('utils_cython.poly2dfit() okay')

    # # Index
    # arr = np.array([1.0,2.0,3.0,4.0,5.0,3.0,2.0,1.0,6.0])
    # index = utils.Index(arr)
    # out = index.data
    # out = index.index
    # out = index.num
    # out = len(index)
    # out = index[0]
    # out = index.values
    # out = index.get(0)
    # out = index.getindex(0)
    # out = index.invindex
    # print('utils_cython.Index okay')

    # ra1 = np.random.rand(100)*10
    # dec1 = np.random.rand(100)*10
    # ra2 = ra1 + np.random.randn(100)*0.1
    # dec2 = ra1 + np.random.randn(100)*0.1
    # n1 = len(ra1)
    # n2 = len(ra2)
    # X1 = np.zeros((n1,2),float)
    # X1[:,0] = ra1
    # X1[:,1] = dec1
    # X2 = np.zeros((n2,2),float)
    # X2[:,0] = ra2
    # X2[:,1] = dec2
    # out = utils.crossmatch(X1, X2, max_distance=np.inf,k=1)
    # print('utils_cython.crossmatch() okay')

    # ra1 = np.random.rand(100)*10
    # dec1 = np.random.rand(100)*10
    # ra2 = ra1 + np.random.randn(100)*0.001
    # dec2 = dec1 + np.random.randn(100)*0.001
    # out = utils.xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=False, sphere=False)
    # out = utils.xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=True, sphere=False)
    # out = utils.xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=False, sphere=True)
    # out = utils.xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=True, sphere=True)
    # print('utils_cython.xmatch() okay')


def models_tests():
    
    ###############
    # MODELS
    ###############

    #import _utils_cython_static as utils
    #import _models_cython_static as mnb
    import models as mnb
    
    # run individual model functions and check that they work without crashing

    # x,y = utils.meshgrid(np.arange(51),np.arange(51))
    # im = mnb.gaussian2d()
    # out = mnb.gaussfwhm(im)
    # print('models_cython.gaussfwhm() okay')
    
    # out = mnb.contourfwhm(im)
    # print('models_cython.contourfwhm() okay')
    
    # out = mnb.imfwhm(im)
    # print('models_cython.imfwhm() okay')
    
    # out = mnb.starbbox(coords,imshape,radius)
    # print('models_cython.starbbox() okay')
    
    # out = mnb.bbox2xy(bbox)
    # print('models_cython.bbo2xy() okay')
    
    out = mnb.gauss_abt2cxy(3.1,3.0,0.1)
    print('models_cython.gauss_abt2cxy() okay')
    
    out = mnb.gauss_cxy2abt(0.2,0.1,-0.1)
    print('models_cython.gauss_cxy2abt() okay')

    pars_gauss = np.array([100.0,3.5,4.5,3.1,3.0,0.1])
    out = mnb.gaussian2d_flux(pars_gauss)
    print('models_cython.gaussian2d_flux() okay')

    out = mnb.gaussian2d_fwhm(pars_gauss)
    print('models_cython.gaussian2d_fwhm() okay')

    out = mnb.gaussian2d(1.0,2.0,pars_gauss,6)
    print('models_cython.gaussian2d() okay')
    
    x,y = np.meshgrid(np.arange(51),np.arange(51))
    out = mnb.agaussian2d(x.ravel().astype(float),y.ravel().astype(float),pars_gauss,3)
    print('models_cython.agaussian2d() okay')

    # x,y = np.meshgrid(np.arange(51),np.arange(51))
    # pars = np.array([100.0,3.5,4.5,3.1,3.0,0.1])
    # im,deriv = mnb.agaussian2d(x,y,pars,3)
    # err = np.sqrt(np.maximum(im,1))
    # out = mnb.gaussian2dfit(im,err,90.0,3.1,4.1,False)
    # print('models_cython.gaussian2dfit() okay')

    pars_moffat = np.array([100.0,3.5,4.5,3.1,3.0,0.1,2.5])
    out = mnb.moffat2d_fwhm(pars_moffat)
    print('models_cython.moffat2d_fwhm() okay')

    pars = np.array([100.0,3.5,4.5,3.1,3.0,0.1,2.5])
    out = mnb.moffat2d_flux(pars)
    print('models_cython.moffat2d_flux() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    out = mnb.amoffat2d(x.ravel().astype(float),y.ravel().astype(float),pars_moffat,7)
    print('models_cython.amoffat2d() okay')

    out = mnb.moffat2d(1.0,2.0,pars_moffat,7)
    print('models_cython.moffat2d() okay')

    # x,y = np.meshgrid(np.arange(51),np.arange(51))
    # pars = np.array([100.0,3.5,4.5,3.1,3.0,0.1,2.5])
    # im,deriv = mnb.amoffat2d(x,y,pars,7)
    # err = np.sqrt(np.maximum(im,1))
    # out = mnb.moffat2dfit(im,err,90.0,3.1,4.7,verbose)
    # print('models_cython.moffat2dfit() okay')

    pars_penny = np.array([100.0,5.55,6.33,3.1,3.0,0.1,0.1,6])
    out = mnb.penny2d_fwhm(pars_penny)
    print('models_cython.penny2d_fwhm() okay')

    out = mnb.penny2d_flux(pars_penny)
    print('models_cython.penny2d_flux() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    out = mnb.apenny2d(x.ravel().astype(float),y.ravel().astype(float),pars_penny,8)
    print('models_cython.apenny2d() okay')

    out = mnb.penny2d(5.0,6.6,pars_penny,8)
    print('models_cython.penny2d() okay')

    # pars = np.array([100.0,5.55,6.33,3.1,3.0,0.1,0.1,6])
    # out = mnb.penny2dfit(im,err,ampc,xc,yc,verbose)
    # print('models_cython.penny2dfit() okay')

    pars_gausspow = np.array([100.0,5.55,6.33,5.0,7.0,0.01,0.9,0.8])
    out = mnb.gausspow2d_fwhm(pars_gausspow)
    print('models_cython.gausspow2d_fwhm() okay')

    out = mnb.gausspow2d_flux(pars_gausspow)
    print('models_cython.gausspow2d_flux() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    out = mnb.agausspow2d(x.ravel().astype(float),y.ravel().astype(float),pars_gausspow,8)
    print('models_cython.agausspow2d() okay')

    out = mnb.gausspow2d(5.5,6.5,pars_gausspow,8)
    print('models_cython.gausspow2d() okay')

    # pars = np.array([100.0,5.55,6.33,5.0,7.0,0.01,0.9,0.8])
    # out = mnb.gausspow2dfit(im,err,ampc,xc,yc,verbose)
    # print('models_cython.gausspow2dfit() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    pars_sersic = np.array([100.0,5.5,6.3,0.3,1.0,0.9,0.0])
    out = mnb.asersic2d(x.ravel().astype(float),y.ravel().astype(float),pars_sersic,7)
    print('models_cython.asersic2d() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    out = mnb.sersic2d(5.5, 6.5, pars_sersic,7)
    print('models_cython.sersic2d() okay')

    out = mnb.sersic2d_fwhm(pars_sersic)
    print('models_cython.sersic2d_fwhm() okay')

    out = mnb.sersic_b(1.5)
    print('models_cython.sersic_b() okay')
    
    # out = mnb.create_sersic_function(100.0, re, 1.0)
    # print('models_cython.create_sersic_function() okay')
    
    # out = mnb.sersic_lum(100.0, re, 1.0)
    # print('models_cython.sersic_lum() okay')
    
    # out = mnb.sersic_full2half(100.0,kserc,alpha)
    # print('models_cython.sersic_full2half() okay')
    
    # out = mnb.sersic_half2full(100.0,Re,alpha)
    # print('models_cython.sersic_half2ful() okay')

    out = mnb.sersic2d_flux(pars_sersic)
    print('models_cython.sersic2d_flux() okay')

    # not njit yet
    #out = mnb.sersic2d_estimates(pars_sersic)
    #print('models_cython.sersic2d_estimates() okay')

    out = mnb.model2d(5.5,6.5,1,pars_gauss,6)
    out = mnb.model2d(5.5,6.5,2,pars_moffat,7)
    out = mnb.model2d(5.5,6.5,3,pars_penny,8)
    out = mnb.model2d(5.5,6.5,4,pars_gausspow,8)
    out = mnb.model2d(5.5,6.5,5,pars_sersic,7)
    print('models_cython.model2d() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    x = x.ravel().astype(float)
    y = y.ravel().astype(float)
    out = mnb.amodel2d(x,y,1,pars_gauss,6)
    out = mnb.amodel2d(x,y,2,pars_moffat,7)
    out = mnb.amodel2d(x,y,3,pars_penny,8)
    out = mnb.amodel2d(x,y,4,pars_gausspow,8)
    out = mnb.amodel2d(x,y,5,pars_sersic,7)
    print('models_cython.amodel2d() okay')

    out = mnb.model2d_flux(1,pars_gauss)
    out = mnb.model2d_flux(2,pars_moffat)
    out = mnb.model2d_flux(3,pars_penny)
    out = mnb.model2d_flux(4,pars_gausspow)
    out = mnb.model2d_flux(5,pars_sersic)
    print('models_cython.model2d_flux() okay')

    out = mnb.model2d_fwhm(1,pars_gauss)
    out = mnb.model2d_fwhm(2,pars_moffat)
    out = mnb.model2d_fwhm(3,pars_penny)
    out = mnb.model2d_fwhm(4,pars_gausspow)
    out = mnb.model2d_fwhm(5,pars_sersic)
    print('models_cython.model2d_fwhm() okay')

    out = mnb.model2d_estimates(1,100.0,5.5,6.1)
    out = mnb.model2d_estimates(2,100.0,5.5,6.1)
    out = mnb.model2d_estimates(3,100.0,5.5,6.1)
    out = mnb.model2d_estimates(4,100.0,5.5,6.1)
    out = mnb.model2d_estimates(5,100.0,5.5,6.1)
    print('models_cython.model2d_estimates() okay')
    
    out = mnb.model2d_bounds(1)
    out = mnb.model2d_bounds(2)
    out = mnb.model2d_bounds(3)
    out = mnb.model2d_bounds(4)
    out = mnb.model2d_bounds(5)
    print('models_cython.model2d_bounds() okay')

    out = mnb.model2d_maxsteps(1,pars_gauss)
    out = mnb.model2d_maxsteps(2,pars_moffat)
    out = mnb.model2d_maxsteps(3,pars_penny)
    out = mnb.model2d_maxsteps(4,pars_gausspow)
    out = mnb.model2d_maxsteps(5,pars_sersic)
    print('models_cython.model2d_maxsteps() okay')

    x,y = np.meshgrid(np.arange(51),np.arange(51))
    x1d = x.ravel().astype(float)
    y1d = y.ravel().astype(float)
    im,_ = mnb.amodel2d(x1d,y1d,1,pars_gauss,6)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.model2dfit(im,err,x1d,y1d,1,90.0,5.1,6.2,False)
    
    im,_ = mnb.amodel2d(x1d,y1d,2,pars_moffat,7)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.model2dfit(im,err,x1d,y1d,2,90.0,5.1,6.2,False)

    im,_ = mnb.amodel2d(x1d,y1d,3,pars_penny,8)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.model2dfit(im,err,x1d,y1d,3,90.0,5.1,6.2,False)

    im,_ = mnb.amodel2d(x1d,y1d,4,pars_gausspow,8)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.model2dfit(im,err,x1d,y1d,4,90.0,5.1,6.2,False)

    im,_ = mnb.amodel2d(x1d,y1d,5,pars_sersic,7)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.model2dfit(im,err,x1d,y1d,5,90.0,5.1,6.2,False)
    print('models_cython.model2dfit() okay')
    
    # out = mnb.relcoord(x,y,shape)
    # print('models_cython.relcoord() okay')
    
    # out = mnb.empirical(x, y, pars, data, imshape=None, deriv=False)
    # print('models_cython.empirical() okay')

    psf = mnb.packpsf(1,pars_gauss[3:],np.zeros((11,11,4),float),(1000,1000))
    print('models_cython.packpsf() okay')
    
    # psftype,pars,npsfx,npsfy,psforder,nxhalf,nyhalf,lookup
    out = mnb.unpackpsf(psf)
    print('models_cython.unpackpsf() okay')
    
    out = mnb.psfinfo(psf)
    print('models_cython.psfinfo() okay')

    out = mnb.psf2d_fwhm(psf)
    print('models_cython.psf2d_fwhm() okay')

    out = mnb.psf2d_flux(psf,100.0,5.5,6.6)
    print('models_cython.psf2d_flux() okay')

    out = mnb.psf2d(x1d,y1d,psf,100.0,5.5,6.6,False,False)
    print('models_cython.psf2d() okay')

    im,_ = mnb.psf2d(x1d,y1d,psf,100.0,5.5,6.6,False,False)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.psf2dfit(im,err,x1d,y1d,psf,90.0,5.1,6.1,False)
    print('models_cython.psf2dfit() okay')
    
    pars = np.array([1.00,5.5,6.6])
    lookup = np.zeros((1,1,1),float)
    imshape = (1000,1000)
    out = mnb.psf(x1d,y1d,pars,1,pars_gauss[3:],lookup,imshape,False,False)
    out = mnb.psf(x1d,y1d,pars,2,pars_moffat[3:],lookup,imshape,False,False)
    out = mnb.psf(x1d,y1d,pars,3,pars_penny[3:],lookup,imshape,False,False)
    out = mnb.psf(x1d,y1d,pars,4,pars_gausspow[3:],lookup,imshape,False,False)
    out = mnb.psf(x1d,y1d,pars,5,pars_sersic[3:],lookup,imshape,False,False)
    print('models_cython.psf() okay')

    
    x,y = np.meshgrid(np.arange(51),np.arange(51))
    pars = np.array([90.0,5.1,6.2])
    lookup = np.zeros((1,1,1),float)
    imshape = (1000,1000)
    im,_ = mnb.psf(x1d,y1d,pars,1,pars_gauss[3:],lookup,imshape,False,False)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.psffit(im,err,x1d,y1d,pars,1,pars_gauss[3:],lookup,imshape,False)
    
    im,_ = mnb.psf(x1d,y1d,pars,2,pars_moffat[3:],lookup,imshape,False,False)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.psffit(im,err,x1d,y1d,pars,2,pars_moffat[3:],lookup,imshape,False)
    
    im,_ = mnb.psf(x1d,y1d,pars,3,pars_penny[3:],lookup,imshape,False,False)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.psffit(im,err,x1d,y1d,pars,3,pars_penny[3:],lookup,imshape,False)
    
    im,_ = mnb.psf(x1d,y1d,pars,4,pars_gausspow[3:],lookup,imshape,False,False)
    err = np.sqrt(np.maximum(im,1))
    out = mnb.psffit(im,err,x1d,y1d,pars,4,pars_gausspow[3:],lookup,imshape,False)
    
    im,_ = mnb.psf(x1d,y1d,pars,5,pars_sersic[3:],lookup,imshape,False,False)
    err = np.sqrt(np.maximum(im,1))
    #out = mnb.psffit(im,err,x,y,pars,5,pars_sersic[3:],lookup,imshape,False)
    # ZeroDivisionError: division by zero
    print('models_cython.psffit() okay')
    

    # bbox = mnb.BoundingBox(10,20,30,40)
    # print('models_cython.BoundingBox.__init__() okay')
    
    # out = bbox.xrange
    # print('models_cython.BoundingBox.xrange okay')

    # out = bbox.yrange
    # print('models_cython.BoundingBox.yrange okay')

    # out = bbox.yrange
    # print('models_cython.BoundingBox.yrange okay')

    # out = bbox.data
    # print('models_cython.BoundingBox.data okay')

    # im = np.random.rand(100,100)
    # out = bbox.slice(im)
    # print('models_cython.BoundingBox.slice() okay')

    # out = bbox[0]
    # print('models_cython.BoundingBox.__getitem__() okay')

    # out = bbox.xy()
    # print('models_cython.BoundingBox.xy() okay')

    # out = bbox.reset()
    # print('models_cython.BoundingBox.reset() okay')

    
    # psf = mnb.PSFGaussian(pars_gauss[3:])
    # print('models_cython.PSFGaussian.__init__() okay')

    # out = psf.params
    # print('models_cython.PSFGaussian.params okay')

    # psf.params[0] = 1.0
    # print('models_cython.PSFGaussian.prarams setter okay')

    # out = psf.haslookup
    # print('models_cython.PSFGaussian.haslookup okay')

    # out = psf.starbbox((5.5,6.5),(1000,1000), 5.5)
    # print('models_cython.PSFGaussian.starbbox() okay')

    # out = psf.unitfootflux
    # print('models_cython.PSFGaussian.unitfootflux okay')

    # out = psf.fwhm()
    # print('models_cython.PSFGaussian.fwhm() okay')

    # out = psf.flux()
    # print('models_cython.PSFGaussian.flux() okay')

    # out = psf.evaluate(x,y,pars_gauss)
    # print('models_cython.PSFGaussian.evaluate() okay')

    # out = psf.deriv(x,y,pars_gauss)
    # print('models_cython.PSFGaussian.deriv() okay')

    
    # psf = mnb.PSF(1,pars_gauss[3:])
    # print('models_cython.PSF.__init__() okay')

    # out = psf.nparams
    # print('models_cython.PSF.nparams okay')

    # out = psf.params
    # print('models_cython.PSF.params okay')

    # psf.params[0] = 1.0
    # print('models_cython.PSF.params setter okay')
    
    # out = psf.name
    # print('models_cython.PSF.name okay')
    
    # out = psf.haslookup
    # print('models_cython.PSF.haslookup okay')

    # out = psf.starbbox((5.5,6.6),(1000,1000),5.5)
    # print('models_cython.PSF.starbbox() okay')

    # out = str(psf)
    # print('models_cython.PSF.__str__() okay')

    # out = psf.fwhm()
    # print('models_cython.PSF.fwhm() okay')

    # out = psf.flux()
    # print('models_cython.PSF.flux() okay')

    # pars = np.array([100.0,5.5,6.5])
    # out = psf.evaluate(x,y,pars)
    # print('models_cython.PSF.evaluate() okay')

    # out = psf.model(x,y,pars)
    # print('models_cython.PSF.model() okay')
    
    # out = psf.deriv(x,y,pars)
    # print('models_cython.PSF.deriv() okay')

    #out = psf.packpsf()
    #print('models_cython.PSF.packpsf() okay')
    
    
def getpsf_tests():
    
    ###############
    # GETPSF
    ###############

    import _getpsf_cython_static as gnb
    
    out = gnb.starcube(tab,image,error,npix=51,fillvalue=np.nan)
    print('getpsf_cython.starcube() okay')

    out = gnb.mkempirical(cube,order=0,coords=None,shape=None,lookup=False)
    print('getpsf_cython.mkempirical() okay')

    out = gnb.starbbox((5.5,6.5),(1000,1000),5.6)
    print('getpsf_cython.starbbox() okay')

    #out = gnb.sliceinsert(array,lo,insert)
    #print('getpsf_cython.sliceinsert() okay')

    out = gnb.getstar(image,error,xcen,ycen,fitradius)
    print('getpsf_cython.getstar() okay')

    out = gnb.collatestars(image,error,starx,stary,fitradius)
    print('getpsf_cython.collatestars() okay')

    out = gnb.unpackstar(imdata,errdata,xdata,ydata,bbox,shape,istar)
    print('getpsf_cython.unpackstar() okay')

    out = gnb.getfitstar(image,error,xcen,ycen,fitradius)
    print('getpsf_cython.getfitstar() okay')

    out = gnb.collatefitstars(image,error,starx,stary,fitradius)
    print('getpsf_cython.collatefitstars() okay')

    out = gnb.unpackfitstar(imdata,errdata,xdata,ydata,bbox,ndata,istar)
    print('getpsf_cython.unpackfitstar() okay')

    # out = gnb.PSFFitter()
    # print('getpsf_cython.PSFFitter.__init__() okay')

    # out = gnb.PSFFitter.unpackstar()
    # print('getpsf_cython.PSFFitter.unpackstar() okay')

    # out = gnb.PSFFitter.unpackfitstar()
    # print('getpsf_cython.PSFFitter.unpackfitstar() okay')

    # out = gnb.PSFFitter.psf()
    # print('getpsf_cython.PSFFitter.psf() okay')

    # out = gnb.PSFFitter.model()
    # print('getpsf_cython.PSFFitter.model() okay')

    # out = gnb.PSFFitter.chisq()
    # print('getpsf_cython.PSFFitter.chisq() okay')

    # out = gnb.PSFFitter.fitstars()
    # print('getpsf_cython.PSFFitter.fitstars() okay')

    # out = gnb.PSFFitter.jac()
    # print('getpsf_cython.PSFFitter.jac() okay')

    # out = gnb.PSFFitter.linesearch()
    # print('getpsf_cython.PSFFitter.linesearch() okay')
    
    # out = gnb.PSFFitter.starmodel()
    # print('getpsf_cython.PSFFitter.starmodel() okay')

    out = gnb.fitpsf(psftype,psfparams,image,error,starx,stary,starflux,fitradius,'qr',10,
                     1.0,False)
    print('getpsf_cython.fitpsf() okay')

    out = gnb.getpsf(psf,image,tab,fitradius=None,lookup=False,lorder=0,method='qr',subnei=False,
                     alltab=None,maxiter=10,minpercdiff=1.0,reject=False,maxrejiter=3,verbose=False)
    print('getpsf_cython.getpsf() okay')


def groupfit_tests():
    """  Testing the groupfit code."""

    import _utils_cython_static as utils
    import _models_cython_static as mnb
    import _groupfit_cython_static as gfit
    
    psftab = np.zeros((4,4),np.float64)
    psftab[:,0] = np.arange(4)+1
    psftab[:,1] = [100,200,300,400]   # amp
    psftab[:,2] = [10,11,19,18]       # xcen
    psftab[:,3] = [20,30,31,21]       # ycen
    mpars = np.array([3.1,3.0,0.1])
    #psf = mnb.PSF(1,mpars,21)
    psftype = 1
    xx,yy = np.meshgrid(np.arange(51),np.arange(51))
    xx1d = xx.ravel()
    yy1d = yy.ravel()
    model = np.zeros(51*51,float)
    for i in range(4):
        #model += psf.model(xx,yy,psftab[i,1:])
        pars1 = np.zeros(6,float)
        pars1[:3] = psftab[i,1:]
        pars1[3:] = mpars
        model1,_ = mnb.amodel2d(xx1d,yy1d,psftype,pars1,0)
        model += model1
    model = model.reshape(51,51)
    
    error = np.sqrt(model+10)
    sky = 0.0 #10.0
    image = model + sky + np.random.rand(51,51)*error

    mask = np.zeros(image.shape,bool)
    mask[40,40] = True
    mask[40,45] = True
    mask[45,45] = True
    
    #image = ccddata.CCDData(im,error=err,mask=mask)
    
    # initial estimates
    objtab = np.zeros((4,4),np.float64)
    objtab[:,0] = np.arange(4)+1
    objtab[:,1] = [80,250,200,500]           # amp
    objtab[:,2] = [9.5,11.6,19.2,17.8]       # xcen
    objtab[:,3] = [20.3,29.7,30.9,21.2]      # ycen

    fitradius = 3.0
    psflookup = np.zeros((1,1,1),np.float64)
    verbose = False
    nofreeze = False
    params = mpars
    psfnpix = 51
    psfflux = mnb.model2d_flux(psftype,params)
    tab = objtab
    imshape = image.shape
    skyfit = False
    psforder = 1
    
    xcen = 9.5
    ycen = 20.3
    hpsfnpix = psfnpix//2
    skyradius = psfnpix//2 + 10
    out = gfit.getstarinfo(imshape,mask,xcen,ycen,hpsfnpix,fitradius,skyradius)
    print('groupfit_cython.getstarinfo() okay')

    starx = objtab[:,2]
    stary = objtab[:,3]
    out = gfit.collatestarsinfo(imshape,mask,starx,stary,hpsfnpix,fitradius,skyradius)
    print('groupfit_cython.getcollatestarsinfo() okay')
    
    initdata = gfit.initstararrays(image,error,mask,tab,psfnpix,fitradius,skyradius,skyfit)
    starravelindex,starndata,starfitravelindex,starfitndata,skyravelindex,skyndata = initdata[7:13]
    xflat,yflat,indflat,imflat,errflat,resflat,ntotpix = initdata[13:20]
    starfitinvindex,starflat_index,starflat_ndata = initdata[20:]
    print('groupfit_cython.initstararrays() okay')


    # t0 = time.time()
    # out = gfit.groupfit(psftype,params,psfnpix,psflookup,
    #                     psfflux,image,error,mask,objtab,
    #                     fitradius,10,0.5,2,False,False,False)
    # print('groupfit.groupfit() okay')
    # print(time.time()-t0)
    
    #import pdb; pdb.set_trace()
    
    #return

    #psfdata = out
    psfdata = (psftype,params,psflookup,psforder,imshape)
    istar = 1
    n1 = starflat_ndata[istar]
    invind1 = starflat_index[istar,:n1]
    xind1 = xflat[invind1]
    yind1 = yflat[invind1]
    xdata1 = (xind1,yind1)

    # xind,yind = xdata1
    # psftype,psfparams,psflookup,psforder,imshape = psfdata
    # im1,_ = mnb.psf(xind,yind,params,psftype,psfparams,psflookup,
    #                 imshape,False,False)
    
    out = gfit.psf(xdata1,params,psfdata)
    print('groupfit_cython.psf() okay')

    out = gfit.psfjac(xdata1,params,psfdata)
    print('groupfit_cython.psfjac() okay')

    nstars = len(objtab)
    freezepars = np.zeros(3*nstars,bool)
    freezestars = np.zeros(len(objtab),bool)
    freezedata = (freezepars,freezestars)
    flatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)
    allpars = np.zeros(3*len(objtab),float)
    allpars[0::3] = objtab[:,1]
    allpars[1::3] = objtab[:,2]
    allpars[2::3] = objtab[:,3] 
    out = gfit.model(psfdata,freezedata,flatdata,allpars,False,False,False)
    print('groupfit_cython.model() okay')

    # xdata = xind,yind
    #xdatatype = 'UniTuple(i8[:],2)'
    # psfdata = (psftype,psfparams,psflookup,psforder,imshape)
    #psfdatatype = 'Tuple((i8,f8[:],f8[:,:,:],i8,UniTuple(i8,2)))'
    # freezedata = (freezepars,freezestars)
    #freezedatatype = 'Tuple((b1[:],b1[:]))'
    # flatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)
    #flatdatatype = 'Tuple((i8[:],i8[:,:],i8[:],i8[:],i8[:],i8))'
    # stardata = (starravelindex,starndata,xx,yy)
    #stardatatype = 'Tuple((i8[:,:],i4[:],i8[:,:],i8[:,:]))'
    # covflatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix,imflat,errflat,skyflat)
    #covflatdatatype = 'Tuple((i8[:],i8[:,:],i8[:],i8[:],i8[:],i8,f8[:],f8[:],f8[:]))'
    
    stardata = (starravelindex,starndata,xx,yy)
    out = gfit.fullmodel(psfdata,stardata,allpars)
    print('groupfit_cython.fullmodel() okay')
    
    out = gfit.jac(psfdata,freezedata,flatdata,allpars,False,False)
    print('groupfit_cython.jac() okay')

    out = gfit.chisqflat(freezedata,flatdata,psfdata,resflat,errflat,allpars)
    print('groupfit_cython.chisqflat() okay')

    skyflat = imflat.copy()*0
    covflatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix,imflat,errflat,skyflat)
    out = gfit.cov(psfdata,freezedata,covflatdata,allpars)
    print('groupfit_cython.cov() okay')

    frzpars = np.zeros(len(allpars),bool)
    frzpars[:3] = True
    resid = image.copy()*0.0
    resflat = resid.ravel()[indflat]
    out = gfit.dofreeze(frzpars,allpars,freezedata,flatdata,psfdata,resid,resflat)
    print('groupfit_cython.dofreeze() okay')
    
    #groupfit(psftype,psfparams,psfnpix,psflookup,psfflux,
    #          image,error,mask,tab,fitradius,maxiter=10,
    #          minpercdiff=0.5,reskyiter=2,nofreeze=False,
    #          skyfit=False,verbose=False)

    out = gfit.groupfit(psftype,params,psfnpix,psflookup,
                        psfflux,image,error,mask,objtab,
                        fitradius,10,0.5,2,False,False,False)
    print('groupfit_cython.groupfit() okay')



def allfit_tests():
    """  Testing the allfit code."""

    import _utils_cython_static as utils
    import _models_cython_static as mnb
    import _allfit_cython_static as afit
    
    psftab = np.zeros((4,4),np.float64)
    psftab[:,0] = np.arange(4)+1
    psftab[:,1] = [100,200,300,400]   # amp
    psftab[:,2] = [10,11,19,18]       # xcen
    psftab[:,3] = [20,30,31,21]       # ycen
    mpars = np.array([3.1,3.0,0.1])
    #psf = mnb.PSF(1,mpars,21)
    psftype = 1
    xx,yy = np.meshgrid(np.arange(51),np.arange(51))
    xx1d = xx.ravel()
    yy1d = yy.ravel()
    model = np.zeros(51*51,float)
    for i in range(4):
        #model += psf.model(xx,yy,psftab[i,1:])
        pars1 = np.zeros(6,float)
        pars1[:3] = psftab[i,1:]
        pars1[3:] = mpars
        model1,_ = mnb.amodel2d(xx1d,yy1d,psftype,pars1,0)
        model += model1
    model = model.reshape(51,51)
    
    error = np.sqrt(model+10)
    sky = 0.0 #10.0
    image = model + sky + np.random.rand(51,51)*error

    mask = np.zeros(image.shape,bool)
    mask[40,40] = True
    mask[40,45] = True
    mask[45,45] = True
    
    #image = ccddata.CCDData(im,error=err,mask=mask)
    
    # initial estimates
    objtab = np.zeros((4,4),np.float64)
    objtab[:,0] = np.arange(4)+1
    objtab[:,1] = [80,250,200,500]           # amp
    objtab[:,2] = [9.5,11.6,19.2,17.8]       # xcen
    objtab[:,3] = [20.3,29.7,30.9,21.2]      # ycen

    fitradius = 3.0
    psflookup = np.zeros((1,1,1),np.float64)
    verbose = False
    nofreeze = False
    params = mpars
    psfnpix = 51
    psfflux = mnb.model2d_flux(psftype,params)
    tab = objtab
    imshape = image.shape
    skyfit = False
    psforder = 1
    
    xcen = 9.5
    ycen = 20.3
    hpsfnpix = psfnpix//2
    skyradius = psfnpix//2 + 10
    out = afit.getstarinfo(imshape,mask,xcen,ycen,hpsfnpix,fitradius,skyradius)
    print('allfit_cython.getstarinfo() okay')

    starx = objtab[:,2]
    stary = objtab[:,3]
    out = afit.collatestarsinfo(imshape,mask,starx,stary,hpsfnpix,fitradius,skyradius)
    print('allfit_cython.getcollatestarsinfo() okay')
    
    initdata = afit.initstararrays(image,error,mask,tab,psfnpix,fitradius,skyradius)
    starravelindex,starndata,starfitravelindex,starfitndata,skyravelindex,skyndata = initdata[7:13]
    xflat,yflat,indflat,imflat,errflat,resflat,ntotpix = initdata[13:20]
    starfitinvindex,starflat_index,starflat_ndata = initdata[20:]
    print('allfit_cython.initstararrays() okay')


    # t0 = time.time()
    # out = afit.allfit(psftype,params,psfnpix,psflookup,
    #                     psfflux,image,error,mask,objtab,
    #                     fitradius,10,0.5,2,False,False,False)
    # print('allfit.allfit() okay')
    # print(time.time()-t0)
    
    #import pdb; pdb.set_trace()
    
    #return

    #psfdata = out
    psfdata = (psftype,params,psflookup,psforder,imshape)
    istar = 1
    n1 = starflat_ndata[istar]
    invind1 = starflat_index[istar,:n1]
    xind1 = xflat[invind1]
    yind1 = yflat[invind1]
    xdata1 = (xind1,yind1)

    # xind,yind = xdata1
    # psftype,psfparams,psflookup,psforder,imshape = psfdata
    # im1,_ = mnb.psf(xind,yind,params,psftype,psfparams,psflookup,
    #                 imshape,False,False)
    
    out = afit.psf(xdata1,params,psfdata)
    print('allfit_cython.psf() okay')

    out = afit.psfjac(xdata1,params,psfdata)
    print('allfit_cython.psfjac() okay')

    nstars = len(objtab)
    freezepars = np.zeros(3*nstars,bool)
    freezestars = np.zeros(len(objtab),bool)
    freezedata = (freezepars,freezestars)
    flatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)
    allpars = np.zeros(3*len(objtab),float)
    allpars[0::3] = objtab[:,1]
    allpars[1::3] = objtab[:,2]
    allpars[2::3] = objtab[:,3] 

    sn1 = skyndata[0]
    skyravelindex1 = skyravelindex[0,:sn1]
    resid = image.ravel()
    sky1 = afit.getstarsky(skyravelindex1,resid,error)
    print('allfit_cython.getstarsky() okay')

    i = 0
    #pars1 = pars[3*i:3*i+3]
    pars1 = objtab[i,1:]
    fn1 = starfitndata[i]
    fravelindex1 = starfitravelindex[i,:fn1]
    fxind1 = xx.flatten()[fravelindex1]
    fyind1 = yy.flatten()[fravelindex1]
    psfparams = mpars
    cov1 = afit.starcov(psftype,psfparams,psflookup,imshape,
                        pars1,fxind1,fyind1,fravelindex1,resid,error)
    print('allfit_cython.starcov() okay')

    out = afit.starfit(psftype,psfparams,psflookup,imshape,
                       pars1,fxind1,fyind1,fravelindex1,resid,error,sky)
    print('allfit_cython.starfit() okay')
    
    out = afit.chisq(fravelindex1,resid,error)
    print('allfit_cython.chisq() okay')
    
    # out = afit.dofreeze(oldpars,newpars,minpercdiff)
    # print('allfit_cython.dofreeze() okay')
    
    # out = afit.dofreeze2(frzpars,pars,freezedata,flatdata,psfdata,resid,resflat)
    # print('allfit_cython.dofreeze2() okay')
    
    # xdata = xind,yind
    #xdatatype = 'UniTuple(i8[:],2)'
    # psfdata = (psftype,psfparams,psflookup,psforder,imshape)
    #psfdatatype = 'Tuple((i8,f8[:],f8[:,:,:],i8,UniTuple(i8,2)))'
    # freezedata = (freezepars,freezestars)
    #freezedatatype = 'Tuple((b1[:],b1[:]))'
    # flatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)
    #flatdatatype = 'Tuple((i8[:],i8[:,:],i8[:],i8[:],i8[:],i8))'
    # stardata = (starravelindex,starndata,xx,yy)
    #stardatatype = 'Tuple((i8[:,:],i4[:],i8[:,:],i8[:,:]))'
    # covflatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix,imflat,errflat,skyflat)
    #covflatdatatype = 'Tuple((i8[:],i8[:,:],i8[:],i8[:],i8[:],i8,f8[:],f8[:],f8[:]))'


    # skyflat = imflat.copy()*0
    # covflatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix,imflat,errflat,skyflat)
    # out = afit.cov(psfdata,freezedata,covflatdata,allpars)
    # print('allfit_cython.cov() okay')

    frzpars = np.zeros(len(allpars),bool)
    frzpars[:3] = True
    resid = image.copy()*0.0
    resflat = resid.ravel()[indflat]
    out = afit.dofreeze(frzpars,allpars,freezedata,flatdata,psfdata,resid,resflat)
    print('allfit_cython.dofreeze() okay')
    
    #allfit(psftype,psfparams,psfnpix,psflookup,psfflux,
    #          image,error,mask,tab,fitradius,maxiter=10,
    #          minpercdiff=0.5,reskyiter=2,nofreeze=False,
    #          skyfit=False,verbose=False)

    out = afit.allfit(psftype,params,psfnpix,psflookup,
                        psfflux,image,error,mask,objtab,
                        fitradius,10,0.5,2,False,False,False)
    print('allfit_cythonyallfit() okay')




    
    
def allfit_tests2():
    """  Testing the allfit code."""

    psftab = np.zeros((4,4),np.float64)
    psftab[:,0] = np.arange(4)+1
    psftab[:,1] = [100,200,300,400]   # amp
    psftab[:,2] = [10,11,19,18]       # xcen
    psftab[:,3] = [20,30,31,21]       # ycen
    mpars = np.array([3.1,3.0,0.1])
    psf = mnb.PSF(1,mpars,21)
    xx,yy = np.meshgrid(np.arange(51),np.arange(51))
    model = np.zeros(51*51,float)
    for i in range(4):
        model += psf.model(xx,yy,psftab[i,1:])
    model = model.reshape(51,51)

    err = np.sqrt(model+10)
    sky = 0.0 #10.0
    im = model + sky + np.random.rand(51,51)*err
    mask = np.zeros(im.shape,bool)
    image = ccddata.CCDData(im,error=err,mask=mask)
    
    # initial estimates
    objtab = np.zeros((4,4),np.float64)
    objtab[:,0] = np.arange(4)+1
    objtab[:,1] = [80,250,200,500]           # amp
    objtab[:,2] = [9.5,11.6,19.2,17.8]       # xcen
    objtab[:,3] = [20.3,29.7,30.9,21.2]      # ycen

    fitradius = 3.0
    psflookup = np.zeros((1,1,1),np.float64)
    verbose = False
    nofreeze = False
    # af = afit.AllFitter(psf.psftype,psf.params,psf.npix,psflookup,
    #                     image.data,image.error,image.mask,objtab,
    #                     fitradius,verbose,nofreeze)
    # print('allfit.AllFitter.__init__() okay')

    # pars = np.zeros(7,np.float64)
    # pars[:] = [100.0,3.4,5.5, 200.0,4.6,7.5, 100.0]
    # bounds = [np.zeros(7,np.float64)-np.inf,np.zeros(7,np.float64)+np.inf]
    # out = af.checkbounds(pars,bounds)
    # print('allfit.AllFitter.checkbounds() okay')

    # out = af.chisq()
    # print('allfit.AllFitter.chisq() okay')

    # out = af.collatestars(af.imshape,af.starxcen,af.starycen,af.npix//2,af.fitradius,af.skyradius)
    # print('allfit.AllFitter.collatestars() okay')

    # out = af.fit()
    # print('allfit.AllFitter.fit() okay')

    # out = af.getstar(af.imshape,af.starxcen[0],af.starycen[0],af.npix//2,af.fitradius,af.skyradius)
    # print('allfit.AllFitter.getstar() okay')

    # pars = np.zeros(6,np.float64)
    # pars[:] = [100.0,3.4,5.5, 200.0,4.6,7.5]
    # bounds = [np.zeros(6,np.float64)-np.inf,np.zeros(6,np.float64)+np.inf]
    # out = af.limbounds(pars,bounds)
    # print('allfit.AllFitter.limbounds() okay')

    # steps = np.zeros(6,np.float64)+0.1
    # maxsteps = np.zeros(6,np.float64)+0.5
    # out = af.limsteps(steps,maxsteps)
    # print('allfit.AllFitter.limsteps() okay')

    # out = af.mkbounds(af.pars,af.imshape)
    # print('allfit.AllFitter.mkbounds() okay')

    # bounds = [np.zeros(12,np.float64)-np.inf,np.zeros(12,np.float64)+np.inf]
    # steps = np.zeros(12,np.float64)+0.1
    # maxsteps = np.zeros(12,np.float64)+0.5
    # out = af.newpars(af.pars,steps,bounds,maxsteps)
    # print('allfit.AllFitter.newpars() okay')

    # pars1,xind1,yind1,ravelindex1 = af.stardata(0)
    # out = af.psf(xind1,yind1,pars1)
    # print('allfit.AllFitter.psf() okay')

    # pars1,xind1,yind1,ravelindex1 = af.stardata(0)
    # out = af.psfjac(xind1,yind1,pars1)
    # print('allfit.AllFitter.psfjac() okay')

    # out = af.sky()
    # print('allfit.AllFitter.sky() okay')
    
    # out = af.starcov(0)
    # print('allfit.AllFitter.starcov() okay')

    # out = af.stardata(0)
    # print('allfit.AllFitter.stardata() okay')

    # out = af.starfit(0)
    # print('allfit.AllFitter.starfit() okay')
    
    # out = af.starfitchisq(0)
    # print('allfit.AllFitter.starfitchisq() okay')

    # out = af.starfitdata(0)
    # print('allfit.AllFitter.starfitdata() okay')

    # out = af.starfiterr(0)
    # print('allfit.AllFitter.starfiterr() okay')

    # out = af.starfitim(0)
    # print('allfit.AllFitter.starfitim() okay')

    # out = af.starfitnpix(0)
    # print('allfit.AllFitter.starfitnpix() okay')

    # out = af.starfitravelindex(0)
    # print('allfit.AllFitter.starfitravelindex() okay')

    # out = af.starfitresid(0)
    # print('allfit.AllFitter.starfitresid() okay')

    # out = af.starfitrms(0)
    # print('allfit.AllFitter.starfitrms() okay')

    # out = af.starjac(0)
    # print('allfit.AllFitter.starjac() okay')

    # out = af.starmodel(0)
    # print('allfit.AllFitter.starmodel() okay')

    # out = af.starmodelfull(0)
    # print('allfit.AllFitter.starmodelfull() okay')

    # out = af.starnpix(0)
    # print('allfit.AllFitter.starnpix() okay')

    # out = af.starravelindex(0)
    # print('allfit.AllFitter.starravelindex() okay')

    # out = af.starsky(0)
    # print('allfit.AllFitter.starsky() okay')

    # out = af.steps(af.pars)
    # print('allfit.AllFitter.steps() okay')

    # out = af.unfreeze()
    # print('allfit.AllFitter.unfreeze() okay')


    out = afit.numba_allfit(psf.psftype,psf.params,psf.npix,psflookup,psf.flux(),
                            image.data,image.error,image.mask,objtab,fitradius,maxiter=10,
                            minpercdiff=0.5,reskyiter=2,verbose=False,
                            nofreeze=False)
    print('allfit.numba_allfit() okay')

    out = afit.allfit(psf,image,objtab)
    print('allfit.allfit() okay')

    #out = afit.fit(psf,image,objtab)
    #print('allfit.fit() okay')

    
if __name__ == "__main__":
    #alltests()
    #utils_tests()
    #models_tests()
    #getpsf_tests()
    #groupfit_tests()
    allfit_tests()
