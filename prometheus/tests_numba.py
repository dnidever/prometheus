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

    out = mnb.gaussfwhm(im)
    print('models_numba.gaussfwhm() okay')
    
    out = mnb.contourfwhm(im)
    print('models_numba.contourfwhm() okay')
    
    out = mnb.imfwhm(im)
    print('models_numba.imfwhm() okay')
    
    #out = mnb.linearinterp(binim,fullim,binsize)
    #print('models_numba.linearinterp() okay')
    
    #out = mnb.checkbounds(pars,bounds)
    #print('models_numba.checkbounds() okay')
    
    #out = mnb.limbounds(pars,bounds)
    #print('models_numba.limbounds() okay')
    
    #out = mnb.limsteps(steps,maxsteps)
    #print('models_numba.limsteps() okay')
    
    out = mnb.newlsqpars(pars,steps,bounds,maxsteps)
    print('models_numba.newlsqpars() okay')
    
    out = mnb.newbestpars(bestpars,dbeta)
    print('models_numba.newbestpars() okay')
    
    out = mnb.starbbox(coords,imshape,radius)
    print('models_numba.starbbox() okay')
    
    out = mnb.bbox2xy(bbox)
    print('models_numba.bbo2xy() okay')
    
    out = mnb.gauss_abt2cxy(asemi,bsemi,theta)
    print('models_numba.gauss_abt2cxy() okay')
    
    out = mnb.gauss_cxy2abt(cxx,cyy,cxy)
    print('models_numba.gauss_cxy2abt() okay')
    
    out = mnb.gaussian2d_flux(pars)
    print('models_numba.gaussian2d_flux() okay')
    
    out = mnb.gaussian2d_fwhm(pars)
    print('models_numba.gaussian2d_fwhm() okay')
    
    out = mnb.agaussian2d(x,y,pars,nderiv)
    print('models_numba.agaussian2d() okay')
    
    out = mnb.gaussian2d(x,y,pars,nderiv)
    print('models_numba.gaussian2d() okay')
    
    out = mnb.gaussian2dfit(im,err,ampc,xc,yc,verbose)
    print('models_numba.gaussian2dfit() okay')
    
    out = mnb.moffat2d_fwhm(pars)
    print('models_numba.moffat2d_fwhm() okay')
    
    out = mnb.moffat2d_flux(pars)
    print('models_numba.moffat2d_flux() okay')
    
    out = mnb.amoffat2d(x,y,pars,nderiv)
    print('models_numba.amoffat2d() okay')
    
    out = mnb.moffat2d(x,y,pars,nderiv)
    print('models_numba.moffat2d() okay')
    
    out = mnb.moffat2dfit(im,err,ampc,xc,yc,verbose)
    print('models_numba.moffat2dfit() okay')
    
    out = mnb.penny2d_fwhm(pars)
    print('models_numba.penny2d_fwhm() okay')
    
    out = mnb.penny2d_flux(pars)
    print('models_numba.penny2d_flux() okay')
    
    out = mnb.apenny2d(x,y,pars,nderiv)
    print('models_numba.apenny2d() okay')
    
    out = mnb.penny2d(x,y,pars,nderiv)
    print('models_numba.penny2d() okay')
    
    out = mnb.penny2dfit(im,err,ampc,xc,yc,verbose)
    print('models_numba.penny2dfit() okay')
    
    out = mnb.gausspow2d_fwhm(pars)
    print('models_numba.gausspow2d_fwhm() okay')
    
    out = mnb.gausspow2d_flux(pars)
    print('models_numba.gausspow2d_flux() okay')
    
    out = mnb.agausspow2d(x,y,pars,nderiv)
    print('models_numba.agausspow2d() okay')
    
    out = mnb.gausspow2d(x,y,pars,nderiv)
    print('models_numba.gausspow2d() okay')
    
    out = mnb.gausspow2dfit(im,err,ampc,xc,yc,verbose)
    print('models_numba.gausspow2dfit() okay')
    
    out = mnb.asersic2d(x,y,pars,nderiv)
    print('models_numba.asersic2d() okay')
    
    out = mnb.sersic2d(x, y, pars, nderiv)
    print('models_numba.sersic2d() okay')
    
    out = mnb.sersic2d_fwhm(pars)
    print('models_numba.sersic2d_fwhm() okay')
    
    out = mnb.sersic_b(n)
    print('models_numba.sersic_b() okay')
    
    out = mnb.create_sersic_function(Ie, re, n)
    print('models_numba.create_sersic_function() okay')
    
    out = mnb.sersic_lum(Ie, re, n)
    print('models_numba.sersic_lum() okay')
    
    out = mnb.sersic_full2half(I0,kserc,alpha)
    print('models_numba.sersic_full2half() okay')
    
    out = mnb.sersic_half2full(Ie,Re,alpha)
    print('models_numba.sersic_half2ful() okay')
    
    out = mnb.sersic2d_flux(pars)
    print('models_numba.sersic2d_flux() okay')
    
    out = mnb.sersic2d_estimates(pars)
    print('models_numba.sersic2d_estimates() okay')
    
    out = mnb.model2d(x,y,psftype,pars,nderiv)
    print('models_numba.model2d() okay')
    
    out = mnb.amodel2d(x,y,psftype,pars,nderiv)
    print('models_numba.amodel2d() okay')
    
    out = mnb.model2d_flux(psftype,pars)
    print('models_numba.model2d_flux() okay')
    
    out = mnb.model2d_fwhm(psftype,pars)
    print('models_numba.model2d_fwhm() okay')
    
    out = mnb.model2d_estimates(psftype,ampc,xc,yc)
    print('models_numba.model2d_estimates() okay')
    
    out = mnb.model2d_bounds(psftype)
    print('models_numba.model2d_bounds() okay')
    
    out = mnb.model2d_maxsteps(psftype,pars)
    print('models_numba.model2d_maxsteps() okay')
    
    out = mnb.model2dfit(im,err,x,y,psftype,ampc,xc,yc,verbose=False)
    print('models_numba.model2dfit() okay')
    
    out = mnb.relcoord(x,y,shape)
    print('models_numba.relcoord() okay')
    
    out = mnb.empirical(x, y, pars, data, imshape=None, deriv=False)
    print('models_numba.empirical() okay')
    
    out = mnb.unpackpsf(psf)
    print('models_numba.unpackpsf() okay')
    
    out = mnb.packpsf(psftype,pars,lookup=None,imshape=None)
    print('models_numba.packpsf() okay')
    
    out = mnb.psfinfo(psf)
    print('models_numba.psfinfo() okay')
    
    out = mnb.psf2d_fwhm(psf)
    print('models_numba.psf2d_fwhm() okay')
    
    out = mnb.psf2d_flux(psf,amp,xc,yc)
    print('models_numba.psf2d_flux() okay')
    
    out = mnb.psf(x,y,pars,psftype,psfparams,lookup,imshape,deriv=False,verbose=False)
    print('models_numba.psf() okay')
    
    out = mnb.psffit(im,err,x,y,pars,psftype,psfparams,lookup,imshape=None,verbose=False)
    print('models_numba.psffit() okay')
    
    out = mnb.psf2d(x,y,psf,amp,xc,yc,deriv=False,verbose=False)
    print('models_numba.psf2d() okay')
    
    out = mnb.psf2dfit(im,err,x,y,psf,ampc,xc,yc,verbose=False)
    print('models_numba.psf2dfit() okay')

    bbox = mnb.BoundingBox()
    print('models_numba.BoundingBox.__init__() okay')
    
    bbox = mnb.BoundingBox()
    out = bbox.xrange
    print('models_numba.BoundingBox.xrange okay')

    bbox = mnb.BoundingBox()
    out = bbox.yrange
    print('models_numba.BoundingBox.yrange okay')

    bbox = mnb.BoundingBox()
    out = bbox[0]
    print('models_numba.BoundingBox.__getitem__() okay')

    bbox = mnb.BoundingBox()
    out = bbox.slice(im)
    print('models_numba.BoundingBox.slice() okay')

    bbox = mnb.BoundingBox()
    out = bbox.xy()
    print('models_numba.BoundingBox.xy() okay')

    bbox = mnb.BoundingBox()
    out = bbox.reset()
    print('models_numba.BoundingBox.reset() okay')

    psf = mnb.PSFGaussian()
    print('models_numba.PSFGaussian.__init__() okay')

    psf = mnb.PSFGaussian()
    out = psf.params
    print('models_numba.PSFGaussian.params okay')

    psf = mnb.PSFGaussian()
    psf.params[0] = 1.0
    print('models_numba.PSFGaussian.prarams setter okay')

    psf = mnb.PSFGaussian()
    out = psf.haslookup
    print('models_numba.PSFGaussian.haslookup okay')

    psf = mnb.PSFGaussian()
    out = psf.starbbox()
    print('models_numba.PSFGaussian.starbbox() okay')

    psf = mnb.PSFGaussian()
    out = psf.unitfootflux
    print('models_numba.PSFGaussian.unitfootflux okay')

    psf = mnb.PSFGaussian()
    out = psf.fwhm()
    print('models_numba.PSFGaussian.fwhm() okay')

    psf = mnb.PSFGaussian()
    out = psf.evaluate()
    print('models_numba.PSFGaussian.evaluate() okay')

    psf = mnb.PSFGaussian()
    out = psf.deriv()
    print('models_numba.PSFGaussian.deriv() okay')

    psf = mnb.PSF()
    print('models_numba.PSF.__init__() okay')

    psf = mnb.PSF()
    out = psf.nparams
    print('models_numba.PSF.nparams okay')

    psf = mnb.PSF()
    out = psf.params
    print('models_numba.PSF.params okay')

    psf = mnb.PSF()
    psf.params[0] = 1.0
    print('models_numba.PSF.params setter okay')
    
    psf = mnb.PSF()
    out = psf.name
    print('models_numba.PSF.name okay')
    
    psf = mnb.PSFGaussian()
    out = psf.haslookup
    print('models_numba.PSF.haslookup okay')

    psf = mnb.PSF()
    out = psf.starbbox()
    print('models_numba.PSF.starbbox() okay')

    psf = mnb.PSFGaussian()
    out = str(psf)
    print('models_numba.PSF.__str__() okay')

    psf = mnb.PSF()
    out = psf.fwhm()
    print('models_numba.PSF.fwhm() okay')

    psf = mnb.PSF()
    out = psf.flux()
    print('models_numba.PSF.flux() okay')

    psf = mnb.PSF()
    out = psf.evaluate()
    print('models_numba.PSF.evaluate() okay')

    psf = mnb.PSF()
    out = psf.model()
    print('models_numba.PSF.model() okay')
    
    psf = mnb.PSF()
    out = psf.deriv()
    print('models_numba.PSF.deriv() okay')

    psf = mnb.PSF()
    out = psf.packpsf()
    print('models_numba.PSF.packpsf() okay')
    
    
def getpsf_tests():
    
    ###############
    # GETPSF
    ###############

    out = gnb.starcube(tab,image,error,npix=51,fillvalue=np.nan)
    print('getpsf_numba.starcube() okay')

    out = gnb.mkempirical(cube,order=0,coords=None,shape=None,lookup=False)
    print('getpsf_numba.mkempirical() okay')

    out = gnb.starbbox(coords,imshape,radius)
    print('getpsf_numba.starbbox() okay')

    out = gnb.sliceinsert(array,lo,insert)
    print('getpsf_numba.sliceinsert() okay')

    out = gnb.getstar(image,error,xcen,ycen,fitradius)
    print('getpsf_numba.getstar() okay')

    out = gnb.collatestars(image,error,starx,stary,fitradius)
    print('getpsf_numba.collatestars() okay')

    out = gnb.unpackstar(imdata,errdata,xdata,ydata,bbox,shape,istar)
    print('getpsf_numba.unpackstar() okay')

    out = gnb.getfitstar(image,error,xcen,ycen,fitradius)
    print('getpsf_numba.getfitstar() okay')

    out = gnb.collatefitstars(image,error,starx,stary,fitradius)
    print('getpsf_numba.collatefitstars() okay')

    out = gnb.unpackfitstar(imdata,errdata,xdata,ydata,bbox,ndata,istar)
    print('getpsf_numba.unpackfitstar() okay')

    out = gnb.PSFFitter()
    print('getpsf_numba.PSFFitter.__init__() okay')

    out = gnb.PSFFitter.unpackstar()
    print('getpsf_numba.PSFFitter.unpackstar() okay')

    out = gnb.PSFFitter.unpackfitstar()
    print('getpsf_numba.PSFFitter.unpackfitstar() okay')

    out = gnb.PSFFitter.psf()
    print('getpsf_numba.PSFFitter.psf() okay')

    out = gnb.PSFFitter.model()
    print('getpsf_numba.PSFFitter.model() okay')

    out = gnb.PSFFitter.chisq()
    print('getpsf_numba.PSFFitter.chisq() okay')

    out = gnb.PSFFitter.fitstars()
    print('getpsf_numba.PSFFitter.fitstars() okay')

    out = gnb.PSFFitter.jac()
    print('getpsf_numba.PSFFitter.jac() okay')

    out = gnb.PSFFitter.linesearch()
    print('getpsf_numba.PSFFitter.linesearch() okay')
    
    out = gnb.PSFFitter.starmodel()
    print('getpsf_numba.PSFFitter.starmodel() okay')

    out = gnb.fitpsf(psftype,psfparams,image,error,starx,stary,starflux,fitradius,method='qr',maxiter=10,
                     minpercdiff=1.0,verbose=False)
    print('getpsf_numba.fitpsf() okay')

    out = gnb.getpsf(psf,image,tab,fitradius=None,lookup=False,lorder=0,method='qr',subnei=False,
                     alltab=None,maxiter=10,minpercdiff=1.0,reject=False,maxrejiter=3,verbose=False)
    print('getpsf_numba.getpsf() okay')
