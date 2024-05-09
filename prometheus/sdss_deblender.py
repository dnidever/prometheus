import os
import numpy as np

# Simple implementation of the SDSS Deblender (Lupton+20XX)

def peak_find(im,width=5,threshold=5):
    """
    Find peaks in the image.
    """

    # Convolve the image with PSF width
    # Peak values above some threshold
    cim = convolve(im,weight)
    mask = (cim <= threshold)
    
    return tab

def make_templates(im,tab):
    """
    Make templates for each peak
    """

    # Assume an axis of symmetry for each peak
    # Compare the intensities of pairs of pixels symmetricaly placed about
    # the peak, and replacing both by the lower of the two.

    # the symmetry assumed is that the child has a two-fold rotational axis through its centre

    # Two-fold rotational symmetry, also known as rotational symmetry of order two,
    # is a type of symmetry in which an object looks the same after being rotated by
    # 180 degrees (half a full turn) around a central point. This means that if you
    # rotate the object by 180 degrees, it will appear unchanged.
    
    # Must have non-negative flux
    
    return templates

def get_weights(im,templates):
    """
    Calculate weights by minimizing the cost function E.
    E = Sum((I_i - Sum(w_r * T_r,i))^2)
    where
    I_i is the blended image
    w_r is the weight for image r
    T_r,i is the ith pixel value of template T_r
    """

    # Minimize the cost function
    # Can do this with simply solving the linear algebra problem
    
    return weights

def deblend_flux(im,templates,weights):
    """
    Parcel out the flux based on the templates and weights
    """

    # C_r,i = Ii * (w_r * T_r,i)/Sum(w_r*T_r,i)
    sumwttemp = np.sum(templates*weights.reshape())

    images = []
    for i in range(len(weights)):
        childim = im * (weights[i]*templates[i])/sumwttemp
        images.append(childim)
        
    return images
    
    
def deblend(im):
    """
    Implementation of the SDSS Deblender.
    """

    # Step 1. Detect peaks
    tab = peak_find(im)
    nchild = len(tab)
    
    # Step 2. Generate template
    templates = make_templates(im,tab)

    # Step 3. Calculate weights
    weights = get_weights(im,templates)
    
    # Step 4. Parcel out flux based on template and weights
    images = deblend_flux(im,templates,weights)

    # Allow for images to be used for multiple bands.
    
    # Could we iterate?  Use the deblended image as the templatews
    # and do the deblending again?
    
    # Make list of dictionaries for each child
    out = nchild*[]
    for i in range(nchild):
        new = {'n':i+1,'xpeak':tab['x'][i],'ypeak':tab['y'][i],
               'template':templates[i],'weight':weights[i],
               'image':images[i]}
        out[i] = new
    
    return out
