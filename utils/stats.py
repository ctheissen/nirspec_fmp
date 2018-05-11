import nirspec_pip as nsp
import numpy as np

def chisquare(data,model,dof=0):
    """
    Compute the chi-square value given a data and a model.
    """
    d = data.flux
    m = model.flux
    #return np.sum((d-m)**2/m) #the definition from scipy
    return np.sum((d-m)**2/data.noise**2)