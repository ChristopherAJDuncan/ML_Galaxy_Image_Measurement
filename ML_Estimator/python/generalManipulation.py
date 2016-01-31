"""
Module containing `helper` routines. These are not intrinsically linked with the evaluation of the end result, but rather manipulation of structures.
Author cajd
"""

def isIterableList(x):
    """
    Takes input x, and returns True if x is a list, tuple or numpy array, False otherwise (including for strings which are iterabel by Python standards
    """
    import numpy as np
    ##Tests x to see if it is an iterable list or tuple
    return (isinstance(x, list) or isinstance(x,tuple) or (isinstance(x,np.ndarray) and sum(x.shape)>0))

def makeIterableList(x):
    """
    Tests whether x is an iterable list (as defined by the output of isIterableList). If isIterableList==False for x, then x is returned as a tuple contaning x.
    Useful when the application of a routine requires a certain parameter to be iterable.
    """
    ##Tests x to see if it is an iterable list, and if not (e.g. scalar, int, string) then returns a list which contains only x as it's element
    ## If is is a tuple, it si returned as-is
    if(isIterableList(x)):
        return x
    else:
        return [x]
