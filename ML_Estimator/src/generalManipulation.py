

def isIterableList(x):
    #from numpy import ndarray
    import numpy as np
    ##Tests x to see if it is an iterable list or tuple
    return (isinstance(x, list) or isinstance(x,tuple) or (isinstance(x,np.ndarray) and sum(x.shape)> 1))

def makeIterableList(x):
    ##Tests x to see if it is an iterable list, and if not (e.g. scalar, int, string) then returns a list which contains only x as it's element
    ## If is is a tuple, it si returned as-is
    if(isIterableList(x)):
        return x
    else:
        return [x]
