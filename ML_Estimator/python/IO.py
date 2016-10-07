
def initialise_Output(filename, mode = 'w', verbose = True):
    import os
    '''
    Checks for directory existence and opens file for output.
    Modes are python default:
    --r : read-only (should most likely not be used with this routine
    --a : append
    --w : write

    verbose : If true, will output filename to screen
    '''

    
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    handle = open(filename, mode)

    if(verbose):
        print 'File will be output to: ',filename

    return handle
