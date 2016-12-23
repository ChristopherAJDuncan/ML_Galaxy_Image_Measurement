import os

def initialise_Directory(direct):
        
    directory = os.path.dirname(os.path.dirname(direct))
    if not os.path.exists(directory):
        os.makedirs(directory)

    return 1
        
def initialise_Output(filename, mode = 'w', verbose = True):
    '''
    Checks for directory existence and opens file for output.
    Modes are python default:
    --r : read-only (should most likely not be used with this routine
    --a : append
    --w : write

    verbose : If true, will output filename to screen
    '''

    initialise_Directory(filename)
        
    handle = open(filename, mode)

    if(verbose):
        print 'File will be output to: ',filename

    return handle
