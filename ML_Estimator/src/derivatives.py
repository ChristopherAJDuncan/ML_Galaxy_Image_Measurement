'''
Library routine which calculates nth order derivatives over a scalar function, or a scalar function on a grid.

This is essentially a (fairly poor) extension to the scipy default, which only works on scalar values

Author: cajd
'''


def finite_difference_derivative(fun, x0, args, n = [1], order = 5, dx = 1., maxEval = 100, eps = 1.e-4, convergenceType = 'sum'):
    '''
    Compute numerical derivatives using finite differences.

    Requires:
    -- fun: symbolic link to function to be differentiated. Function must take arguements of the form (x, *args), where an element of the args tuple can defined which variable is being differentiated.
    -- x0: The central point around which the derivative is being taken.
    -- args: tuple containing the arguements required by the function which is being differentiated, beyond the variable being passed as x (the one being differentiated). Where multiple varables are differentiable, args shold contain a reference labelling the variable which is being differentiated wrt.
    -- n: tuple or integer detailing the order of differentiation (1: 1st derivative etc). Result is output as a tuple containing the same number of elements as n is input.
    -- order: the number fo finte intervals used to calculate the derivative.
    -- dx: The interval width to use
    -- maxEval: the maximum number of evaluations to use to calculate the derivative
    -- eps: the Tolerance which must be satisfied to return a converged estimate.
    -- convergenceType:  determines how the convergnece test is applied. If == sum, convergence test is applied to the sum over all pixels. If == pix, convergenceTest carried out on all pixels individually.

    Returns:
    -- result: Tuple contining the derivative of the function to order as set by `n` input.
    '''
    
    import numpy as np

    ##Check whether n is tuple, and if not make it one
    if(not hasattr(n, "__iter__")):
       n = [n]

    ##Evaluate once if width is passed in and is scalar
    if dx <= 0:
        raise ValueError('finite_difference_derivative - dx Entered is not valid (negative)')
    if(not hasattr(dx, "__iter__")):
        maxEval = 1
        dx = [dx, 0.]
    if hasattr(dx, "__iter__"):
        if len(dx) != 2:
            raise ValueError('finite_difference_derivative - Entered dx is not applicable: Entered dx should either be a scalar (for single run), or a tuple of length two (for convergence test [start point, width]). dx:'+str(dx))


    result = []; convdx = []
    for nn in range(len(n)):

        ##Convergence Testing - tests for convergence individually
        #Initialise result_store
        result_store = [0]*maxEval
        foundConvergence = False
        for ev in range(maxEval):

            idx = dx[0] + ev*dx[1]

            if maxEval == 1:
                foundConvergence = True
            
            ##Get finite interval fucntion evaluations up to order entered
            f = []
            
            no = int(order/2)
            for i in range(-no, no+1, 1):
                try:
                    f.append(np.array(fun(x0 - i*idx, *args)))
                except:
                    raise ValueError('finite_difference_derivative - Failed to evaluate function over range considered - interval:', no, ' of:', order,'. Value, interval:', x0 - i*idx, idx)
                if(i >= -no+1 and f[i+no].shape != f[i+no-1].shape):
                    raise RuntimeError('finite_difference_derivative - Evaluation of function'+str(fun)+' at point '+str(x0 - i*idx)+' failed')

            if(n[nn] == 1):
                #ck for order [1,2,3,4,5,6,7,8,9]
                ick = [['1'],['2'],[1.,0,-1.] 
                       ,['4'],[1.,-8.,0.,8.,-1.],['6']
                       ,[-1.,9.,-45.,0.,45.,-9.,1.],['8'],[3.,-32.,168.,-672.,672.,168.,32.,-3.]]
                
                ich = ['1','2',2. 
                       ,'4',12.,'6'
                       ,60.,'8',840.]
                
                ck = ick[order-1]
                ch = ich[order-1]
                
                if(len(ck) != len(f)):
                    raise ValueError('derivative - Error assigning coefficents to function evaluations.')
                
                res = np.zeros(f[0].shape)
                for o in range(len(ck)):
                    res += ck[o]*f[o]
                res /= ch*idx
                
            elif(n[nn] == 2):
                #ck for order [1,2,3,4,5,6,7,8,9]
                ick = [['1'],['2'],[1.,-2.,1.] 
                       ,['4'],[-1.,16.,-30.,16.,-1.],['6']
                       ,[2.,-27.,270.,-490.,270.,-27.,2.],['8'],['9']]
                
                ich = ['1','2',1. 
                       ,'4',12.,'6'
                       ,180.,'8','9']
                
                
                ck = ick[order-1]
                ch = ich[order-1]
                
                if(len(ck) != len(f)):
                    raise ValueError('derivative - Error assigning coefficents to function evaluations.', order, ch)
                
                res = np.zeros(f[0].shape)
                for o in range(len(ck)):
                    res += ck[o]*f[o]
                res /= ch*idx*idx
                
            elif(n[nn] == 3):
                #ck for order [1,2,3,4,5,6,7,8,9]
                ick = [['1'],['2'],['3'] 
                       ,['4'],[-1.,2.,0.,-2.,1.],['6']
                       ,[1.,-8.,13.,0.,-13.,8.,-1.],['8'],['9']]
                
                ich = ['1','2','3' 
                       ,'4',2.,'6'
                       ,8.,'8','9']
                
                ck = ick[order-1]
                ch = ich[order-1]
                
                if(len(ck) != len(f)):
                    raise ValueError('derivative - Error assigning coefficents to function evaluations.', order, ch)

                res = np.zeros(f[0].shape)
                for o in range(len(ck)):
                    res += ck[o]*f[o]
                    '''
                    try:
                        res += ck[o]*f[o]
                    except:
                        print 'Failed in third order:'
                        print 'Arrays:', ck[o], f[o]
                        print 'Shape check:', ck[o], f[o].shape
                    '''
                res /= ch*idx*idx

                
            else:
                raise ValueError('derivative - Derivative order (n) not valid. I can only consider 1st, 2nd and 3rd derivatives. Order entered is:', n[nn])

            result_store[ev] = res

            '''
            if(ev >= 1):
                print 'Result Check:'
                print result_store[ev-2:ev], idx
                print (np.absolute(result_store[ev]-result_store[ev-1]) <= eps), (np.absolute(result_store[ev]-result_store[ev-1]) <= eps).sum()
                raw_input('Check')
            '''

            ##Validity check
            if(ev >= 1 and result_store[ev].shape != result_store[ev-1].shape):
                raise RuntimeError('finite_difference_derivative - Fatal error in determining result: result shape not consistent:: ', result_store[ev].shape, result_store[ev-1].shape)

            ##Convergence Test
            if(ev >= 1 and convergenceType.lower() == 'pix' and (np.absolute(result_store[ev]-result_store[ev-1]) <= eps).sum() == np.prod(result_store[ev].shape)):
                foundConvergence = True
                break
            if(ev >= 1 and convergenceType.lower() == 'sum' and (np.absolute((result_store[ev]-result_store[ev-1]).sum()) <= eps)):
                #Should np.absolute(result_store[ev]-result_store[ev-1].sum()) <= eps, in which case no bias cancelling across pixels is allowed?
                foundConvergence = True
                break


        if(foundConvergence):
            result.append(res)
            convdx.append(idx)
        else:
            print '\n ---Failed to find convergence in derivative across all pixels for order: ', n[nn],'::'
            print 'Final Difference (Pix) logical sum:', (np.absolute(result_store[-1]-result_store[-2]) <= eps).sum(), eps, np.prod(result_store[ev].shape)
            print 'Final Difference (sum) :', np.absolute((result_store[ev]-result_store[ev-1]).sum()), eps
            print 'Final Width:', idx
            print 'nEval:', ev
            print '\n -------------------------------------------------------------------------------------'
        
    return result
        
