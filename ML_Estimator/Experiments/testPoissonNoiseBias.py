import poissonNoiseBias as pNoise
import numpy as np
import time

def testError():
    nRun = 100

    ML, bias, err = np.zeros(nRun), np.zeros(nRun), np.zeros(nRun)

    ETA = None
    eTime = time.time(); sTime = time.time()
    sumTime = 0
    
    for run in range(nRun):
        print "\n ------ Run Number: ", run, "  of ", nRun-1
        if(ETA is not None):
            print "------- ETA:", ETA
        ML[run], bias[run], err[run] = pNoise.run()

        if(run%1 == 0):
            eTime = time.time()
            sumTime += eTime-sTime

            #Edit to use nUpdate or similar if nupdate != 1
            ETA = (sumTime/float(nRun))*(nRun-1-run)

            sTime = eTime
        
    #Calculate Stats
    print "Mean, StD, Var :"
    print "--ML :: ", ML.mean(), ML.std(), ML.std()*ML.std()
    print "--bias :: ", bias.mean(), bias.std(), bias.std()*bias.std()
    print "--err :: ", err.mean(), err.std(), err.std()*err.std()

        
if __name__ == "__main__":
    testError()

    print "FINISHED SUCCESSFULLY"
