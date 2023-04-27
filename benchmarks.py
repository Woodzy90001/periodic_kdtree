import numpy as np
import time

from scipy.spatial import KDTree
from periodic_kdtree import PeriodicKDTree



def runBenchmarks(m, n, r, size, cutoff):


    cutoffDist = np.ones((3))*cutoff
    data = np.random.random((n, 3))*size
    queries = np.random.random((n, 3))*size
    
    print("------------------------------")
    print("---     Periodic Tree      ---")
    print("------------------------------\n")
    
    print("dimension {}, {} points\n\n".format(m,n))
    
    t = time.time()
    T1 = PeriodicKDTree(cutoffDist, data, size)
    print("PeriodicKDTree constructed:\t{}".format(time.time()-t))
    
    t = time.time()
    w = T1.query(queries)
    print("PeriodicKDTree {} lookups:\t{}".format(r, time.time()-t))
    del w

    
    
    

    print("\n\n\n\n------------------------------")
    print("---      Normal Tree       ---")
    print("------------------------------\n\n")
    
    t = time.time()
    T1 = KDTree(data)
    print("KDTree constructed:\t{}".format(time.time()-t))
    
    t = time.time()
    w = T1.query(queries)
    print("KDTree {} lookups:\t{}".format(r, time.time()-t))
    del w

size = 200
cutoff = 5

m = 3
n = 10000
r = 1000
runBenchmarks(m, n, r, size, cutoff)

print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

m = 3
n = 1000000
r = 1000000
runBenchmarks(m, n, r, size, cutoff)