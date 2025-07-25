import numpy as np
import plotly.graph_objects as go
from itertools import permutations

def brute_force(planes,data):
#     planes = np.array([1,2,3,4])
    
    gps_data = np.copy(data)
    np.random.shuffle(gps_data)
    
#     p1,p2,p3,p4 = planes[0],planes[1],planes[2],planes[3]
#     gpsA,gpsB,gpsC,gpsD = gps_data[0],gps_data[1],gps_data[2],gps_data[3]

    gps_perms = np.array(list(permutations(gps_data)))
    
    sums = []
    perm_array = []
    for perm in gps_perms:
        gxA,gyA,gzA = perm[0,0,0],perm[0,1,0],perm[0,2,0]
        gxB,gyB,gzB = perm[1,0,0],perm[1,1,0],perm[1,2,0]
        gxC,gyC,gzC = perm[2,0,0],perm[2,1,0],perm[2,2,0]
        gxD,gyD,gzD = perm[3,0,0],perm[3,1,0],perm[3,2,0]
        
        x1,y1,z1 = planes[0].mu[0,0],planes[0].mu[1,0],planes[0].mu[2,0]
        x2,y2,z2 = planes[1].mu[0,0],planes[1].mu[1,0],planes[1].mu[2,0]
        x3,y3,z3 = planes[2].mu[0,0],planes[2].mu[1,0],planes[2].mu[2,0]
        x4,y4,z4 = planes[3].mu[0,0],planes[3].mu[1,0],planes[3].mu[2,0]
        
        val1 = np.square(x1-gxA)+np.square(y1-gyA)+np.square(z1-gzA)
        val2 = np.square(x2-gxB)+np.square(y2-gyB)+np.square(z2-gzB)
        val3 = np.square(x3-gxC)+np.square(y3-gyC)+np.square(z3-gzC)
        val4 = np.square(x4-gxD)+np.square(y4-gyD)+np.square(z4-gzD)
        
        val = val1+val2+val3+val4
        sums.append(val)
        perm_array.append(perm)
    n = np.argmin(sums)

    return perm_array[n]