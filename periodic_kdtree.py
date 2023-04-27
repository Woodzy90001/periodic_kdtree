# periodic_kdtree_custom.py
#
# A wrapper around scipy.spatial.kdtree to implement periodic boundary
# conditions
#
# Written by Callum Wood, 27 April 2023
#
# A modification of the original periodic KD tree written by Patrick Varilly
# https://github.com/patvarilly/periodic_kdtree
#
# Released under the scipy license

import time
import numpy as np
from scipy.spatial import KDTree

def _CalculateOffsets(inOffsets, testArray, n):
    
    '''
    
    Calculates the overlap. 
    ie.
    [1, 0, -1] = [[1, 0, 0], [0, 0, -1], [1, 0, -1]]
    
    '''

    outList = []
    
    index = 0
    for offset in inOffsets:
        if (offset != 0):
        
            '''
            For any offset add it to each previous offset
            
            Eg from the example above, the first iteration it is empty so the
            first two for loops are skipped. In the third iteration, the offset
            is applied to the existing offset/s and then added individually.
            
            This ensures all combinations are accounted for. 
            
            '''
            addLoop = []
            
            for element in outList:
                offsetVect = np.array(element, copy=True)
                offsetVect[index] = offset
                addLoop.append(offsetVect)
                
            for elem in addLoop:
                outList.append(elem)
        
            # Finally, if there is an offset make it in the appropriate 
            # direction and add it
            singleOffset = np.zeros((n))
            singleOffset[index] = offset
        
            if not np.array_equal(testArray, singleOffset):
                outList.append(singleOffset)
        
        index += 1
        
    return outList

def _allPoints(points, cutoffDists, size, debug=False):
    
    '''
    This function copies all points within some distance from the edge of the 
    box one unit cell over. This allows the KD tree to sample these points
    and find neighbours over the boundaries. 
    
    points : array_like, shape (k)
        An array of all the original points
    cutoffDists : array_like, shape (n)
        The depth into the box that the points will be copied over
    size : positive int
        The size of the box. Note it is assumed the dimensions of the box
        are the same in all directions
    debug : bool
        Flag to show timing info
    '''
    
    # Create a list to hold all points, including overlapped points. 
    allPoints = []
    if debug: startOrigList = time.time()
    for point in points:
        allPoints.append(point)
    if debug: print("Original points added. Time taken   - {}".format(time.time() - startOrigList))
    
    
    if debug: dupOrigList = time.time()

    # Create a list to add point indexes to which allows for mapping of 
    # periodic points to original points
    idList = list(range(len(allPoints)))

    # Find the dimensionality of points. Assumes all points are of the same 
    # dim, so only first is checked
    n = len(allPoints[0])
    
    # Create a zero array to check against inside loop to prevent initialising 
    # it each iteration
    testArray = np.zeros((n))


    maxCutoffDist = max(cutoffDists)
    count = 0
    for point in points:
        offset = np.array([0,0,0])

        '''
        
        Witihin 0->cutoffDist, +1
        Within size-cutoffDist->cutoffDist, -1
        
          +1           -1
        [. . ' . . . ' . .]
        [. .][. . ' . . . ' . .][. .]
        
        I tried this a few ways, including using heaviside but this was the 
        fastest
        
        '''
        
        if (min(point) < maxCutoffDist) or (max(point) > (size-maxCutoffDist)):
            index = 0
            for elem in point:
                if (elem < cutoffDists[index]):
                    offset[index] = 1
                elif (elem > (size - cutoffDists[index])):
                    offset[index] = -1
                
                index += 1
            
            
        if not np.array_equal(testArray, offset):
            outList = _CalculateOffsets(offset, testArray, n)
            
            for offsetArray in outList:
                offsetArray *= size
                allPoints.append(np.add(point, offsetArray))
                idList.append(count)
                
        count += 1
                
     
    idList = np.asarray(idList)
    allPoints = np.asarray(allPoints)       

    if debug: print("Periodic points coppied. Time taken - {}\nNumber of coppied points - {}".format(time.time() - dupOrigList, len(allPoints) - len(points)))
    
    return allPoints, idList


class PeriodicKDTree():
    """
    kd-tree for quick nearest-neighbor lookup with periodic boundaries
    See scipy.spatial.kdtree for details on kd-trees.
    Searches with periodic boundaries are implemented by copying all
    points within some distance from the edges one unit cell over. 
    This then allows the tree to sample these periodic points. 
    
    The copying is done in the initilisation. This step therefore takes a 
    relatively longer period of time, but the lookup time is then comparable to 
    a standard non periodic kd tree
    
    Note the input data needs to be from 0-box size, not centred about 0
    """
    
    def __init__(self, cutoffDists, points, size, debug=False):# leafsize=10
        """Construct a kd-tree.
        Parameters
        ----------
        cutoffDists : array_like, shape (n)
            How far the periodic search will look into neighbouring boxes.
        points : array_like, shape (n,k)
            Array of all original points. This will be overwritten with the 
            periodic points allowing a sampling of the periodic points through
            use of a normal KDTree
        size : positive int
            The size of the bounding box. It is assumed the size is identical
            on all sides
        debug : bool
            Flag to show timing info
        """

        # Format input data
        self.cutoffDists = np.array(cutoffDists)
        self.points = np.asarray(points)
        self.size = size
        
        # Generate array of all overlaped points
        if debug: totTimeStart = time.time()
        self.points, self.overlapIDs = _allPoints(self.points, self.cutoffDists, self.size, debug=debug)
        
        # Set up underlying kd-tree
        if debug: treeMakeStart = time.time()
        self.tree = KDTree(self.points)
        if debug: print("Tree made. Time taken               - {}\nTotal time taken                    - {}".format(time.time() - treeMakeStart, time.time() - totTimeStart))

    # The following name is a kludge to override KDTree's private method
    #_KDTree__query
    def query(self, data, k=1, eps=0, p=2, distance_upper_bound=float("inf"), workers=1):
        # Calls tree and returns NN
        dists, index = self.tree.query(data, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound, workers=workers)
        
        index = self.overlapIDs[index]
        
        return dists, index
    
    def query_outer(self, data, k=1, eps=0, p=2, distance_upper_bound=float("inf"), workers=1):
        # Calls the tree and returns points outside the box
        dists, index = self.tree.query(data, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound, workers=workers)
        
        points = self.points[index]
        
        index = self.overlapIDs[index]
        
        return dists, index, points
    
    def find_centre(self, points, NNIndexes, size):
        
        '''
        Not working currently 
        '''
        # Takes in an array of points and their nearest neighbours and offsets
        # the atoms until the center point is within the bounding cube.
        # Use case: When defining bonds by two atoms, if chunking the data
        #           the centre of the bond must be within the original box
        
        retPoints = []
        count = 0
        for point in points:
            neighbours = []
            
            neighbourPoints = self.points[NNIndexes[count]]
            neighbourInt = 0
            for neighbourPoint in neighbourPoints:
                mid = np.add(neighbourPoint, point)/2
                
                # If the mid point is out of the bounding box, then the
                # neighbour is placed in the box and the original point is
                # placed outside the box
                if (min(mid) < 0) or (max(mid) > size):
                    NNOriginalCoords = self.points[NNIndexes[count][neighbourInt]]
                    distTest = np.linalg.norm(NNOriginalCoords-point)
                    
                    # Systematically moving the point until the distance 
                    # decreases
                    testOgPoint = np.array(point, copy=True)
                    for i in range(len(testOgPoint)):
                        testOgPoint[i] -= size
                        if (np.linalg.norm(NNOriginalCoords-testOgPoint)>distTest):
                            testOgPoint[i] += 2*size
                            if (np.linalg.norm(NNOriginalCoords-testOgPoint)>distTest):
                                testOgPoint[i] -= size
                                
                    neighbours.append(np.asarray([testOgPoint, NNOriginalCoords]))
                else:
                    neighbours.append(np.asarray([point, neighbourPoint]))
                    
                testMid = np.add(neighbours[-1][0], neighbours[-1][1])/2
                if (min(testMid) < 0) or (max(testMid) > size):
                    print("Error: Mid process failed. \nPoint 1  : {}\nPoint 2  : {}\nMidpoint : {}".format(neighbours[-1][0], neighbours[-1][1], testMid))
            
            retPoints.append(np.asarray(neighbours))
            neighbourInt += 1
            
        return np.asarray(retPoints)
    
        

    def _KDTree__query_ball_point(self, x, r, p=2., eps=0):
        raise NotImplementedError()

    def query_ball_tree(self, other, r, p=2., eps=0):
        raise NotImplementedError()

    def query_pairs(self, r, p=2., eps=0):
        raise NotImplementedError()
    
    def count_neighbors(self, other, r, p=2.):
        raise NotImplementedError()
        
    def sparse_distance_matrix(self, other, max_distance, p=2.):
        raise NotImplementedError()