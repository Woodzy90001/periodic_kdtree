# periodic_kdtree
Periodic wrapper for the scipy.spatial KDTree

This is a modification of Patrick Varilly's periodic KDTree, implimenting periodic boundaries in a different way and updated to python 3.
https://github.com/patvarilly/periodic_kdtree

Only the query function of the KDTree has been implimented. 

When finding bonds between atoms in MD simulations, a nearest neighbour search is often needed to find an atoms bond. These models however have periodic boundaries so a periodic nearest neighbour search is needed. Creating mirror copies can quickly become expensive when dealing with large failes, ie 1000000 atoms. This implimentation maps points within some distance from the boundaries a unit cell over, enabling points near the boundaries to sample points in neighbouring cells. 

For MD simulations, the interatomic distance is usualy 1-2 angstroms, so a distance of 5 angstroms is used in the examples to ensure all relevant points are sampled but not many more. The distance can be changed easily as is required, and for periodic boundiers only along certain axis, simply set the distance to zero for the closed axis. Ie [2,0,2] is periodic in the x and z axis, but not the y, sampling 2 units into the x and z axis.

Basic Usage
-----------

from periodic_kdtree import PeriodicCKDTree
import numpy as np

# Cutoff distance (0 or negative means open boundaries in that dimension)
cutoff = np.array([5, 5, 0])   # xy periodic

# Points
size = 200
numPoints = 10000
points = np.random.random((numPoints, 3))*size

# Build kd-tree
tree = PeriodicKDTree(cutoff, points, size)

# Find 4 closest neighbors to each point
dists, index = tree.query(points, k=4)

Tests and benchmarks
--------------------

See test_periodic_kdtree.py, benchmark.py and nonperiodic_benchmark.py
(based off of Patrick Varilly's benchmarks)

The initilisation time is substantially larger for periodic, but the query time is comparable to the standard KDTree.

Sample periodic benchmarks (time in seconds)
--------------------------------------------

dimensions 3, 10000 points
PeriodicKDTree constructed:	0.07340288162231445
PeriodicKDTree 1000 lookups:	0.00892782211303711
KDTree constructed:	0.002972126007080078
KDTree 1000 lookups:	0.008433818817138672

