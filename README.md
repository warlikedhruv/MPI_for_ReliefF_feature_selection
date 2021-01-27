# MPI_for_ReliefF_feature_selection
This code uses MPI for python for parallel processing with a Master controlling the slaves for the work.


Uses MPI for windows 10 
to set the path : PATH=%PATH%;C:\Program Files\Microsoft MPI\bin in the CMd terminal 
to run the file : mpiexec -n 4 python <filename>.py where -n 4 are the number of cores you want to assign.

the code Reads the data from the file:
line:1 is for number of (master + slave) you want
line:2 is for the N, A, M, T where n is no. of instance , A is the no. of attribute , M is the number of iterations and T as the top features you want.
rest lines are the data for the algorithm.

The master will assing the work to the Slaves and then master will go to sleep and again assing the work to slaves till the queue is not finished.

the algorithm for the Relief algorithm is : https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83
