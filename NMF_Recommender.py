import numpy as np
import random

"""
Non-negative Matrix Factorization using Alternating Least Squares
By Frank Longueira, Max Howald, Miraj Patel
"""


"""
nmf_ALS: Finds a decomposition of matrix R into a matrix product of P & Q that minimizes an objective function
		(being sure to remove test observations & unknown values) using alternating least squares optimization
		
	INPUT:
   		R     : a matrix to be factorized, dimension N x M
    	P     : an initial matrix of dimension N x f
    	Q     : an initial matrix of dimension M x f
    	f     : the number of latent features
    	alpha : the learning rate
    	h  : the regularization parameter
	OUTPUT:
   		the final matrices P and Q
"""
def nmf_ALS(R, P, Q, f, test_observations, sweeps= 10, h=0.01):
	
	hI = h*np.eye(f,f)
	
	Cu = np.zeros((len(R[0]),len(R[0])), dtype = "int32")
	for sweep in xrange(sweeps):
		print "Sweep: ", sweep + 1
		
	# Users greedy optimization
		# Removal of unknown values and test observation
		for i in xrange(len(R)):
			for j in xrange(len(R[0])):
				if (i,j,R[i,j]) in test_observations:
					Cu[j,j] = 0
				elif(R[i,j] > 0):
					Cu[j,j] = 1
				else:
					Cu[j,j] = 0
			Q_T = Q.T
			
			# Computes user latent features using closed form solution
			P[i,:] = (np.linalg.inv((np.dot(np.dot(Q_T,Cu),Q) + hI))*np.dot(np.dot(Q_T,Cu),np.matrix(R[i,:]).T)).T
		
	# Items greedy optimization
		Ci = np.zeros((len(R),len(R)), dtype = "int32")
		for j in xrange(len(R[0])):
			for i in xrange(len(R)):
				if (i,j,R[i,j]) in test_observations:
					Ci[i,i] = 0
				elif(R[i,j] > 0):
					Ci[i,i] = 1
				else:
					Ci[i,i] = 0
			P_T = P.T
			
			# Computes user latent features using closed form solution
			Q[j,:] = (np.linalg.inv((np.dot(np.dot(P_T,Ci),P) + hI))*np.dot(np.dot(P_T,Ci),np.matrix(R[:,j]).T)).T

	return P, Q


"""
Test:
	INPUT:
   		observations: dictionary with key:value pairs (i, (user_id, movie_id, rating)) for the ith rating observation in trainging set
    	test_observation: dictionary with key:value pairs ((user_id, movie_id, rating), i) where (user_id, movie_id, rating) is a test observation found in observations with key i
    	R_approx: this is the approximated matrix found by the product of matrices, P & Q, returned by nmf_ALS
	OUTPUT:
   		avg_error: average absolute error on a test observation
"""
def test_nmf(observations, test_observations, R_approx):
	k = 0
	error = 0

	for i in xrange(len(observations)):
		(user_id, movie_id, rating) = observations[i]
		if (user_id, movie_id, rating) in test_observations:
			rating_approx = R_approx[user_id, movie_id]
			print "Test Observation", k+1, ":", (rating, rating_approx)
			k += 1
			error += abs(rating-rating_approx)
	avg_error = float(error)/len(test_observations)
	
	return avg_error

# Main Script Below

# Load in movie ratings data (from MovieLens) and determine number of users, movies, and ratings
file_name = "ratings.dat"
data = np.genfromtxt(file_name, delimiter = "::", dtype = "int32")
num_users = np.amax(data[:,0])
num_movies = np.amax(data[:,1])
num_ratings = np.shape(data)[0]

# Put movie ratings data into a User-Movie ratings matrix
all_ratings = np.zeros((num_users, num_movies), dtype = "int32")
for i in xrange(num_ratings):
	user_id = data[i, 0] - 1
	movie_id = data[i, 1] - 1
	rating =data[i, 2]
	all_ratings[user_id, movie_id] = rating;

# Reduce matrix size for computation feasibility. 
# We will use R from now on as our matrix to be factored.
num_users_red = num_users/10
num_movies_red = num_movies/10

R = all_ratings[0:num_users_red,0:num_movies_red]

# Store a dictionary that maps a rating observation to it's location in the matrix w/ corresponding rating (to be used for testing)
observations = {}
k = 0
for i in xrange(len(R)):
	for j in xrange(len(R[i])):
		if(R[i,j] > 0):
			observations[k] = (i, j, R[i,j])
			k += 1	
step = 10

# Store a dictionary of test set observations
test_observations = {observations[i]:i for i in xrange(0, len(observations), step)}

# Number of latent features to use
f = 50

# Initialize P & Q for ALS
P = np.random.rand(num_users_red,f)
Q = np.random.rand(num_movies_red,f)

# Number of sweeps to do during alternating least squares
sweeps = 10

# Greedily found regularization parameter
h = 5

print "Performing nonnegative matrix factorization using ALS..."
(P, Q) = nmf_ALS(R, P, Q, f, test_observations, sweeps, h)
print "Finished performing nonnegative matrix factorization using ALS..."
R_approx = np.dot(P, Q.T)
Abs_Error = test_nmf(observations, test_observations, R_approx)
print "Test Error = ", Abs_Error
	
