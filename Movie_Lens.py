
# coding: utf-8

# Input for the function to predict the ratings are as follows:
# 
#     R - The user-movie rating matrix
#     K - Number of latent features
#     alpha - Learning rate of stochastic gradient descent
#     beta - Regularization parameter for bias
#     iterations - Number of iterations to perfrom stochastic gradient   descent.

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 


# In[2]:


#users
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names = u_cols, encoding = 'latin-1' )

#ratings
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep = '\t', names = r_cols, encoding = 'latin-1')

#items
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')


# In[7]:


#dividing the dataset into train and test, already done by grouplens before.
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape


# In[10]:


#Building user-user similarity and item-item similarity:

#Calculates number of unique movie and users:
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

#Creating a user-data matrix:
data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    #subtracting to make the starting index as 0 for the df.
    data_matrix[line[1]-1, line[2]-1] = line[3]
    
#using pairwise function from sklearn to calculate cosine similarity:

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix.T, metric = 'cosine')

# I.E, the item-item and user-user is now in array form, now to make
# predictions on these similarities.


# In[11]:


def predict(ratings, similarity, type = 'user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #Using np.newaxis to have same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        
        # takes mean in column 1, then normalizing it with ratings_diff
        # making predictions based on similarity of users
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
        
    elif type == 'item':
        # making predictions based on item similarity
        pred = ratings.dot(similarity)/np.array([np.abs(similarity).sum(axis=1)])
    return pred
        


# In[20]:


import turicreate 

# Creating SFrames for the testing datasets
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)

#Building a popularity model first:

popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target= 'rating')

#Recommending top 5 for first 5 users in dataset

popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5], k=5)
#popularity_recomm.print_rows(num_rows=25)

#Training the model

item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id = 'user_id', item_id = 'movie_id', target = 'rating', similarity_type='cosine')

#Making recommendations for first five users

item_sim_recomm = item_sim_model.recommend(users = [1,2,3,4,5], k = 5)
#item_sim_recomm.print_rows(num_rows = 25)


# In[53]:


# Recommendation engine using matrix factorization.
# Defining a function to predict ratings givesn by the user to all movies 
# which are not rated by him.

class MF():

    #Initialising user-movie rating matrix, number of latent features, alpha and beta
    
    def __init__(self, R, K, alpha, beta, iterations):
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
        
    # Initialising user-feature and movie-feature matrix
    
    def train(self):
        self.P = np.random.normal(scale = 1./self.K, size = (self.num_users, self.K))
        self.Q = np.random.normal(scale = 1./self.K, size = (self.num_items, self.K))
        
        # Initialising the bias terms
        self.b_u = np.zeros(self.num_users) #Bias for users
        self.b_i = np.zeros(self.num_items) #Bias for items
        self.b = np.mean(self.R[np.where(self.R !=0)])
        
        
        # Listing the training samples:
        
        self.samples = [
            
            (i,j,self.R[i,j]) for i in range(self.num_users) for j in range(self.num_items) if self.R[i,j]>0
        ]
    
    
        # Stochastic Gradient Descent for the given number of iterations:
        
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) %20 ==0:
                print( "Iteration %d ; error = %.4f" %(i+1, mse))
                
        return training_process
    
    # Computing total mean squared error
    
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x,y in zip(xs, ys):
            error += pow(self.R[x,y] - predicted[x,y], 2)
        return np.sqrt(error)
    
    
    # Stochastic gradient descent to get optimized for P and Q matrix
    
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i,j)
            e = (r-prediction)
            
            self.b_u[i] += self.alpha * (e-self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e -self.beta *self.b_i[j])
            
            #Applying Update rule formula: 
            
            self.P[i, :] += self.alpha * (e *self.Q[j,:] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
            
            
            # Ratings for user i and movie J
            
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i,:].dot(self.Q[j,:].T)

        return prediction

    #Full user-movie rating matrix:

    def full_matrix(self):
        return mf.b + mf.b_u[:, np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)








    
    
    


# In[54]:


# Converting the user-item ratings to matrix form. 

R = np.array(ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0))


# Predicting all missing ratings. 
# K = 20, alpha = 0.001, beta = 0.01, iterations = 100

mf = MF(R, K=20, alpha = 0.001, beta = 0.01, iterations = 100)
training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()
print("Done")

