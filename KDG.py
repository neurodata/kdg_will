import numpy as np
#from joblib import Parallel, delayed
from keras.utils import to_categorical
from tqdm import tqdm
from scipy.stats import multivariate_normal
from sklearn.base import ClassifierMixin, BaseEstimator
from itertools import product

class KDG(ClassifierMixin, BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y, predicted_leaf_ids_across_trees):
        #an array of all of the X-values of the class-wise means in each leaf
        self.polytope_means_X = []

        #an array of all of the y-values (i.e. class values) of the class-wise means in each leaf
        self.polytope_means_y = []

        #an array of all of the number of points that comprise the means in each leaf
        self.polytope_means_weight = []

        #an array of all of the average variances of the points in each leaf corresponding to 
        #a single class from the class-wise mean in that leaf
        self.polytope_means_cov = []
        
        unique_polytope_ids_across_trees = [np.unique(predicted_leaf_ids_across_trees[i]) for i in range(len(predicted_leaf_ids_across_trees))]
        
        for tree_id, unique_polytopes in enumerate(unique_polytope_ids_across_trees):
            for polytope_id in unique_polytopes:
                for y_val in np.unique(y):
                    idxs_in_polytope_of_class = np.where((y == y_val) & (predicted_leaf_ids_across_trees[tree_id] == polytope_id))[0]
                    if len(idxs_in_polytope_of_class) > 1:
                        mean = np.mean(X[idxs_in_polytope_of_class], axis = 0)
                        self.polytope_means_X.append(mean)
                        #we already know the y value, so just append it 
                        self.polytope_means_y.append(y_val)
                        #compute the number of points in that leaf corresponding to that y value
                        #and append to the aggregate array
                        self.polytope_means_weight.append(len(idxs_in_polytope_of_class))
                        #compute the distances of all the points in that leaf corresponding to that y value
                        #from the mean X-value of the points in that leaf corresponding to that y value
                        #compute the covariance as the average distance of the class-wise points in that leaf 
                        #and append to the aggregate array
                        cov = np.mean((X[idxs_in_polytope_of_class] - mean) ** 2, axis = 0)
                        self.polytope_means_cov.append(np.eye(len(cov)) * cov)
        return self
                    
    def predict_proba(self, X, pooling = "class"):
        y_vals = np.unique(self.polytope_means_y)
        y_hat = np.zeros((len(X), len(y_vals)))
        
        def compute_pdf(X, polytope_mean_id):
            polytope_mean_X = self.polytope_means_X[polytope_mean_id]
            polytope_mean_y = self.polytope_means_y[polytope_mean_id]
            polytope_mean_weight = self.polytope_means_weight[polytope_mean_id]
            if pooling == "polytope":
                covs_of_class = np.array(self.polytope_means_cov)[np.where(polytope_mean_X ==self.polytope_means_X)[0]]
                weights_of_class = np.array(self.polytope_means_weight)[np.where(polytope_mean_X ==self.polytope_means_X)[0]]
            else:
                covs_of_class = np.array(self.polytope_means_cov)[np.where(polytope_mean_y ==self.polytope_means_y)[0]]
                weights_of_class = np.array(self.polytope_means_weight)[np.where(polytope_mean_y ==self.polytope_means_y)[0]]
            polytope_mean_cov = np.average(covs_of_class, weights = weights_of_class, axis = 0)
            var = multivariate_normal(mean=polytope_mean_X, cov=polytope_mean_cov, allow_singular=True)
            return var.pdf(X) * polytope_mean_weight / np.sum(weights_of_class)
        
        '''  
        likelihoods = Parallel(n_jobs=10, verbose=1)(delayed(compute_pdf)(
            X, polytope_mean_id
        ) for polytope_mean_id in range(len(self.polytope_means_X)))
        '''
        likelihoods = np.array([compute_pdf(X, polytope_mean_id) for polytope_mean_id in range(len(self.polytope_means_X))])
        for polytope_id in range(len(likelihoods)):
            y_hat[:, self.polytope_means_y[polytope_id]] += likelihoods[polytope_id]
        return y_hat
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)
                          
