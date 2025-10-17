import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import obtain_data_new, scale_data
from save_plotdata import savefile_new
from algorithms import add_noise, std_kmeans, base_coreset, our_coreset

data_name = 'adult'  # n: 48842   dim 6

# data_name = 'census1990'  # n: 2458285  subsample: 100000  dim 68


noise_list = [1, 2, 3, 4, 5]
theta_list = [0.0, 0.01, 0.05, 0.25]
eps_list = [0.1, 0.15, 0.2, 0.25, 0.3]

num_of_cluster = 10
# rep = 20
rep = 10

np.random.seed(0)
data = obtain_data_new(data_name)
X = scale_data(data)

# # subsample 100000 data points for census1990
# samples = np.random.choice(range(data.shape[0]), replace=False, size=100000)
# data_samples = data[samples]
# X = scale_data(data_samples)

n = X.shape[0]
dim = X.shape[1]
print('\n=================================')
print('data_name=', data_name)
print('num_of_data=', n, 'dimension=', dim)

X_kmeans_res = std_kmeans(X, num_of_cluster, rep)
opt_cost = min(X_kmeans_res)
print('kmeans opt=', opt_cost)

for noise in noise_list:
    for theta in theta_list:
        print('\n=================================')
        print('noise=', noise, 'theta=', theta)
        eps_types = len(eps_list)
        base_coreset_eps_res = np.zeros((eps_types, rep))
        base_coreset_size_res = np.zeros((eps_types, rep))
        our_coreset_eps_res = np.zeros((eps_types, rep))
        our_coreset_size_res = np.zeros((eps_types, rep))
        base_r = np.zeros((eps_types, rep))
        our_r = np.zeros((eps_types, rep))
        for t in range(rep):
            X_noise = add_noise(X, noise, theta)
            for eps_t in range(eps_types):
                eps = eps_list[eps_t]

                base_coreset_eps_res[eps_t][t], base_coreset_size_res[eps_t][t], base_r[eps_t][t] = base_coreset(X,
                                                                                                                 X_noise,
                                                                                                                 num_of_cluster,
                                                                                                                 theta,
                                                                                                                 eps,
                                                                                                                 opt_cost)
                our_coreset_eps_res[eps_t][t], our_coreset_size_res[eps_t][t], our_r[eps_t][t] = our_coreset(X, X_noise,
                                                                                                             num_of_cluster,
                                                                                                             theta, eps,
                                                                                                             opt_cost)

        savefile_new(data_name, noise, theta, num_of_cluster, X_kmeans_res, base_coreset_eps_res,
                     base_coreset_size_res, base_r, our_coreset_eps_res, our_coreset_size_res, our_r)

        for eps_t in range(eps_types):
            eps = eps_list[eps_t]
            base_r = (1.0 + eps + (theta * n * dim) / opt_cost + np.sqrt(
                theta * n * dim / opt_cost)) ** 2 - 1.0
            our_r = eps + ((theta * num_of_cluster * dim) / opt_cost) + ((theta * n * dim) / opt_cost)
            print('eps=', eps)
            print('base coreset size=', np.average(base_coreset_size_res[eps_t]), 'std_err=',
                  np.std(base_coreset_size_res[eps_t]))
            print('our coreset size=', np.average(our_coreset_size_res[eps_t]), 'std_err=',
                  np.std(our_coreset_size_res[eps_t]))
            print('base approx ratio=', np.average(1.0 + base_coreset_eps_res[eps_t]), 'std_err=',
                  np.std(1.0 + base_coreset_eps_res[eps_t]))
            print('our approx ratio=', np.average(1.0 + our_coreset_eps_res[eps_t]), 'std_err=',
                  np.std(1.0 + our_coreset_eps_res[eps_t]))
            print('base ratio of ratio=',
                  np.average(base_coreset_eps_res[eps_t] / base_r),
                  'std_err=',
                  np.std(base_coreset_eps_res[eps_t] / base_r))
            print('our ratio of ratio=',
                  np.average(our_coreset_eps_res[eps_t] / our_r),
                  'std_err=',
                  np.std(our_coreset_eps_res[eps_t] / our_r))
            print('base ratio of ratio (+1 version)=',
                  np.average((base_coreset_eps_res[eps_t] + 1.0) / (base_r + 1.0)),
                  'std_err=',
                  np.std((base_coreset_eps_res[eps_t] + 1.0) / (base_r + 1.0)))
            print('our ratio of ratio (+1 version)=',
                  np.average((our_coreset_eps_res[eps_t] + 1.0) / (our_r + 1.0)),
                  'std_err=',
                  np.std((our_coreset_eps_res[eps_t] + 1.0) / (our_r + 1.0)))
