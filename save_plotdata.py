import os
import numpy as np
import pandas as pd
import csv


def savefile(data_name, noise, theta, coreset_size, X_kmeans_res, coreset_eps_res, weak_coreset_eps_res):
    path = './data/' + data_name + '/plotdata/'
    if not os.path.exists(path):
        os.makedirs(path)

    pathfile = path + 'noise_' + str(noise) + '_theta_' + str(theta) + '_coreset_size_' + str(coreset_size)
    res = np.stack((X_kmeans_res, coreset_eps_res, weak_coreset_eps_res))
    df = pd.DataFrame(res)
    df.to_csv(pathfile + '.csv', encoding='utf-8', index=False)


def savefile_new(data_name, noise, theta, num_of_cluster, X_kmeans_res, base_coreset_eps_res,
                 base_coreset_size_res, base_r, our_coreset_eps_res, our_coreset_size_res, our_r):
    # path = './data/' + data_name + '/plotdata/'
    path = './data/' + data_name + '/plotdata_' + str(num_of_cluster) + 'clusters/'
    if not os.path.exists(path):
        os.makedirs(path)

    pathfile = path + 'noise_' + str(noise) + '_theta_' + str(theta)
    res = np.vstack(
        (X_kmeans_res, base_coreset_eps_res, base_coreset_size_res, base_r, our_coreset_eps_res, our_coreset_size_res,
         our_r))
    df = pd.DataFrame(res)
    df.to_csv(pathfile + '.csv', encoding='utf-8', index=False)
