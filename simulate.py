import os
#os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from algorithms import add_noise,std_kmeans_withcenter



def simulate(beta,theta,n):
    Xi = [-2*np.sqrt(2)-beta* np.sqrt(2)/2,-beta* np.sqrt(2)/2,beta* np.sqrt(2)/2,2*np.sqrt(2)+beta* np.sqrt(2)/2]
    X = np.array([Xi[0]]*2500 + [Xi[1]]*2500 + [Xi[2]]*2500 + [Xi[3]]*2500).reshape(-1, 1)
    weights = np.array([1]*10000).reshape(-1,1)
    num_of_cluster = 3
    dim = 1
    opt_cost,centers = std_kmeans_withcenter(X, num_of_cluster,1,n,dim)
    #print("opt_cost:",round(opt_cost, 3))
    #print("opt_center:",centers)
    X_noise = add_noise(X,1,theta)
    opt_cost_noise,centers_noise = std_kmeans_withcenter(X_noise, num_of_cluster,10,n,dim)
    #print("opt_cost_noise:",round(opt_cost_noise, 3))
    #print("opt_center_noise:",centers_noise)

    dists = np.linalg.norm(X[:, np.newaxis] - centers_noise[np.newaxis, :], axis=2)
        # nearest center
    closest = np.argmin(dists, axis=1)
    weak_cost = np.sum((X - centers_noise[closest])**2)
    

    rep_sample = 500
    samples = np.random.uniform(low=min(X_noise), high=max(X_noise), size=(rep_sample, num_of_cluster))

    eps = np.zeros(rep_sample)

    for t in range(rep_sample):
        kmeans = KMeans(n_clusters=num_of_cluster,n_init=3)
        kmeans.cluster_centers_ = samples[t].reshape(-1, 1)

        X_distance_space = kmeans.transform(X)
        X_cost = np.sum((np.min(X_distance_space, axis=1)) ** 2)

        coreset_distance_space = kmeans.transform(X_noise)
        coreset_cost = np.sum(((np.min(coreset_distance_space, axis=1)) ** 2))

        eps[t] = abs(coreset_cost - X_cost) / coreset_cost


    t = np.argmax(eps)

    err_alpha = weak_cost/opt_cost - 1
    err = max(eps)
    #print("beta(/sqrt2):",beta)
    #print("err_alpha:",err_alpha)
    #print("err:",err)
    return err_alpha,err 
    #print("max_err_center",samples[t])

def plot_data(beta_list,err_alpha_list,err_list):
    plt.figure()
    

    plt.plot(beta_list, err_alpha_list, color='red', linestyle='-', label=r"$\mathrm{Err_1(\widehat{P},P)}$")
    plt.plot(beta_list, err_list, color='blue', linestyle='-', label=r"$\mathrm{Err(\widehat{P},P)}$")
    

    plt.xlabel(r"separation level $\beta$", fontsize=12)
    plt.ylabel(r"empirical error $\widehat{\varepsilon}$", fontsize=12)

    plt.xlim(min(beta_list), max(beta_list))
    plt.ylim(0.05, max(max(err_alpha_list), max(err_list)) * 1.02)
    plt.legend()
    
    plt.show()
    plt.savefig("err_err_alpha.png")
    return 0 

if __name__ == '__main__':
    n = 10000
    theta = 1
    beta_list = [2.0 + i * 0.05 for i in range(21)]
    avg_err_alpha_list = []
    avg_err_list = []
    rep = 10
    for beta in beta_list:
        err_alpha_list = []
        err_list = []
        for i in range(rep):
            err_alpha,err = simulate(beta,theta,n)
            err_alpha_list.append(err_alpha)
            err_list.append(err)
        err_alpha = round(sum(err_alpha_list) / len(err_alpha_list), 2)    
        err = round(sum(err_list) / len(err_list),2)
        print("beta(/sqrt2):",beta)
        print("err_alpha:",err_alpha)
        print("err:",err)
        avg_err_alpha_list.append(err_alpha)
        avg_err_list.append(err)
    
    plot_data(beta_list,avg_err_alpha_list,avg_err_list)






