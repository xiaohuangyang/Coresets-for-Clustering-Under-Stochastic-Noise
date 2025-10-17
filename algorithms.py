import numpy as np
from sklearn.cluster import KMeans


def add_noise(X, noise, theta):
    n = X.shape[0]
    dim = X.shape[1]
    X_noise = np.empty_like(X)
    if noise == 1:
        n_noises = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), size=n)
        zeros = np.zeros(dim)
        probs = np.random.binomial(1, 1 - theta, size=n)
        n_noises[np.nonzero(probs)] = zeros
        X_noise = X + n_noises
    elif noise == 2:
        X_noise = X + np.random.multivariate_normal(np.zeros(dim), theta * np.identity(dim), size=n)
    elif noise == 3:
        n_noises = np.empty_like(X)
        for i in range(n):
            n_noises[i] = np.random.laplace(0, 1 / np.sqrt(2), size=dim)
        zeros = np.zeros(dim)
        probs = np.random.binomial(1, 1 - theta, size=n)
        n_noises[np.nonzero(probs)] = zeros
        X_noise = X + n_noises
    # elif noise == 4:
    #     for i in range(n):
    #         X_noise[i] = X[i] + np.random.laplace(0, np.sqrt(theta / 2), size=dim)
    elif noise == 4:
        n_noises = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(n, dim))
        zeros = np.zeros(dim)
        probs = np.random.binomial(1, 1 - theta, size=n)
        n_noises[np.nonzero(probs)] = zeros
        X_noise = X + n_noises
    # elif noise == 6:
    #     n_noises = np.random.uniform(low=-np.sqrt(3 * theta), high=np.sqrt(3 * theta), size=(n, dim))
    #     X_noise = X + n_noises
    else:
        A = np.random.uniform(low=-1.0, high=1.0, size=(dim, dim))
        Sigma = A @ A.T
        Sigma = dim / np.trace(Sigma) * Sigma
        X_noise = X + np.random.multivariate_normal(np.zeros(dim), theta * Sigma, size=n)
    return X_noise


def add_noiseI(X, theta):
    n = X.shape[0]
    dim = X.shape[1]
    n_noises = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), size=n)
    zeros = np.zeros(dim)
    probs = np.random.binomial(1, 1 - theta, size=n)
    n_noises[np.nonzero(probs)] = zeros
    X_noise = X + n_noises
    return X_noise


def add_noiseII(X, theta):
    n = X.shape[0]
    dim = X.shape[1]
    X_noise = X + np.random.multivariate_normal(np.zeros(dim), theta * np.identity(dim), size=n)
    return X_noise


def add_noiseIII(X, theta):
    n = X.shape[0]
    dim = X.shape[1]
    X_noise = np.empty_like(X)
    for i in range(n):
        X_noise[i] = X[i] + np.random.laplace(0, theta, size=dim)
    return X_noise


def kmeans_opt(X, k):
    rep = 10
    res = np.zeros(rep)
    for t in range(rep):
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(X)
        res[t] = kmeans.inertia_
    return min(res)


def std_kmeans(X, k, rep):
    res = np.zeros(rep)
    for t in range(rep):
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(X)
        res[t] = kmeans.inertia_
    return res

def std_kmeans_withcenter(X, k, rep,n,dim):
    res = np.zeros(rep)
    centers = np.zeros((rep,k,dim))
    labels = np.zeros((rep,n))
    opt_t = 0
    for t in range(rep):
        #n init
        kmeans = KMeans(n_clusters=k,n_init=10).fit(X)
        res[t] = kmeans.inertia_ 
        centers[t] = kmeans.cluster_centers_ #(num_cluster,dim)
        labels[t] = kmeans.labels_ #(n,)
        if res[t]<res[opt_t]:
            opt_t = t
    return res[opt_t],centers[opt_t]

def noise_kmeans(X, X_noise, k, rep):
    res = np.zeros(rep)
    for t in range(rep):
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(X_noise)

        distance_space = kmeans.transform(X)
        X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
        res[t] = X_cost
    return res


def get_coreset(X, k, m):
    n = X.shape[0]
    alpha = 2
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(X)
    cost = kmeans.inertia_ / n
    cluster_count = np.zeros(k)
    cluster_cost = np.zeros(k)
    label = kmeans.labels_
    distance_space = kmeans.transform(X)
    for i in range(n):
        cluster_count[label[i]] += 1
        cluster_cost[label[i]] += distance_space[i, label[i]] ** 2
    cluster_cost = np.maximum(cluster_cost, np.finfo(cluster_cost.dtype).eps)
    sensitivity = np.zeros(n)
    for i in range(n):
        # sensitivity[i] = (alpha * (distance_space[i, label[i]] ** 2) / cost) \
        #                 + 2 * alpha * cluster_cost[label[i]] / (cluster_count[label[i]] * cost) \
        #                 + 4 * n / cluster_cost[label[i]]
        sensitivity[i] = 0.25 * (1.0 / (k * cluster_count[label[i]]) \
                                 + distance_space[i, label[i]] ** 2 / (k * cluster_cost[label[i]]) \
                                 + distance_space[i, label[i]] ** 2 / (n * cost) \
                                 + cluster_cost[label[i]] / (n * k * cost))
    sensitivity_sum = np.sum(sensitivity)
    sensitivity /= sensitivity_sum
    ent = - np.sum(sensitivity * np.log(sensitivity))
    # print("total sensitivity: %f, sensitivity entropy: %f" % (sensitivity_sum, ent))
    samples = np.random.choice(range(n), replace=False, size=m, p=sensitivity)
    coreset = X[samples]
    weights = 1 / (m * sensitivity[samples])
    return coreset, weights


def coreset_kmeans(coreset, weights, X, k, rep):
    res = np.zeros(rep)
    for t in range(rep):
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(coreset, sample_weight=weights)

        distance_space = kmeans.transform(X)
        X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
        res[t] = X_cost
    return res


def coreset_eps(coreset, weights, X, k, rep, rep_sample):
    n = X.shape[0]
    eps = np.zeros(rep)
    for t in range(rep):
        for r in range(rep_sample):
            samples = np.random.choice(range(n), replace=False, size=k)
            kmeans = KMeans(n_clusters=k, n_init='auto')
            kmeans.cluster_centers_ = X[samples]

            X_distance_space = kmeans.transform(X)
            X_cost = np.sum((np.min(X_distance_space, axis=1)) ** 2)

            coreset_distance_space = kmeans.transform(coreset)
            coreset_cost = np.sum(((np.min(coreset_distance_space, axis=1)) ** 2) * weights)

            cost_eps = abs(coreset_cost - X_cost) / X_cost
            eps[t] = max(eps[t], cost_eps)
    return eps


def weak_coreset_eps(coreset, weights, X, k, rep, opt_cost):
    eps = np.zeros(rep)
    for t in range(rep):
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(coreset, sample_weight=weights)
        distance_space = kmeans.transform(X)
        X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
        eps[t] = (X_cost - opt_cost) / opt_cost
    return eps


def coreset_eps_new(coreset, weights, X, k, rep_sample, samples):
    eps = np.zeros(rep_sample)
    for t in range(rep_sample):
        center_sample = samples[t]
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.cluster_centers_ = X[center_sample]

        X_distance_space = kmeans.transform(X)
        X_cost = np.sum((np.min(X_distance_space, axis=1)) ** 2)

        coreset_distance_space = kmeans.transform(coreset)
        coreset_cost = np.sum(((np.min(coreset_distance_space, axis=1)) ** 2) * weights)

        eps[t] = abs(coreset_cost - X_cost) / X_cost
    return max(eps)


def weak_coreset_eps_new(coreset, weights, X, k, opt_cost):
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(coreset, sample_weight=weights)
    distance_space = kmeans.transform(X)
    X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
    eps = (X_cost - opt_cost) / opt_cost
    return eps


def uniform_coreset(X, X_noise, k, theta, eps, opt_cost):
    n = X.shape[0]
    dim = X.shape[1]
    m = int(k / ((eps - (theta * n * dim / opt_cost)) ** 2))
    if m >= n:
        return 0, n
    samples = np.random.choice(range(n), replace=False, size=m)
    coreset = X_noise[samples]
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(coreset)
    distance_space = kmeans.transform(X)
    X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
    eps_hat = (X_cost - opt_cost) / opt_cost
    return eps_hat, m


def base_coreset(X, X_noise, k, theta, eps, opt_cost):
    n = X.shape[0]
    dim = X.shape[1]
    m = int(3 * np.power(k, 1.5) / (eps ** 2))
    coreset, weights = get_coreset(X_noise, k, m)
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(coreset, sample_weight=weights)
    distance_space = kmeans.transform(X)
    X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
    eps_hat = (X_cost - opt_cost) / opt_cost
    base_r = (1.0 + eps + (theta * n * dim) / opt_cost + np.sqrt(theta * n * dim / opt_cost)) ** 2 - 1.0
    return eps_hat, m, base_r


def our_coreset(X, X_noise, k, theta, eps, opt_cost):
    n = X.shape[0]
    dim = X.shape[1]
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(X_noise)
    cluster_count = np.zeros(k)
    coreset_count = np.zeros(k)
    cluster_cost = np.zeros(k)
    cluster_indices = [[] for _ in range(k)]
    label = kmeans.labels_
    distance_space = kmeans.transform(X)
    for i in range(n):
        cluster_count[label[i]] += 1
        cluster_indices[label[i]].append(i)
        cluster_cost[label[i]] += distance_space[i, label[i]] ** 2
    S_indices = []
    for cluster_i in range(k):
        radius_i = (np.sqrt(cluster_cost[cluster_i] / cluster_count[cluster_i]) + np.sqrt(dim) * np.log(
            10 * (1 + theta * k * dim)))
        to_remove = []
        for index in cluster_indices[cluster_i]:
            if distance_space[index, label[index]] > radius_i:
                to_remove.append(index)
        for index in to_remove:
            cluster_indices[cluster_i].remove(index)
        cluster_i_newsize = len(cluster_indices[cluster_i])
        size_i = int(9.0 / eps + 6.0 / (eps ** 2))
        if size_i >= cluster_i_newsize:
            S_indices.extend(cluster_indices[cluster_i])
            coreset_count[cluster_i] = cluster_i_newsize
        else:
            samples = np.random.choice(range(cluster_i_newsize), replace=False, size=size_i)
            S_indices.extend([cluster_indices[cluster_i][j] for j in samples])
            coreset_count[cluster_i] = size_i
    coreset = X_noise[S_indices]
    S_num = len(S_indices)
    weights = np.zeros(S_num)
    for i in range(S_num):
        weights[i] = len(cluster_indices[label[i]]) / coreset_count[label[i]]
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(coreset, sample_weight=weights)
    distance_space = kmeans.transform(X)
    X_cost = np.sum((np.min(distance_space, axis=1)) ** 2)
    eps_hat = (X_cost - opt_cost) / opt_cost
    our_r = eps + ((theta * k * dim) / opt_cost) + ((theta * n * dim) / opt_cost)
    return eps_hat, S_num, our_r
