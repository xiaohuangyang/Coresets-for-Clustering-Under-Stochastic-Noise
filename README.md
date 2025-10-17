# Coresets for Clustering Under Stochastic Noise

Code for Coresets for Clustering Under Stochastic Noise(NeurIPS 2025)

Run "python main.py" to reproduce the empirical results.

For the main experiment in Sec 4, set data_name = 'adult' and noise list = [1]. The result is shown in Table 1 in our paper.

For the experiment with Census1990 dataset in Sec G.2, set data_name = 'census1990' and noise list = [1]. The result is shown in Table 3 in our paper.

For the experiments under different noise model in Sec G.3, set data_name = 'adult' and noise list = [2,3,4,5]. The result is shown in Table 4-7 in our paper.

Addtionally, run "python simulate.py" to reproduce the results in Sec E.1, which further compare the behavior of two error measure.

The datasets used are listed in the 'data' folder. All experiments are conducted using Python 3.11 on an Apple M3 Pro machine with an 11-core CPU, 14-core GPU, and 36 GB of memory.
