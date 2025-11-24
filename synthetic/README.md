Raw data: data_n_final.txt, data_d_final.txt, data_k_final.txt

The columns are: size of dataset / dimension / number of clusters, epsilon, time, number of iterations, RSS, ARI, NMI

The first line starting with "iterations" counts the number of trials that were used in computing time, number of iterations, RSS, ARI, and NMI, respectively.



Main c++ file: q_means_n_variable.cpp computes the dependence on the dataset size n, q_means_d_variable.cpp computes the dependence on the dimension d, q_means_k_variable.cpp computes the dependence on the number of clusters k, q_means_sampling_time.cpp computes the dependence of sampling on the fianl runtime.

"make_plots.py" reads the raw data files and make the plots from them.
