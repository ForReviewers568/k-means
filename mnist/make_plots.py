import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots

plt.style.use('science')
plt.rcParams["mathtext.fontset"] = "cm"

n = []
eps_n = []

with open('data_n_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line: 
			if float((line.strip('\n').split(" "))[0]) not in n:
				n.append( float((line.strip('\n').split(" "))[0]) )
			if float((line.strip('\n').split(" "))[1]) not in eps_n:
	    			eps_n.append( float((line.strip('\n').split(" "))[1]) )

data_n = {
	'time': [[] for i in range(len(eps_n))],
	'iterations': [[] for i in range(len(eps_n))],
	'RSS': [[] for i in range(len(eps_n))],
	'ARI': [[] for i in range(len(eps_n))],
	'NMI': [[] for i in range(len(eps_n))],
}

with open('data_n_final.txt', "r") as data_file:
	for line in data_file:
		if 'iterations' not in line:
			n_value = float((line.strip('\n').split(" "))[0])
			eps_value = float((line.strip('\n').split(" "))[1])
			
			i = 0
			while eps_n[i] != eps_value:
				i += 1 
			
			time = float((line.strip('\n').split(" "))[2])
			data_n['time'][i].append( time / 1000 )
			
			iterations = float((line.strip('\n').split(" "))[3])
			data_n['iterations'][i].append( iterations )
			
			if eps_value != 0.0:
				RSS = float((line.strip('\n').split(" "))[4])
				ARI = float((line.strip('\n').split(" "))[5])
				NMI = float((line.strip('\n').split(" "))[6])
				data_n['RSS'][i].append( RSS )
				data_n['ARI'][i].append( ARI )
				data_n['NMI'][i].append( NMI )
	 

###############################################################################

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Total runtime (s)', fontsize=20)
plt.xlim([20000,50000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_time.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
#plt.plot(N, data['time'][0], label='Standard')
ax.plot(n, data_n['time'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['time'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['time'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
plt.yscale("log")
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Total runtime (s)', fontsize=20)
plt.xlim([20000,50000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_time_log.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('time_log.png', bbox_inches='tight')
plt.close()


plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['RSS'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['RSS'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['RSS'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['RSS'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['RSS'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Residual Sum of Squares (RSS) difference (\%)', fontsize=20)
plt.xlim([20000,50000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_RSS.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['ARI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['ARI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['ARI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['ARI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['ARI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=20)
plt.xlim([20000,50000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_ARI.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['NMI'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['NMI'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['NMI'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['NMI'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['NMI'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Normalized Mutual Information (NMI)', fontsize=20)
plt.xlim([20000,50000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_NMI.pdf', format='pdf', bbox_inches='tight')
plt.close()

plt.rcParams.update({'font.size': 13})
fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
plt.grid(color='gray', linestyle='dotted', linewidth=0.1)
ax.plot(n, data_n['iterations'][0], label=r'$\varepsilon$ = '+str(eps_n[0]), color='k', marker='o', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][1], label=r'$\varepsilon$ = '+str(eps_n[1]), marker='v', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][2], label=r'$\varepsilon$ = '+str(eps_n[2]), marker='^', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][3], label=r'$\varepsilon$ = '+str(eps_n[3]), marker='s', linewidth=2.0, markersize=9)
ax.plot(n, data_n['iterations'][4], label=r'$\varepsilon$ = '+str(eps_n[4]), marker='x', linewidth=2.0, markersize=11)
ax.plot(n, data_n['iterations'][5], label=r'$\varepsilon$ = '+str(eps_n[5]), marker='*', linewidth=2.0, markersize=11)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlabel('Dataset size $n$', fontsize=20)
ax.set_ylabel('Number of iterations', fontsize=20)
plt.xlim([20000,50000])
legend = ax.legend(loc='upper left', prop={'size': 13}, shadow=True, frameon=True)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_facecolor('white')
#plt.show()
plt.savefig('plots/n_iterations.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('RSS.png', bbox_inches='tight')
plt.close()


###############################################################################

