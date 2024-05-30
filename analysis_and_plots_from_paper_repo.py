# -*- coding: utf-8 -*-
"""analysis_and_plots_from_paper_repo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gAHNUXO63RXNrrYAvwXio0zZgUDG66fQ
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,NullFormatter
from matplotlib.colors import LogNorm, LinearSegmentedColormap

#results_d_best,results_d_best_best=pickle.load(open('simulations_all_results_filtered_6Nov2019.pkl','rb'))
#results_d_best,results_d_best_best=pickle.load(open('simulations_all_results_filtered_10Nov2019.pkl','rb'))

'''
results_d_best is a dictionary indexed by d, h, and the sat
configs. It contains all sat configs that did not have any time gaps
in 24 hours.

results_d_best_best is a dictionary indexed by d and h. It contains
the sat config with the least number of sats such that there is no
time gap in 24 hours.
'''

results_d_best,results_d_best_best=pickle.load(open('simulations_all_results_filtered_12Nov2019.pkl','rb'))

H=list(results_d_best_best[1000].keys()) # List of altitudes
D=list(results_d_best_best.keys()) # List of distances

results_d_best[1000][1500]

N={}  # Least number of satellites (as a list as a function of h)
R={}  # Rates (as a list as a function of h, corresponding to the least number of satellites)
L={}  # Loss (as a list as a function of h, corresponding to the least number of satellites)
for d in D:
    N[d]=[]
    R[d]=[]
    L[d]=[]
    for h in H:
        if not results_d_best_best[d][h]:
            N[d].append(None)
            R[d].append(None)
            L[d].append(None)
        else:
            N[d].append(np.prod(list(results_d_best_best[d][h][0][0])))
            R[d].append(results_d_best_best[d][h][0][1][3])
            L[d].append(results_d_best_best[d][h][0][1][2])
    #N[d]=[np.prod(list(results_d_best_best[d][h][0][0])) for h in H]

def calculate_cost(d,h,alpha=1):

    '''
    For a given d and h, returns a dictionary indexed by sat configs
    that contains the cost function. Also returns the minimum value
    of the cost function among all available sat configs.
    '''

    C={}

    for sat_config in results_d_best[d][h].keys():
        n=np.prod(list(sat_config))
        eta=results_d_best[d][h][sat_config][2]
        C[sat_config]=n**(1/alpha)*eta

    return C,min(C.values())

def calculate_FOM(d,h):

    '''
    For a given d and h, returns a dictionary indexed by sat configs
    that contains the figure of merit (rate divided by total number
    of satellites). Also returns the maximum value of the figure of
    merit among all available sat configs, and the sat config that
    achieves the maximum.
    '''

    F={}

    for sat_config in results_d_best[d][h].keys():
        n=np.prod(list(sat_config))
        r=results_d_best[d][h][sat_config][3]
        F[sat_config]=r/n

    if not F:
        m=None
        s=None
    else:
        Fval=np.array(list(F.values()))
        Fkey=list(F.keys())
        m=max(Fval)
        i=np.argwhere(Fval==m)[0][0]
        s=Fkey[i]

    return F,m,s

calculate_FOM(1500,2000)

calculate_FOM(1500,2000)[1]

def calculate_rate(d,h):

    '''
    For a given d and h, returns a dictionary indexed by sat configs
    that contains the rate. Also returns the maximum rate among all
    available sat configs.
    '''

    R={}

    for sat_config in results_d_best[d][h].keys():
        R[sat_config]=results_d_best[d][h][sat_config][3]

    if not R:
        m=None
    else:
        m=max(R.values())

    return R,m

fig=plt.figure(figsize=(3.5,3))

ax=fig.add_subplot(111)

ax.plot(H,N[500],linestyle='-',marker='^',label=r'$d=500$ km')
ax.plot(H,N[1500],linestyle='-',marker='o',label=r'$d=1500$ km')
ax.plot(H,N[2500],linestyle='-',marker='P',label=r'$d=2500$ km')
ax.plot(H,N[3500],linestyle='-',marker='s',label=r'$d=3500$ km')
ax.plot(H,N[4500],linestyle='-',marker='*',label=r'$d=4500$ km')
ax.plot(H,N[5000],linestyle='-',marker='d',label=r'$d=5000$ km')

ax.set_yscale('log')
ax.get_yaxis().set_minor_formatter(NullFormatter())
ax.set_yticks([30,40,50,60,80,100,150,200,300,400])
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.minorticks_off()

plt.xticks([0,2000,4000,6000,8000,10000],fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel('$h$ (km)',fontsize=12)
ax.set_ylabel('$N_{\mathrm{opt}}$',fontsize=12)


#plt.ylim([20,100])

#ax.grid()

#plt.legend(fontsize=12)

plt.tight_layout()

#plt.savefig('simulations_all_numSats_13Nov2019.pdf',bbox_inches='tight')
plt.savefig('simulations_all_numSats_22July2020.pdf',bbox_inches='tight')

plt.show()

F_500=[calculate_FOM(500,h)[1] for h in H]
F_1000=[calculate_FOM(1000,h)[1] for h in H]
F_1500=[calculate_FOM(1500,h)[1] for h in H]
F_2000=[calculate_FOM(2000,h)[1] for h in H]
F_2500=[calculate_FOM(2500,h)[1] for h in H]
F_3000=[calculate_FOM(3000,h)[1] for h in H]
F_3500=[calculate_FOM(3500,h)[1] for h in H]
F_4000=[calculate_FOM(4000,h)[1] for h in H]
F_4500=[calculate_FOM(4500,h)[1] for h in H]
F_5000=[calculate_FOM(5000,h)[1] for h in H]


S_500=[calculate_FOM(500,h)[2] for h in H]
S_1000=[calculate_FOM(1000,h)[2] for h in H]
S_1500=[calculate_FOM(1500,h)[2] for h in H]
S_2000=[calculate_FOM(2000,h)[2] for h in H]
S_2500=[calculate_FOM(2500,h)[2] for h in H]
S_3000=[calculate_FOM(3000,h)[2] for h in H]
S_3500=[calculate_FOM(3500,h)[2] for h in H]
S_4000=[calculate_FOM(4000,h)[2] for h in H]
S_4500=[calculate_FOM(4500,h)[2] for h in H]
S_5000=[calculate_FOM(5000,h)[2] for h in H]

calculate_FOM(5000,3500)

results_d_best[5000][2500]

R_500=[]
R_1000=[]
R_1500=[]
R_2000=[]
R_2500=[]
R_3000=[]
R_3500=[]
R_4000=[]
R_4500=[]
R_5000=[]


for i in range(len(H)):
    if S_500[i]==None:
        R_500.append(None)
    else:
        R_500.append(results_d_best[500][H[i]][S_500[i]][3])

    if S_1000[i]==None:
        R_1000.append(None)
    else:
        R_1000.append(results_d_best[1000][H[i]][S_1000[i]][3])

    if S_1500[i]==None:
        R_1500.append(None)
    else:
        R_1500.append(results_d_best[1500][H[i]][S_1500[i]][3])

    if S_2000[i]==None:
        R_2000.append(None)
    else:
        R_2000.append(results_d_best[2000][H[i]][S_2000[i]][3])

    if S_2500[i]==None:
        R_2500.append(None)
    else:
        R_2500.append(results_d_best[2500][H[i]][S_2500[i]][3])

    if S_3000[i]==None:
        R_3000.append(None)
    else:
        R_3000.append(results_d_best[3000][H[i]][S_3000[i]][3])

    if S_3500[i]==None:
        R_3500.append(None)
    else:
        R_3500.append(results_d_best[3500][H[i]][S_3500[i]][3])

    if S_4000[i]==None:
        R_4000.append(None)
    else:
        R_4000.append(results_d_best[4000][H[i]][S_4000[i]][3])

    if S_4500[i]==None:
        R_4500.append(None)
    else:
        R_4500.append(results_d_best[4500][H[i]][S_4500[i]][3])

    if S_5000[i]==None:
        R_5000.append(None)
    else:
        R_5000.append(results_d_best[5000][H[i]][S_5000[i]][3])


#R_500=[results_d_best[500][H[i]][S_500[i]][3] for i in range(len(H))]
#R_1000=[results_d_best[1000][H[i]][S_1000[i]][3] for i in range(len(H))]
#R_1500=[results_d_best[1500][H[i]][S_1500[i]][3] for i in range(len(H))]
#R_2000=[results_d_best[2000][H[i]][S_2000[i]][3] for i in range(len(H))]
#R_2500=[results_d_best[2500][H[i]][S_2500[i]][3] for i in range(len(H))]
#R_3000=[results_d_best[3000][H[i]][S_3000[i]][3] for i in range(len(H))]
#R_3500=[results_d_best[3500][H[i]][S_3500[i]][3] for i in range(len(H))]
#R_4000=[results_d_best[4000][H[i]][S_4000[i]][3] for i in range(len(H))]
#R_4500=[results_d_best[4500][H[i]][S_4500[i]][3] for i in range(len(H))]
#R_5000=[results_d_best[5000][H[i]][S_5000[i]][3] for i in range(len(H))]

fig=plt.figure(figsize=(3.5,3))

plt.semilogy(H,F_500,linestyle='-',marker='^',label=r'$d=500$ km')
plt.semilogy(H,F_1500,linestyle='-',marker='o',label=r'$d=1500$ km')
plt.semilogy(H,F_2500,linestyle='-',marker='P',label=r'$d=2500$ km')
#plt.semilogy(H,F_3000,linestyle='-',marker='P',label=r'$d=3000$ km')
plt.semilogy(H,F_3500,linestyle='-',marker='s',label=r'$d=3500$ km')
plt.semilogy(H,F_4500,linestyle='-',marker='*',label=r'$d=4500$ km')
plt.semilogy(H,F_5000,linestyle='-',marker='d',label=r'$d=5000$ km')

plt.xlabel('$h$ (km)',fontsize=12)
plt.ylabel('$C(h,d)$',fontsize=12)

plt.xlim([0,10500])

plt.xticks([0,2000,4000,6000,8000,10000],fontsize=12)
plt.yticks(fontsize=12)

#plt.legend(fontsize=12)
fig.legend(ncol=6,loc='lower center',bbox_to_anchor = (0.0,-0.15,1,1),bbox_transform = plt.gcf().transFigure,fontsize=12)

plt.tight_layout()

plt.savefig('simulations_all_FOM_22July2020.pdf',bbox_inches='tight')

plt.show()

'''
Rates corresponding to points in the plot above for C(h,d).
'''

fig=plt.figure(figsize=(3.5,3))

#for d in D[9::10]:
#    plt.semilogy(H,R[d],linestyle='-',marker='o',label='$d=$ '+str(d)+' km')

plt.semilogy(H,R_500,linestyle='-',marker='^',label='$d=500$ km')
plt.semilogy(H,R_1500,linestyle='-',marker='o',label='$d=1500$ km')
plt.semilogy(H,R_2500,linestyle='-',marker='P',label='$d=2500$ km')
plt.semilogy(H,R_3500,linestyle='-',marker='s',label='$d=3500$ km')
plt.semilogy(H,R_4500,linestyle='-',marker='*',label='$d=4500$ km')
plt.semilogy(H,R_5000,linestyle='-',marker='d',label='$d=5000$ km')

plt.xlim([0,10000])

plt.xlabel('$h$ (km)',fontsize=12)
plt.ylabel('Rate (ebits/sec)',fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))


#plt.grid()

#plt.legend()

#fig.legend(ncol=1,loc='upper right',bbox_to_anchor = (0.37,-0.021,1,1),bbox_transform = plt.gcf().transFigure,fontsize=12)


plt.tight_layout()

#plt.savefig('simulations_all_rate_13Nov2019.pdf',bbox_inches='tight')
plt.savefig('simulations_all_rateFOM_23July2020.pdf',bbox_inches='tight')


plt.show()

Rmax_500=[calculate_rate(500,h)[1] for h in H]
Rmax_1000=[calculate_rate(1000,h)[1] for h in H]
Rmax_1500=[calculate_rate(1500,h)[1] for h in H]
Rmax_2000=[calculate_rate(2000,h)[1] for h in H]
Rmax_2500=[calculate_rate(2500,h)[1] for h in H]
Rmax_3000=[calculate_rate(3000,h)[1] for h in H]
Rmax_3500=[calculate_rate(3500,h)[1] for h in H]
Rmax_4000=[calculate_rate(4000,h)[1] for h in H]
Rmax_4500=[calculate_rate(4500,h)[1] for h in H]
Rmax_5000=[calculate_rate(5000,h)[1] for h in H]

'''
Rates corresponding to (d,h) pairs with the fewest satellites needed
for continuous global coverage over 24 hours.
'''


fig=plt.figure(figsize=(3.5,3))


#for d in D[9::10]:
#    plt.semilogy(H,R[d],linestyle='-',marker='o',label='$d=$ '+str(d)+' km')

plt.semilogy(H,R[500],linestyle='-',marker='^',label='$d=500$ km')
plt.semilogy(H,R[1500],linestyle='-',marker='o',label='$d=1500$ km')
#plt.semilogy(H,R[2000],linestyle='-',marker='o',label='$d=2000$ km')
plt.semilogy(H,R[2500],linestyle='-',marker='P',label='$d=2500$ km')
#plt.semilogy(H,R[3000],linestyle='-',marker='s',label='$d=3000$ km')
plt.semilogy(H,R[3500],linestyle='-',marker='s',label='$d=3500$ km')
#plt.semilogy(H,R[4000],linestyle='-',marker='*',label='$d=4000$ km')
plt.semilogy(H,R[4500],linestyle='-',marker='*',label='$d=4500$ km')
plt.semilogy(H,R[5000],linestyle='-',marker='d',label='$d=5000$ km')

plt.xlim([0,10000])

plt.xlabel('$h$ (km)',fontsize=12)
plt.ylabel('Rate (ebits/sec)',fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))


#plt.grid()

#plt.legend()

#fig.legend(ncol=1,loc='upper right',bbox_to_anchor = (0.37,-0.021,1,1),bbox_transform = plt.gcf().transFigure,fontsize=12)


plt.tight_layout()

#plt.savefig('simulations_all_rate_13Nov2019.pdf',bbox_inches='tight')
plt.savefig('simulations_all_rate_21July2020.pdf',bbox_inches='tight')


plt.show()

'''
Maximum rates for (d,h) pairs among all sat configs tested.
'''

fig=plt.figure(figsize=(3.5,3))


#for d in D[9::10]:
#    plt.semilogy(H,R[d],linestyle='-',marker='o',label='$d=$ '+str(d)+' km')

plt.semilogy(H,Rmax_500,linestyle='-',marker='^',label='$d=500$ km')
plt.semilogy(H,Rmax_1500,linestyle='-',marker='o',label='$d=1500$ km')
#plt.semilogy(H,R[2000],linestyle='-',marker='o',label='$d=2000$ km')
plt.semilogy(H,Rmax_2500,linestyle='-',marker='P',label='$d=2500$ km')
#plt.semilogy(H,R[3000],linestyle='-',marker='s',label='$d=3000$ km')
plt.semilogy(H,Rmax_3500,linestyle='-',marker='s',label='$d=3500$ km')
#plt.semilogy(H,R[4000],linestyle='-',marker='*',label='$d=4000$ km')
plt.semilogy(H,Rmax_4500,linestyle='-',marker='*',label='$d=4500$ km')
plt.semilogy(H,Rmax_5000,linestyle='-',marker='d',label='$d=5000$ km')

plt.xlim([0,10000])

plt.xlabel('$h$ (km)',fontsize=12)
plt.ylabel('$\overline{R}^{\mathrm{opt}}(h,d)$ (ebits/sec)',fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))


#plt.grid()

#plt.legend(fontsize=12)

fig.legend(ncol=1,loc='upper right',bbox_to_anchor = (0.37,-0.021,1,1),bbox_transform = plt.gcf().transFigure,fontsize=12)


plt.tight_layout()

#plt.savefig('simulations_all_rate_13Nov2019.pdf',bbox_inches='tight')
plt.savefig('simulations_all_rateMax_22July2020.pdf',bbox_inches='tight')


plt.show()

# Combine data into a 2D array
rates = np.array([Rmax_500, Rmax_1500, Rmax_2500, Rmax_3500, Rmax_4500, Rmax_5000], dtype=float)

# Set zero values to a small positive value for logarithmic scaling, but treat them as zero
zero_threshold = 1e-10
rates[rates == 0] = zero_threshold

# Ensure vmin and vmax are valid for LogNorm
vmin = np.nanmin(rates[rates > zero_threshold])
vmax = np.nanmax(rates)

# Create a custom colormap from red to yellow to green
colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red to yellow to green
n_bins = 100  # Number of bins
cmap_name = 'red_yellow_green'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Reversed distances for correct labeling
distances = [5000, 4500, 3500, 2500, 1500, 500]

# Create the heatmap with a logarithmic color scale
plt.figure(figsize=(10, 6))
plt.imshow(rates, aspect='auto', cmap=cm, extent=[H[0], H[-1], distances[-1], distances[0]], origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
cbar = plt.colorbar(label='$\overline{R}^{\mathrm{opt}}(h,d)$ (ebits/sec)')
cbar.set_ticks([10**i for i in range(int(np.log10(vmin)), int(np.log10(vmax)) + 1)])
cbar.set_ticklabels([f'$10^{i}$' for i in range(int(np.log10(vmin)), int(np.log10(vmax)) + 1)])

# Configure the plot
plt.xlabel('$h$ (km)', fontsize=12)
plt.ylabel('$d$ (km)', fontsize=12)
plt.title('Optimal Entanglement-Distribution Rate Heatmap', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(distances[::-1], fontsize=12)  # Reverse the y-ticks to match the correct orientation

plt.tight_layout()
plt.savefig('heatmap_entanglement_distribution_rate.pdf', bbox_inches='tight')
plt.show()