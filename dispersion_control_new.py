### 16/04/2024
### Logan Roberts
### This is the control function for evaluating the dispersion relation of pion (kaon) momentum at non zero twist
### New changes made are for pickle files split by current type - some mass splitting artifacts remain

### 12/08/2025
### Logan updates script to work simply and easily with new fit procedure

import numpy as np
import gvar as gv
from matplotlib import pyplot as plt
import collections
from tabulate import tabulate

plt.rc("font",**{"size":14})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

### Custom files directory changes may have occured
from fit_choice_catalog_by_decay_channel import *
from dispersion.dispersion_functions import*


### Picking ensemble and mass choice
### Ensembles go as F, Fp, SF, SFp, UF -> 0,1,2,3,4
### Mass choices go from lightest to heavy as 0,1,2,3.  Mass shouldnt affect results in meaningful way beyond statistics of different fits over the same data
### Strange == False: Hlpi,  Strange == True: HsK
### To look at all ensembles on same plot -> Do_All_Ensembles == True
### Setting this to true ignores ensemble choice
#ensemble_choice = 1
#mass_choice = 1 ### 0 mass is being funy, not working, missing correlators?
#Strange = False
#Do_All_Ensembles = True
keyword = 'custom'

### Automatic #########
'''
if Do_All_Ensembles == False:
    e_string, m_string = index_to_Fit(ensemble_choice, mass_choice)
    fit_choice = gv.load(fit_Pick_by_Current_Type(e_string,'real'))
    twist_list = index_to_Twist(ensemble_choice)
    if Strange == True: decay = 'BsK'
    else: decay = 'Bpi'

    ### Energy Dispersion Plot
    x_vals, y_vals, c2 = get_E_Dispersion_Vars(Strange, twist_list, fit_choice, ensemble_choice)

    plt.figure(figsize=(8,4))
    plot_E_Dispersion_Relation(x_vals, y_vals, c2, e_string, color = 'blue')
    plt.title('{} {} ensemble Energy dispersion relation, c2 = {}'.format(decay, e_string,c2))
    plot_Discetization_Bounds()
    plt.savefig('./dispersion/{}.png'.format('E_dispersion_plot'),format = 'png')

    ### Amplitude Dispersion Plot
    x_vals, y_vals, d2 = get_Amp_Dispersion_Vars(Strange, twist_list, fit_choice, ensemble_choice)

    plt.figure(figsize=(8,4))
    plot_Amp_Dispersion_Relation(x_vals, y_vals, d2, e_string, color = 'red')
    plt.title('{} {} ensemble Amplitude dispersion relation, d2 = {}'.format(decay, e_string,d2))
    plot_Discetization_Bounds()
    plt.savefig('./dispersion/{}.png'.format('Amp_dispersion_plot'),format = 'png')
'''
'''ticker = 0
plt.figure(figsize=(16,12))
plt.tight_layout()
colors = ['red', 'blue', 'green', 'purple', 'tab:orange', 'tab:cyan']
markers = ['^', '>', 'v', '<', 'D']
for decay_str in ['Hpi', 'HsK']:
    ticker += 1
    plt.subplot(2,2,ticker)
    plt.tight_layout()
    ### Energy
    headers = ['Daughter','twist', '|ap|^2', 'E theory', 'E fit', 'E differnece', 'E theory percent difference']
    table = []
    for i in range(5): 
        ens, m_string = index_to_Fit(i, 0)
        fit_choice = gv.load(fit_Pick_by_Decay_Channel(ens, decay_str))
        twist_list = index_to_Twist(i)
        N_x = index_to_Nx(i)

        x_vals, y_vals, eps2= get_E_Dispersion_Vars(decay_str, twist_list, fit_choice, i, N_x)
        plot_E_Dispersion_Relation(x_vals, y_vals, eps2, ens, colors[i], markers[i])

        ensemble_E_table = get_E_daughter_table(decay_str, twist_list, fit_choice, i, ens)

        table = table + ensemble_E_table + [[]*6]
        #if ticker == 1: plt.ylim([0.95,1.15])
        #elif ticker == 3: plt.ylim([0.95, 1.1])
    #plt.title('{} Energy dispersion relations, am = {}'.format(decay, m_string))
    print(tabulate(table, headers= headers))
    plt.title('{} Energy dispersion relations'.format(decay_str))
    plot_Discetization_Bounds()
    plt.legend(frameon=False, fontsize=12, loc = 2)
    #plt.savefig('./dispersion/dispersion_plots/{}_E_dispersion.pdf'.format(decay_str),format = 'pdf')
    ticker += 1
    ### Amplitude
    #plt.figure(figsize=(8,6))
    plt.subplot(2,2,ticker)
    plt.tight_layout()
    for i in [0,1,2,3,4]:
        ens, m_string = index_to_Fit(i, 0)
        fit_choice = gv.load(fit_Pick_by_Decay_Channel(ens, decay_str))
        twist_list = index_to_Twist(i)
        N_x = index_to_Nx(i)

        x_vals, y_vals, A2 = get_Amp_Dispersion_Vars(decay_str, twist_list, fit_choice, i, N_x)
        plot_Amp_Dispersion_Relation(x_vals, y_vals, A2, ens, colors[i], markers[i])
    #plt.title('{} Amplitude dispersion relations, am={}'.format(decay,m_string))
    plt.title('{} Amplitude dispersion relations'.format(decay_str))
    plot_Discetization_Bounds()
    plt.legend(frameon=False, fontsize=12, loc = 2)

    if ticker == 2: plt.ylim([0.65,1.4])
    elif ticker == 4: plt.ylim([0.6, 1.4])

    #plt.savefig('./dispersion/dispersion_plots/{}_Amp_dispersion.pdf'.format(decay_str),format = 'pdf')
#plt.suptitle('{} Dispersion Plots'.format(keyword))
#plt.subplots_adjust(top=0.93)
#plt.savefig('./dispersion/dispersion_plots/'+keyword+'_dispersion_plots.pdf',format = 'pdf')'''

def do_Es_only(decay_str, ylim_list):
    if decay_str == 'Hpi': alt_str = '\pi'
    elif decay_str == 'HsK': alt_str = 'K'
    plt.figure(figsize=(8,6))
    plt.tight_layout()
    colors = ['red', 'blue', 'green', 'purple', 'tab:orange', 'tab:cyan']
    markers = ['o', '^', 's', 'p', 'h']
    ### Energy
    plot_Discetization_Bounds()
    for i in range(5): 
        ens, m_string, alt_ens = index_to_Fit(i, 0)
        print('Loading in {} {} fit...'.format(decay_str, ens))
        fit_choice = gv.load(fit_Pick_by_Decay_Channel(ens, decay_str))
        print('{} {} fit loaded.'.format(decay_str, ens))
        twist_list = index_to_Twist(i)
        N_x = index_to_Nx(i)

        x_vals, y_vals, eps2= get_E_Dispersion_Vars(decay_str, twist_list, fit_choice, i, N_x)
        plot_E_Dispersion_Relation(x_vals, y_vals, eps2, ens, colors[i], markers[i], alt_ens)
    plt.xlabel(r'$|ap_{0}|^2$'.format(alt_str))
    #plt.ylabel(r'$(E^2_{0} - M^2_{0})/p^2_{0}$'.format(alt_str))
    plt.ylabel(r'$E_{0}/f(M_{0}, p_{0})$'.format(alt_str))
    plt.minorticks_on()
    ax = plt.gca()
    plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
    plt.tick_params(axis = 'x',  which= 'both', top = False, direction = 'in')
    plt.ylim(ylim_list)
    plt.legend(frameon=False, loc = 2, ncols = 2)
    plt.savefig('./dispersion/dispersion_plots/'+keyword+decay_str+'_Edisp_plots.pdf',format = 'pdf')

def do_Amps_only(decay_str, ylim_list):
    if decay_str == 'Hpi': alt_str = '\pi'
    elif decay_str == 'HsK': alt_str = 'K'
    plt.figure(figsize=(8,6))
    plt.tight_layout()
    colors = ['red', 'blue', 'green', 'purple', 'tab:orange', 'tab:cyan']
    markers = ['o', '^', 's', 'p', 'h']
    ### Energy
    plot_Discetization_Bounds()
    for i in range(5): 
        ens, m_string, alt_ens = index_to_Fit(i, 0)
        print('Loading in {} {} fit...'.format(decay_str, ens))
        fit_choice = gv.load(fit_Pick_by_Decay_Channel(ens, decay_str))
        print('{} {} fit loaded.'.format(decay_str, ens))
        twist_list = index_to_Twist(i)
        N_x = index_to_Nx(i)

        x_vals, y_vals, A2 = get_Amp_Dispersion_Vars(decay_str, twist_list, fit_choice, i, N_x)
        plot_Amp_Dispersion_Relation(x_vals, y_vals, A2, ens, colors[i], markers[i], alt_ens)
        #plt.title('{} Energy dispersion relations'.format(decay_str))
    plt.xlabel(r'$|ap_{0}|^2$'.format(alt_str))
    plt.ylabel(r'$A_{0}/f(A_0, M_{0}, p_{0})$'.format(alt_str))
    #plt.ylabel(r'$A_{{\vec{{p}}}}/(A_{{\vec{{0}}}}\sqrt{{M^2_{0}+p^2_{0}}})$'.format(alt_str))
    plt.minorticks_on()
    ax = plt.gca()
    plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
    plt.tick_params(axis = 'x',  which= 'both', top = False, direction = 'in')
    plt.ylim(ylim_list)
    plt.legend(frameon=False, loc = 2, ncols = 2)
    plt.savefig('./dispersion/dispersion_plots/'+keyword+decay_str+'_Ampdisp_plots.pdf',format = 'pdf')


#do_Es_only('Hpi', [0.94,1.20])
#do_Es_only('HsK', [0.95,1.10])
do_Amps_only('Hpi', [0.8, 1.25])
do_Amps_only('HsK', [0.65, 1.24])