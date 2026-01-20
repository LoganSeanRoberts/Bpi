### 30/01/2024
### Logan Roberts
### Backround functions for dispersion control in higher directory

import numpy as np
import gvar as gv
from matplotlib import pyplot as plt
from tabulate import tabulate

def index_to_Fit(i,j):
    ensembles = ['F', 'Fp', 'SF', 'SFp', 'UF']
    alt_ens = ['f5', 'fphys', 'sf5', 'sfphys', 'uf5']
    masses = [
        ['0.450','0.55','0.675','0.8'],
        ['0.433','0.555','0.678','0.8'],
        ['0.274','0.5','0.65','0.8'],
        ['0.2585','0.5','0.65','0.8'],
        ['0.194','0.4','0.6','0.8']
    ]
    return ensembles[i],masses[i][j], alt_ens[i]

def index_to_Twist(i):
    twists = [
        ['0.0','0.4281','1.282','2.1410','2.570'],
        ['0.0','0.58','0.87','1.13','3.000','5.311'], 
        ['0.0','1.261','2.108','2.666','5.059'],
        ['0.0','2.522','4.216','7.94','10.118'],
        ['0.0','0.706','1.529','2.235','4.705']
    ]
    return twists[i]

def index_to_Nx(i):
    Nxs = [32, 64, 48, 96, 64]
    return(Nxs[i])

# The non trunctated version of the dispersion relation
def full_Edispersion_relation(aM_daughter, twist, N_x):
    aE = gv.arccosh(1 + 0.5*(aM_daughter**2) + 3*(1 - gv.cos(float(twist)*np.pi/N_x)))
    return aE
def full_Adispersion_relation(aM_daughter, aE_daughter, aAmp_tw0):
    aAmp = aAmp_tw0 * gv.sqrt(aM_daughter/aE_daughter)
    return aAmp


def calc_Daughter_3Momentum(twist, i): #twist, lattice length in lattice units. 
    N_xes = [32, 64, 48, 96, 64]
    N_x = N_xes[i]
    ap_daughter = float(twist)*np.pi*np.sqrt(3)/N_x
    return ap_daughter

def get_E_Dispersion_Vars(decay_str, twist_list, g, ensemble_choice, N_x):  
    if decay_str == 'HsK':
        daughter_choice = 'kaon'
        eps = 'eps_K2'
    else: 
        daughter_choice = 'pion'
        eps = 'eps_pi2'
    m_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th0.0)'][0])

    y_vals, x_vals, = [], []
    for twist in twist_list[1:]:
        ap_daughter = calc_Daughter_3Momentum(twist, ensemble_choice)
        E_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th'+twist+')'][0])
        #y_val = (E_daughter**2 - m_daughter**2) / gv.abs(ap_daughter)**2
        y_val = E_daughter / full_Edispersion_relation(m_daughter, twist, N_x)
        y_vals.append(y_val)
        x_vals.append(gv.abs(ap_daughter)**2)
    eps2 = g['{}'.format(eps)]
    return x_vals, y_vals, eps2

### This function will replicate the axes like fig 4 in https://arxiv.org/pdf/2010.07980.pdf
# def get_E_Dispersion_Vars(decay_str, twist_list, g, ensemble_choice):  
#     if decay_str == 'HsK':
#         daughter_choice = 'kaon'
#         eps = 'eps_K2'
#     else: 
#         daughter_choice = 'pion'
#         eps = 'eps_pi2'
#     m_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th0.0)'][0])

#     y_vals, x_vals, = [], []
#     for twist in twist_list[1:]:
#         ap_daughter = calc_Daughter_3Momentum(twist, ensemble_choice)
#         E_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th'+twist+')'][0])
#         #y_val = (E_daughter**2 - m_daughter**2) / gv.abs(ap_daughter)**2
#         y_val = (E_daughter**2) / (m_daughter**2 + gv.abs(ap_daughter)**2)
#         y_vals.append(y_val)
#         x_vals.append(gv.abs(ap_daughter)**2)
#     eps2 = g['{}'.format(eps)]
#     return x_vals, y_vals, eps2

### This prints a handy dandy table for the dispersion realtion to pair with the plot
def get_E_daughter_table(decay_str, twist_list, g, ensemble_choice, e_string):
    if decay_str == 'HsK':
        daughter_choice = 'kaon'
    else: 
        daughter_choice = 'pion'
    m_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th0.0)'][0])
    ensemble_E_table = []
    for twist in twist_list:
        ap_daughter = calc_Daughter_3Momentum(twist, ensemble_choice)
        E_daughter_actual = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th'+twist+')'][0])
        E_daughter_theory = gv.sqrt(m_daughter**2 + gv.abs(ap_daughter)**2)
        E_difference = (E_daughter_theory - E_daughter_actual)
        E_percent_diff = (E_difference / E_daughter_theory) * 100
        if gv.abs(E_difference) < 1e-10: E_difference = gv.gvar(0,0)  
        #array_1d = ['{} {}'.format(e_string, daughter_choice), float(twist),np.round(np.abs(ap_daughter), decimals = 4), E_daughter_theory, E_daughter_actual, E_difference]
        array_1d = ['{} {}'.format(e_string, daughter_choice), '{}'.format(float(twist)),
                    '{}'.format(np.round(gv.abs(ap_daughter)**2, decimals = 4)), '{}'.format(E_daughter_theory), 
                    '{}'.format(E_daughter_actual), '{}'.format(E_difference), '{}'.format(E_percent_diff)]
        ensemble_E_table.append(array_1d)
    return ensemble_E_table

def get_Amp_Dispersion_Vars(decay_str, twist_list, g, ensemble_choice, N_x):  ### This is like the function before but for Amp instead of energy
    if decay_str == 'HsK':
        daughter_choice = 'kaon'
        A = 'A_K2'
    else: 
        daughter_choice = 'pion'
        A = 'A_pi2'
    m_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th0.0)'][0])
    th0_amp_daughter = gv.exp(g['log(2pt_' +daughter_choice+ 'G5-G5_th0.0:a)'][0])
    y_vals, x_vals = [],[]
    for twist in twist_list[1:]:
        ap_daughter = calc_Daughter_3Momentum(twist, ensemble_choice)
        thx_amp_daughter = gv.exp(g['log(2pt_' +daughter_choice+ 'G5-G5_th'+twist+':a)'][0]) 
        E_daughter = full_Edispersion_relation(m_daughter, twist, N_x)
        y_val = thx_amp_daughter / full_Adispersion_relation(m_daughter, E_daughter, th0_amp_daughter)
        #y_val = ((th0_amp_daughter**4 / thx_amp_daughter**4  -1) * m_daughter**2)/gv.abs(ap_daughter)**2
        y_vals.append(y_val)
        x_vals.append(gv.abs(ap_daughter)**2)
    A2 = g['{}'.format(A)]
    return x_vals, y_vals, A2

# def get_Amp_Dispersion_Vars(decay_str, twist_list, g, ensemble_choice):  ### This is like the function before but for Amp instead of energy
#     if decay_str == 'HsK':
#         daughter_choice = 'kaon'
#         A = 'A_K2'
#     else: 
#         daughter_choice = 'pion'
#         A = 'A_pi2'
#     m_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th0.0)'][0])
#     th0_amp_daughter = gv.exp(g['log(2pt_' +daughter_choice+ 'G5-G5_th0.0:a)'][0])
#     y_vals, x_vals = [],[]
#     for twist in twist_list[1:]:
#         ap_daughter = calc_Daughter_3Momentum(twist, ensemble_choice)
#         thx_amp_daughter = gv.exp(g['log(2pt_' +daughter_choice+ 'G5-G5_th'+twist+':a)'][0]) 
#         y_val = thx_amp_daughter / th0_amp_daughter * (1 + (gv.abs(ap_daughter)/m_daughter)**2)**(1/4)
#         y_vals.append(y_val)
#         x_vals.append(gv.abs(ap_daughter)**2)
#     A2 = g['{}'.format(A)]
#     return x_vals, y_vals, A2


def gvar_splitter(gvars):
    means, errs = [] , []
    for y in gvars:
        means.append(y.mean)
        errs.append(y.sdev)
    return means, errs

def calc_Discretization_Bounds(ap_sqrd):
    plus_line = 1 + ap_sqrd /(np.pi**2)
    minus_line = 1 - ap_sqrd /(np.pi**2)
    return plus_line, minus_line

def plot_Discetization_Bounds():
    ap_sqrd_space = np.linspace(0,0.35,1000)
    plus_line, minus_line = calc_Discretization_Bounds(ap_sqrd_space)
    #plt.plot(ap_sqrd_space, plus_line, color = 'black', linestyle = 'dotted')
    #plt.plot(ap_sqrd_space, minus_line, color = 'black', linestyle = 'dotted')
    plt.fill_between(ap_sqrd_space, plus_line, minus_line, color = 'tab:cyan', alpha = 0.2)
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    plt.xlim([-0.02,0.35])

def plot_E_Dispersion_Relation(x_vals, y_vals, eps2, ensemble, color, marker, alt_ens):
    y_means, y_errs = gvar_splitter(y_vals)
    #plt.scatter(x_vals, y_means, marker='o', edgecolors= color, facecolors='none', label = '{}'.format(ensemble, eps2))
    plt.errorbar(x_vals, y_means, yerr=2*np.array(y_errs), linestyle = '', color = color, mfc='none', capsize=5, marker = 'none', alpha = 0.4)
    plt.errorbar(x_vals, y_means, yerr=y_errs, linestyle = '', color = color, mfc='none', capsize=5, ms=10, marker = marker, label = '{}'.format(alt_ens))
    

def plot_Amp_Dispersion_Relation(x_vals, y_vals, A2, ensemble, color, marker, alt_ens):
    y_means, y_errs = gvar_splitter(y_vals)
    #plt.scatter(x_vals, y_means, marker='o', edgecolors= color, facecolors='none', label = '{}'.format(ensemble, A2))
    plt.errorbar(x_vals, y_means, yerr=2*np.array(y_errs), linestyle = '', color = color, mfc='none', capsize=5, marker = 'none', alpha = 0.4)
    plt.errorbar(x_vals, y_means, yerr=y_errs, linestyle = '', color = color, mfc='none', capsize=5, ms=10, marker = marker, label = '{}'.format(alt_ens))
    
