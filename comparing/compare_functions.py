import numpy as np
import gvar as gv
import math
from matplotlib import pyplot as plt
import pickle



# Takes 2pt fit outputs and converts to correlator over tspace range
#by default selects g5-g5 (scalar) spin taste correlator - only used for comparing original to recomp corrs
def recon_2pt_corr(fit2pt, Nexp, Nt, comp_choice):
    x = comp_choice
    t_space = np.arange(0, Nt, 1, dtype = float)
    an, En, ao, Eo = fit2pt
    recon_corr = []
    for i in range(Nexp):
        non_sum = np.power(gv.abs(an[x][i]),2) * (np.exp(-En[x][i]*t_space) + np.exp(-En[x][i]*(Nt - t_space))) 
        osc_sum = (-1)**t_space * np.power(gv.abs(ao[x][i]),2) * (np.exp(-Eo[x][i]*t_space) + np.exp(-Eo[x][i]*(Nt - t_space)))
        recon_corr.append(non_sum - osc_sum)
    recon_corr = np.sum(recon_corr, axis = 0)
    return recon_corr


def calc_aM_eff(Corr2pt):
    arg = (np.roll(Corr2pt,-2)+np.roll(Corr2pt, 2))/(2*Corr2pt)
    new_arg = []
    # This part is the banaid for arccosh getting arguments <=1
    for i in range(len(arg)):
        if arg[i] <= 1:
            arg[i] = gv.gvar(1.00000000001)
        new_arg.append(arg[i])
    return (1/2) * gv.arccosh(new_arg)

def calc_Amp_eff(Corr2pt, aM_eff, Nt):
    t_space = np.arange(0, Nt, 1, dtype = float)
    output = np.sqrt(Corr2pt / (np.exp(-aM_eff*t_space) + np.exp(-aM_eff*(Nt - t_space))))
    return output

def calc_J_eff(Corr_3pt, m_mother, m_daughter, Aeff_mother, Aeff_daughter, width, Nt):
    m_mother, m_daughter = m_daughter, m_mother  #This swap fixes thing J_effs plots, same as B,Pi swap in recon_3pt_corr
    t_space = np.arange(0, Nt, 1, dtype = float)
    return Corr_3pt * np.exp(np.array(m_daughter) * t_space) * np.exp(np.array(m_mother)*(width - t_space)) / (np.array(Aeff_mother) * np.array(Aeff_daughter)) 

#converts differential a and E values and converts to absolute for any Nexp > 1
def de_logger(array, Nexp):
    array = np.exp(np.array(array))
    new_array = []
    new_array.append(array[0])
    ticker = 1
    while ticker < Nexp:
        new_value = array[ticker] + new_array[ticker-1]
        new_array.append(new_value)
        ticker += 1
    return new_array

def load_PKL_file(PKL_file):
    return gv.load(PKL_file)

#loading in text file with averaged corr data
def load_Corr_txt(txt_file):
    tag_array = np.array(np.loadtxt(txt_file, delimiter = '&', dtype = str, usecols = 0))
    temp_corr_array = (np.loadtxt(txt_file, delimiter = '&', dtype = str, usecols = 1))
    #Preserving gvar formatting
    corr_array = []
    for line in range(len(temp_corr_array)):
        corr_array.append(gv.gvar(temp_corr_array[line][1:-1].split(',')))
    return tag_array, corr_array

#defining function for picking the right 2pt averaged correlator
def Corr2pt_picker(strange, mass, twist, tag_array, corr_array):
    if strange == True:
        spectator, daughter = 's', 'kaon'
    else: spectator, daughter = 'l', 'pion'
    tag_mother = '2pt_H{}G5-G5_m{}_th0.0'.format(spectator, mass)
    index_mother = list(tag_array).index(tag_mother)
    tag_daughter = '2pt_{}G5-G5_th{}'.format(daughter, twist)
    index_daughter = list(tag_array).index(tag_daughter)
    return tag_array[index_mother], corr_array[index_mother], tag_array[index_daughter], corr_array[index_daughter]

#defining function for picking the right 3pt averaged correlator
def Corr3pt_picker(strange, mass, twist, width, tag_array, corr_array):
    if strange == True:
        mesons = 'HsK'
    else: mesons = 'Hlpi'
    components, tag_array_3pt, corr_array_3pt = ['scalar', 'tempVec', 'spatVec', 'tensor'], [], []
    for comp in components:
        tag_3pt = '3pt_{}_{}_width{}_m{}_th{}'.format(comp, mesons, width, mass, twist)
        index_3pt = list(tag_array).index(tag_3pt)
        tag_array_3pt.append(tag_3pt)
        corr_array_3pt.append(corr_array[index_3pt])
    return tag_array_3pt, corr_array_3pt
    
# Function for picking out 3pt posteriors from pkl file
# Order string is just a reminder of the order that the arrays come in.
#Output array is unique for each mass, twist, and strange choice
def get_3pt_posteriors(g_real, g_imaginary, strange, mass, twist):
    if strange == True:
        amptags_comp = ['Ss', 'Vs', 'Xs', 'Ts']
    else: amptags_comp = ['S', 'V', 'X', 'T']
    amptags_osc = ['nn', 'no', 'on', 'oo']
    amplist_all_comps = [] #order is S, V (temporal), X  (spatial), T
    index = 0
    print('Checking proper pkl file selection...')
    for amptag_comp in amptags_comp:
        print(amptag_comp)
        if index < 2: 
            g_choice = g_real
            print('g_choice = real')
        else: 
            g_choice = g_imaginary
            print('g_choice = imaginary')
        amp_list_osc_states = [] # Order is nn, no, on, oo
        for amptag_osc in amptags_osc:
            osc_choice = g_choice['{}V{}_m{}_tw{}'.format(amptag_comp, amptag_osc, mass, twist)]
            amp_list_osc_states.append(osc_choice) #osc_choice order of 00,01,02,10,11,12,,20,21,22, and so forth in case of Nexp > 3
        amplist_all_comps.append(amp_list_osc_states)
        index +=1
    order_string = 'ORDER: {}, {}, [[0_0, ..., 0_Nexp-1]...[Nexp-1_0, ... Nexp-1_Nexp-1]]'.format(amptags_comp, amptags_osc)
    return amplist_all_comps, order_string


def get_2pt_posteriors(g_real, g_imaginary, mass, twist, spectator, daughter, Nexp):
    an_mothers, En_mothers, ao_mothers, Eo_mothers = [], [], [], []
    an_daughters, En_daughters, ao_daughters, Eo_daughters = [], [], [], []
    
    spin_tastes = ['G5-G5', 'G5T-G5T', 'G5-G5X', 'G5T-GYZ']
    spin_taste_ticker = 0
    for spin_taste in spin_tastes:
        g = g_real
        if spin_taste_ticker > 1: g = g_imaginary
        #
        an_mothers.append(np.exp(np.array(list(g['log(2pt_H{}{}_m{}_th0.0:a)'.format(spectator, spin_taste, mass)]))))
        En_mothers.append(de_logger(list(g['log(dE:2pt_H{}{}_m{}_th0.0)'.format(spectator, spin_taste, mass)]), Nexp))
        ao_mothers.append(np.exp(np.array(list(g['log(o2pt_H{}{}_m{}_th0.0:a)'.format(spectator, spin_taste, mass)]))))
        Eo_mothers.append(de_logger(list(g['log(dE:o2pt_H{}{}_m{}_th0.0)'.format(spectator, spin_taste, mass)]), Nexp))
        
        #
        an_daughters.append(np.exp(np.array(list(g['log(2pt_{}G5-G5_th{}:a)'.format(daughter, twist)]))))
        En_daughters.append(de_logger(list(g['log(dE:2pt_{}G5-G5_th{})'.format(daughter, twist)]), Nexp))
        ao_daughters.append(np.exp(np.array(list(g['log(o2pt_{}G5-G5_th{}:a)'.format(daughter, twist)]))))
        Eo_daughters.append(de_logger(list(g['log(dE:o2pt_{}G5-G5_th{})'.format(daughter, twist)]), Nexp))
        spin_taste_ticker += 1   
    #
    posteriors_2pt_mother = [an_mothers, En_mothers, ao_mothers, Eo_mothers]
    posteriors_2pt_daughter = [an_daughters, En_daughters, ao_daughters, Eo_daughters]    
    return posteriors_2pt_mother, posteriors_2pt_daughter

# the big-daddy equation of reconstructing 3pt correlators from the fit posteriors, for all 4 current components.
#
def recon_3pt_corr(posteriors_3pt, posteriors_2pt_mother, posteriors_2pt_daughter, width, Nexp, Nt):
    t = np.arange(0, Nt, 1)
    S, V, X, T = posteriors_3pt[0], posteriors_3pt[1], posteriors_3pt[2], posteriors_3pt[3]
    comps = [S, V, X, T]
    i_index, j_index = range(Nexp), range(Nexp)
    pi, B = posteriors_2pt_mother, posteriors_2pt_daughter  ### Why does this (maybe) work better? Swappin mother and daughter corr
    reconned_3pt_corrs = []
    x = 0
    for comp in comps:
        corrs = []
        for i in i_index:
            for j in j_index:
                line_1 = pi[0][x][i] * comp[0][i][j] * B[0][x][j] * np.exp(-pi[1][x][i] * t) * np.exp(-B[1][x][j] * (width - t))
                line_2 = ((-1.0)**(width - t)) * pi[0][x][i] * comp[1][i][j] * B[2][x][j] * np.exp(-pi[1][x][i] * t) * np.exp(-B[3][x][j] * (width - t))
                line_3 = ((-1.0)**t) * pi[2][x][i] * comp[2][i][j] * B[0][x][j] * np.exp(-pi[3][x][i] * t) * np.exp(-B[1][x][j] * (width - t))
                line_4 = ((-1.0)**width) * pi[2][x][i] * comp[3][i][j] * B[2][x][j] * np.exp(-pi[3][x][i] * t) * np.exp(-B[3][x][j] * (width - t))
                corr_ij = (line_1 - line_2 - line_3 + line_4)
                corrs.append(corr_ij)
        summed_corrs = np.sum(corrs, axis = 0)
        reconned_3pt_corrs.append(summed_corrs)
        x += 1
    return reconned_3pt_corrs

# Gvar splitter
def gvar_splitter(gvars):
    means, errs = [] , []
    for y in gvars:
        means.append(y.mean)
        errs.append(y.sdev)
    return means, errs

#log funcitons dont like x <= 0, this is a bandaid for that
def zero_setter(gvar_list):
    new_gvar_list = []
    for i in range(len(gvar_list)):
        if gvar_list[i] <= 0:
            gvar_list[i] = gv.gvar(0.00000000001)
        new_gvar_list.append(gvar_list[i])
    return new_gvar_list