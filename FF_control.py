import numpy as np
import gvar as gv
import csv
from matplotlib import pyplot as plt
import collections
from form_factors.FF_functions_new import *
from fit_choice_catalog_by_decay_channel import *
#from form_factors.FF_plotter import func_FF_Plot

########################
### Ensemble Prereqs ###
########################

F = collections.OrderedDict()
F['tag'] = 'F'
F['masses'] = ['0.450','0.55','0.675','0.8']
F['twists'] = ['0.0','0.4281','1.282','2.1410','2.570']
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['Ts'] = [15,18,21,24]
F['tp'] = 96
F['L'] = 32
F['a'] = 0.1715/(1.9006*0.1973) # = 1GeV in lattice untis
F['w0/a'] = gv.gvar('1.9006(20)')

Fp = collections.OrderedDict()
Fp['tag'] = 'Fp'
Fp['masses'] = ['0.433','0.555','0.678','0.8']
Fp['twists'] = ['0.0','0.58','0.87','1.13','3.000','5.311']
Fp['m_l'] = '0.0012'
Fp['m_s'] = '0.036'
Fp['Ts'] = [15,18,21,24]
Fp['tp'] = 96
Fp['L'] = 64
Fp['a'] = 0.1715/(1.9518*0.1973)
Fp['w0/a'] = gv.gvar('1.9518(7)')

SF = collections.OrderedDict()
SF['tag'] = 'SF'
SF['masses'] = ['0.274','0.5','0.65','0.8']
SF['twists'] = ['0.0','1.261','2.108','2.666','5.059']
SF['m_l'] = '0.0048'
SF['m_s'] = '0.0234'
SF['Ts'] = [22,25,28,31]
SF['tp'] = 144
SF['L'] = 48
SF['a'] = 0.1715/(2.896*0.1973)
SF['w0/a'] = gv.gvar('2.896(6)')


SFp = collections.OrderedDict()
SFp['tag'] = 'SFp'
SFp['masses'] = ['0.2585','0.5','0.65','0.8']
SFp['twists'] = ['0.0','2.522','4.216','7.94','10.118']
SFp['m_l'] = '0.0008'
SFp['m_s'] = '0.0219'
SFp['Ts'] = [22,25,28,31]
SFp['tp'] = 192
SFp['L'] = 96
SFp['a'] = 0.1715/(3.0170*0.1973) #from 2207.04765
SFp['w0/a'] = gv.gvar('3.0170(23)')


UF = collections.OrderedDict()
UF['tag'] = 'UF'
UF['masses'] = ['0.194','0.4','0.6','0.8']
UF['twists'] = ['0.0','0.706','1.529','2.235','4.705']
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['Ts'] = [29,34,39,44]
UF['tp'] = 192
UF['L'] = 64
UF['a'] = 0.1715/(3.892*0.1973)
UF['w0/a'] = gv.gvar('3.892(12)')

#####################
### Main Function ###
def main_loop(ensemble, Strange, print_notes=False, append_csv= False, do_gvdump = False, FLAG_dispersion = False, E_budge = None, ensembles2budge = None):
    ### Automatic ###
    mass_choice = [0,1,2,3]
    twist_choice = [0,1,2,3,4]
    if ensemble['tag'] == 'Fp': twist_choice = [0,1,2,3,4,5]

    m_light = float(ensemble['m_l'])    #setting ensemble specific variables
    m_strange = float(ensemble['m_s'])
    #a_lattice = ensemble['a']
    #N_t = ensemble['tp']
    N_x = ensemble['L']

    if Strange == False:                #setting spectator quark specific variables
        amp3_tags = ['SVnn','VVnn','XVnn','TVnn']
        m_spectator = m_light
        call_tag = 'Hpi'
        daughter = 'pion'
    if Strange == True:
        amp3_tags = ['SsVnn','VsVnn','XsVnn','TsVnn']
        m_spectator = m_strange
        call_tag = 'HsK'
        daugter = 'kaon'
    mass_list = []
    for i in mass_choice:
        mass_list.append(ensemble['masses'][i])
    twist_list = []
    for i in twist_choice:
        twist_list.append(ensemble['twists'][i])




    # Reading in pickle file from fit
    g = gv.load(fit_Pick_by_Decay_Channel(ensemble['tag'], call_tag))
    # initialize gv dump if that is set to true
    if do_gvdump == True:
        savedict = gv.BufferDict()
        current_keys = ['S', 'V', 'X', 'T']#, 'Par', 'Perp']
        M_S, M_V, M_X, M_T = [], [], [], []
        M_list = [M_S, M_V, M_X, M_T]
    if print_notes == True:
        print('#########################################################')
        print('{0} Form Factors stats with b-quark mass range \n m_b = {1} * a'.format(call_tag,mass_list))
    
    ####### Begin loop for heavy quark mass in mass list
    for heavy_q_mass in mass_list:
        #PKL_pick(ensemble['tag'],heavy_q_mass)
        Scalar_FFs =[]
        xVector_FFs =[]
        tVector_FFs = []
        Tensor_FFs =[]
        #Parallel_FFs = []
        #Perpendicular_FFs = []
        for twist in twist_list:
            #Distinct FF data point for a given current, mass, and twist
            S_FF = calc_Scalar_FF(amp3_tags[0], heavy_q_mass, twist, m_spectator, N_x, g, FLAG_dispersion = FLAG_dispersion)
            tV_FF = calc_tVector_FF(amp3_tags[1], heavy_q_mass, twist, m_spectator, N_x, g, FLAG_dispersion = FLAG_dispersion)
            xV_FF = calc_xVector_FF(amp3_tags[2], heavy_q_mass, twist, m_spectator, N_x, g, FLAG_dispersion = FLAG_dispersion)
            T_FF = calc_Tensor_FF(amp3_tags[3], heavy_q_mass, twist, N_x, g, ensemble['tag'], FLAG_dispersion = FLAG_dispersion)
            #Par_FF = calc_Parallel_FF(amp3_tags[1], heavy_q_mass, twist, N_x, g, ensemble['w0/a'])
            #Perp_FF = calc_Perpendicular_FF(amp3_tags[2], heavy_q_mass, twist, N_x, g, ensemble['w0/a'])

            #Appending to lists for plotting purposes only, not carrying forward correlations...
            Scalar_FFs.append(S_FF)
            #Parallel_FFs.append(Par_FF)
            if twist != twist_list[0]:
                tVector_FFs.append(tV_FF)
                xVector_FFs.append(xV_FF)
                Tensor_FFs.append(T_FF)
                #Perpendicular_FFs.append(Perp_FF)
        
        FF_list = [Scalar_FFs, tVector_FFs, xVector_FFs, Tensor_FFs]#, Parallel_FFs, Perpendicular_FFs]
        #Implementing gvdump procedure
        if do_gvdump == True:
            for current_key in current_keys:
                array = np.array(FF_list[current_keys.index(current_key)]) 
                if current_key in current_keys[0:4]:
                    M_list[current_keys.index(current_key)].append(array[:,2][0])
                if heavy_q_mass == mass_list[0]:
                    if current_key == current_keys[0]:
                        #Commented out following line cus only m daughter = E daughter [0].  Only one value per ensemble and strange choice
                        #savedict['M-daughters'] = array[:,3][0]
                        savedict['E-daughters_{}'.format(ensemble['tag'])] = array[:,4].T
                        if E_budge != None:
                            if ensemble['tag'] in ensembles2budge:
                                new_arr = []
                                arr = array[:,4].T
                                for E in arr:
                                    new_arr.append(gv.gvar(E.mean, E.sdev*E_budge))
                                savedict['E-daughters_{}'.format(ensemble['tag'])] = np.array(new_arr)
                savedict['FFs_{}_{}_m{}'.format(ensemble['tag'], current_key, heavy_q_mass)] = array[:,0].T
                savedict['q2s_{}_{}_m{}'.format(ensemble['tag'], current_key, heavy_q_mass)] = array[:,1].T

        if print_notes == True:
            print('---------------------------------------------------------')
            print('{0} m_b = {1}a'.format(ensemble['tag'], heavy_q_mass))
            print('\t Scalar Form Factors f_0')
            for entry in Scalar_FFs:
                print('\t \t q^2 = {}: f_0 = {}'.format(entry[1],entry[0]))
            print('\t Temporal Vector Form Factors f_+t')
            for entry in tVector_FFs:
                print('\t \t q^2 = {}: f_+t = {}'.format(entry[1],entry[0]))
            print('\t Spacial Vector Form Factors f_+x')
            for entry in xVector_FFs:
                print('\t \t q^2 = {}: f_+x = {}'.format(entry[1],entry[0]))
            print('\t Tensor Form Factors f_T')
            for entry in Tensor_FFs:
                print('\t \t q^2 = {}: f_T = {}'.format(entry[1],entry[0]))
            print('\t Parallel Form Factors f_∥')
            #for entry in Parallel_FFs:
            #    print('\t \t q^2 = {}: f_∥ = {}'.format(entry[1],entry[0]))
            #print('\t Perpendicular Form Factors f_⊥')
            #for entry in Perpendicular_FFs:
            #    print('\t \t q^2 = {}: f_⊥ = {}'.format(entry[1],entry[0]))
            #######
        
        if append_csv == True:
            FF_components = [[Scalar_FFs, xVector_FFs, tVector_FFs, Tensor_FFs], ['f_0', 'f_+x', 'f_+t', 'f_T']]#,'f_∥', 'f_⊥']]#, Parallel_FFs,Perpendicular_FFs],
            f = open('./form_factors/FF_data_list.csv', 'a')
            writer = csv.writer(f)
            tick = -1
            for component in FF_components[0]:
                tick += 1
                for entry in component:
                    if entry[0] != None:
                        if entry[0] != 'Div by Zero':
                            row = [call_tag, ensemble['tag'], heavy_q_mass, FF_components[1][tick],entry[1], entry[0]]
                            writer.writerow(row)
            f.close()
    if append_csv == True:
        print('Appended {0} {1} data to FF_data_list.csv'.format(call_tag, ensemble['tag']))
    ####### End loop for heavy quark mass in mass list
    if '0.0' in twist_list:
        if print_notes == True:
            #print('---------------------------------------------------------')
            #print('NOTE: No 0-twist (q^2 max) 3-point spacial vector amplitudte')
            #print('NOTE: No 0-twist (q^2 max) 3-point Tensor amplitudte')
            #print('NOTE: As q^2 -> q^2 max, temporal vector FF gives Div by Zero numerical error')
            print('#########################################################')   
    # Final savedict addition - spin taste mother meson masses
    if do_gvdump == True:
        for current_key in current_keys[0:4]:
            savedict['M-mother_{}_{}'.format(ensemble['tag'], current_key)] = np.array(M_list[current_keys.index(current_key)])

        #Beginning writing to textdoc for personal use -not to be used in actual code
        filename_txt = './form_factors/FF_gvdumps/FF-gvdump_{0}_{1}'.format(call_tag, ensemble['tag'])
        f = open('{0}.txt'.format(filename_txt),'w')
        f.write(str(savedict))
        f.close()
        # Beginning form factor gv dump
        #print(savedict) #Uncomment for testing purposes
        filename = './form_factors/FF_gvdumps/FF-gvdump_{0}_{1}.pickle'.format(call_tag, ensemble['tag'])
        print('Beginning gvdump to {}'.format(filename))
        gv.gdump(savedict,filename)
        print('Successful gvdump to {}'.format(filename))
    
#####################
### Control Panel ###
#####################
#Covering all ensembles and both Bpi and BsK should now be taken care of by the following main() function
#Before running, if you care about plotting, clear all lines of text from ./form_factors/GG_data_list.csv

#FLAG dispersion = true means, rather than using pion/kaon energy fit posteriors at all energies to calc form factors..
#   we instead use the zero twist (mass) and the dispersion relation to calculate non-zero-twist energies
FLAG_dispersion = False

#for dealing with enforcing the dispersion relation in select ensembles, and "budging" the uncertaties on forced E posteriors to be -
    # E_budge*E.sdev.  Doing this should only be done if the fits in fit_choice_catalog are from forced dispersion
        #   Ensembles you want to affect, include kwarg ensembles2budge = [...]


def main():
    for ensemble in [F, Fp, SF, SFp, UF]:
        main_loop(ensemble, False, print_notes=False, append_csv=False, do_gvdump=True, FLAG_dispersion = FLAG_dispersion)
        main_loop(ensemble, True, print_notes=False, append_csv=False, do_gvdump=True, FLAG_dispersion = FLAG_dispersion, E_budge = None, ensembles2budge = ['Fp', 'SF', 'SFp'])
    #func_FF_Plot()
main()