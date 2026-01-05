import numpy as np
import gvar as gv
from matplotlib import pyplot as plt
import collections

#######################################
### Defining prerequisite functions ###
#######################################

def calc_Zdisc(mass_choice):  ### Got this from Will, I need to be able to explain what is going on here
    mh=float(mass_choice)
    mtree= mh * ( 1 - 3.0*mh**4 / 80.0 + 23*mh**6/2240 + 1783*mh**8/537600 - 76943*mh**10/23654400 )
    eps = ( 4 - np.sqrt( 4 + 12*mtree/( np.cosh(mtree)*np.sinh(mtree) ) )) / (np.sinh(mtree))**2 - 1
    Z_disc = np.sqrt(np.cosh(mtree)*(1-((1+eps)/2)*np.sinh(mtree)**2))
    return Z_disc 

def calc_Daughter_3Momentum(twist, N_x): #twist, lattice length in lattice units. 
    ap_daughter = float(twist)*np.pi/N_x
    return ap_daughter

def calc_Lattice_Current(m_mother, E_daughter, Three_pt_amp, mass_choice):
    return 2*calc_Zdisc(mass_choice) * gv.sqrt(m_mother* E_daughter)* Three_pt_amp

def calc_Z_V(m_h, m_l, m_mother, m_daughter, amp_S_zerotwist, amp_V_zerotwist,  mass_choice): #I think all the a terms cancel?
    Z_V = (m_h - m_l) / (m_mother - m_daughter) * calc_Lattice_Current(m_mother, m_daughter, amp_S_zerotwist, mass_choice) / calc_Lattice_Current(m_mother, m_daughter, amp_V_zerotwist, mass_choice)
    return Z_V

#Now we can emplement some q_sqrd equations that either use the fit posterior E_daughter for non zero twist(more uncertaint)...
# or use the full lattice dispersion relation, which just uses the daughter meson mass, a twist, and the spacial length of the lattice
def calc_q_sqrd_v1(M_mother,M_daughter, E_daughter):#,twist, N_x):
    #q_sqrd = ((m_mother-E_daughter)**2)-(np.sqrt(3)*calc_Daughter_3Momentum(twist, N_x)**2)
    q_sqrd = M_mother**2 + M_daughter**2 - 2*M_mother*E_daughter
    return q_sqrd

def calc_q_sqrd_v2(M_mother,M_daughter, twist, N_x):
    E_daughter = gv.arccosh(1 + 0.5*(M_daughter**2) + 3*(1 - gv.cos(float(twist)*np.pi/N_x)))
    q_sqrd = M_mother**2 + M_daughter**2 - 2*M_mother*E_daughter
    return q_sqrd

def calc_q_sqrd(M_mother, M_daughter, E_daughter, FLAG_dispersion = False, twist = '', N_x = 0):
    if FLAG_dispersion == False:
        q_sqrd = calc_q_sqrd_v1(M_mother,M_daughter, E_daughter)
    elif FLAG_dispersion == True:
        if twist != '':
            if N_x != 0:
                q_sqrd = calc_q_sqrd_v2(M_mother, M_daughter, twist, N_x)
            else: print('FLAG_dispersion error, need to set N_x in function argument')
        else: print('FLAG_dispersion error, need to set twist in function argument')
    else: print('Invalid value of FLAG disperison in calc qsqrd funtion, must be True or False')
    return q_sqrd

def get_Scalar_stats(Scalar_amp_choice, mass_choice, twist_choice, g, spin_taste, FLAG_dispersion = False, N_x = 0):
    if 's' in Scalar_amp_choice:
        mother_choice = 'Hs'
        daughter_choice = 'kaon'  ### on 30/01/2024 Logan swapped these to where they are now
                                  ### I believe before it was wrong -> if 's' in ampchoice -> non strange case
    else: 
        mother_choice = 'Hl'
        daughter_choice = 'pion'
    m_mother = gv.exp(g['log(dE:2pt_'+ mother_choice + spin_taste+'_m'+mass_choice+'_th0.0)'][0])
    m_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th0.0)'][0])
    #IF FLAG_dispersion == True, then we only take the pion/kaon mass posterior to get non-zero-twist energy, not fit posterior of non-zero-twist
    if FLAG_dispersion == True:
        if twist_choice != '0.0':
            E_daughter = gv.arccosh(1 + 0.5*(m_daughter**2) + 3*(1 - gv.cos(float(twist_choice)*np.pi/N_x)))
            print('Flag Disp = True check!')
        else: E_daughter = m_daughter
    else: E_daughter = gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th'+twist_choice+')'][0])
    Scalar_amp = g[(Scalar_amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0][0]
    #print('Twist choice = {}'.format(twist_choice))
    #print('Dispflag = False: {}'.format(gv.exp(g['log(dE:2pt_'+daughter_choice+'G5-G5_th'+twist_choice+')'][0])))
    #print('Dispflag = True : {}'.format(gv.arccosh(1 + 0.5*(m_daughter**2) + 3*(1 - gv.cos(float(twist_choice)*np.pi/N_x)))))
    #print('\n')
    return m_mother, m_daughter, E_daughter, Scalar_amp

def calc_Z_T(ensemble_tag): ### from https://arxiv.org/abs/2008.02024 table 8
    #Assume ensemble order is:
    tag_list = ['F', 'Fp', 'SF', 'SFp', 'UF']
    Z_T_list = [gv.gvar('1.0029(43)'), gv.gvar('1.0029(43)'), gv.gvar('1.0342(43)'), 
                gv.gvar('1.0342(43)'), gv.gvar('1.0476(42)')]
    index = tag_list.index(ensemble_tag)
    if index in range(5): return Z_T_list[index]
    else: print('error in calc_Z_T function, invalid index')

######################################
### Defining Form Factor equations ###
######################################

def calc_Scalar_FF(amp_choice, mass_choice, twist_choice, am_l, N_x,g, FLAG_dispersion = False):
    if 'S' in amp_choice:
        m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(amp_choice, mass_choice, twist_choice, g, 'G5-G5', FLAG_dispersion= FLAG_dispersion, N_x = N_x)
        q_sqrd = calc_q_sqrd(m_mother,m_daughter, E_daughter, FLAG_dispersion=FLAG_dispersion, twist=twist_choice, N_x=N_x)
        FF = calc_Lattice_Current(m_mother, E_daughter, Three_pt_amp, mass_choice) * (float(mass_choice)-am_l)/((m_mother**2)-(m_daughter**2))
        #print(FF)
        return FF, q_sqrd, m_mother, m_daughter, E_daughter
    else:
        print('ERROR: Options for Scalar_FF are "SVnn" and "SsVnn". Cannot apply to {}'.format(amp_choice))

def calc_tVector_FF(amp_choice, mass_choice, twist_choice, am_l, N_x, g, FLAG_dispersion = False):
    if twist_choice == '0.0':
        return None, None, None, None, None
    else:
        if amp_choice.startswith('V'):     
            if 's' in amp_choice:
                Scalar_amp = 'SsVnn'
            else: 
                Scalar_amp = 'SVnn'
            m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(Scalar_amp, mass_choice, twist_choice, g, 'G5T-G5T',FLAG_dispersion= FLAG_dispersion, N_x = N_x)
            Scalar_FF = calc_Scalar_FF(Scalar_amp, mass_choice, twist_choice, am_l, N_x, g)[0]
            #Three_p_daughter = calc_Daughter_3Momentum(twist_choice, N_x)
            tVector_amp_zerotwist = g[(amp_choice+'_m'+mass_choice+'_tw0.0')][0][0]
            Scalar_amp_zerotwist = g[(Scalar_amp+'_m'+mass_choice+'_tw0.0')][0][0]
            tVector_amp = g[(amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0][0]
            Z_V = calc_Z_V(float(mass_choice),am_l, m_mother,m_daughter, Scalar_amp_zerotwist, tVector_amp_zerotwist, mass_choice)
            V_matrix = calc_Lattice_Current(m_mother, E_daughter, tVector_amp, mass_choice)
            q_sqrd = calc_q_sqrd(m_mother,m_daughter, E_daughter, FLAG_dispersion=FLAG_dispersion, twist=twist_choice, N_x=N_x)
            fraction = ((m_mother**2-m_daughter**2)/q_sqrd)*(m_mother - E_daughter)
            FF = (Z_V*V_matrix - Scalar_FF*fraction) / (m_mother + E_daughter - fraction)
            return FF, q_sqrd ,m_mother, m_daughter, E_daughter
        else:
            print('ERROR: Options for tVector_FF are "VVnn" and "VsVnn". Cannot apply to {}'.format(amp_choice))


def calc_xVector_FF(amp_choice, mass_choice, twist_choice, am_l, N_x, g, FLAG_dispersion = False):
    if twist_choice == '0.0':
        return None, None, None, None, None
    else:
        if amp_choice.startswith('X'):
            if 's' in amp_choice:
                Scalar_amp, tVector_amp = 'SsVnn', 'VsVnn'
            else: 
                Scalar_amp, tVector_amp = 'SVnn', 'VVnn'
            m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(Scalar_amp, mass_choice, twist_choice, g, 'G5-G5X', FLAG_dispersion= FLAG_dispersion, N_x = N_x)
            Scalar_FF = calc_Scalar_FF(Scalar_amp, mass_choice, twist_choice, am_l, N_x, g)[0]
            Three_p_daughter = calc_Daughter_3Momentum(twist_choice, N_x)
            xVector_amp = g[(amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0][0]
            tVector_amp_zerotwist = g[(tVector_amp+'_m'+mass_choice+'_tw0.0')][0][0]
            Scalar_amp_zerotwist = g[(Scalar_amp+'_m'+mass_choice+'_tw0.0')][0][0]
            Z_V = calc_Z_V(float(mass_choice),am_l, m_mother,m_daughter, Scalar_amp_zerotwist, tVector_amp_zerotwist, mass_choice)
            V_matrix = calc_Lattice_Current(m_mother, E_daughter, xVector_amp, mass_choice)
            q_sqrd = calc_q_sqrd(m_mother,m_daughter, E_daughter, FLAG_dispersion=FLAG_dispersion, twist=twist_choice, N_x=N_x)
            fraction = ((m_mother**2-m_daughter**2)/q_sqrd)*(-Three_p_daughter)
            FF = (Z_V*V_matrix - Scalar_FF*fraction) / (Three_p_daughter - fraction)
            return FF, q_sqrd, m_mother, m_daughter, E_daughter
        else:
            print('ERROR: Options for xVector_FF are "XVnn" and "XsVnn". Cannot apply to {}'.format(amp_choice))

### Testing new version

'''def calc_xVector_FF(amp_choice, mass_choice, twist_choice, am_l, N_x, g_real, g_imaginary):
    if twist_choice == '0.0':
        return None, None, None, None, None
    else:
        if amp_choice.startswith('X'):
            if 's' in amp_choice:
                Scalar_amp, tVector_amp = 'SsVnn', 'VsVnn'
            else: 
                Scalar_amp, tVector_amp = 'SVnn', 'VVnn'
            m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(Scalar_amp, mass_choice, twist_choice, g_real)
            Scalar_FF = calc_Scalar_FF(Scalar_amp, mass_choice, twist_choice, am_l, N_x, g_real)[0]
            Three_p_daughter = calc_Daughter_3Momentum(twist_choice, N_x)
            xVector_amp = g_imaginary[(amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0]
            tVector_amp_zerotwist = g_real[(tVector_amp+'_m'+mass_choice+'_tw0.0')][0]
            Scalar_amp_zerotwist = g_real[(Scalar_amp+'_m'+mass_choice+'_tw0.0')][0]
            Z_V = calc_Z_V(float(mass_choice),am_l, m_mother,m_daughter, Scalar_amp_zerotwist, tVector_amp_zerotwist, mass_choice)
            V_matrix = calc_Lattice_Current(m_mother, E_daughter, xVector_amp, mass_choice)
            q_sqrd = calc_q_sqrd(m_mother, E_daughter, twist_choice, N_x)
            fraction = ((m_mother**2-m_daughter**2)/q_sqrd)
            FF = (-Z_V*V_matrix/Three_p_daughter + Scalar_FF*fraction) / (1 + fraction)
            return FF[0], q_sqrd, m_mother, m_daughter, E_daughter
        else:
            print('ERROR: Options for xVector_FF are "XVnn" and "XsVnn". Cannot apply to {}'.format(amp_choice))'''

def calc_Tensor_FF(amp_choice, mass_choice, twist_choice, N_x, g, ensemble_tag, FLAG_dispersion = False):
    if twist_choice == '0.0':
        return None, None, None, None, None
    else:
        if amp_choice.startswith('T'):
            if 's' in amp_choice:
                Scalar_amp = 'SsVnn'
            else: 
                Scalar_amp = 'SVnn'
            m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(Scalar_amp, mass_choice, twist_choice, g, 'G5T-GYZ', FLAG_dispersion= FLAG_dispersion, N_x = N_x)
            Three_p_daughter = calc_Daughter_3Momentum(twist_choice, N_x)
            T_amp = g[(amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0][0]
            T_matrix = calc_Lattice_Current(m_mother, E_daughter, T_amp, mass_choice)
            Z_T = calc_Z_T(ensemble_tag)
            fraction = (m_mother+m_daughter)/(2*m_mother*Three_p_daughter) ### MISSING COMPLEX NUMBER
            q_sqrd = calc_q_sqrd(m_mother,m_daughter, E_daughter, FLAG_dispersion=FLAG_dispersion, twist=twist_choice, N_x=N_x)
            FF = Z_T * T_matrix * fraction
            return FF, q_sqrd, m_mother, m_daughter, E_daughter
        else:
            print('ERROR: Options for Tensor_FF are "TVnn" and "TsVnn". Cannot apply to {}'.format(amp_choice))

######################################
### f parellel and f perpendicular ###
######################################

def calc_Parallel_FF(amp_choice, mass_choice, twist_choice, N_x, g, w0_a):
    if amp_choice.startswith('V'):     
        if 's' in amp_choice:
            Scalar_amp = 'SsVnn'
        else: 
            Scalar_amp = 'SVnn'
        m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(Scalar_amp, mass_choice, twist_choice, g, 'G5T-G5T')
        tVector_amp = g[(amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0]
        V_matrix = calc_Lattice_Current(m_mother, E_daughter, tVector_amp, mass_choice)
        q_sqrd = calc_q_sqrd(m_mother,m_daughter, E_daughter)
        FF = V_matrix / gv.sqrt(2*m_mother)
        return gv.sqrt(w0_a)*FF, q_sqrd ,m_mother, m_daughter, E_daughter
    else:
        print('ERROR: Options for f_parallel are "VVnn" and "VsVnn". Cannot apply to {}'.format(amp_choice))


def calc_Perpendicular_FF(amp_choice, mass_choice, twist_choice, N_x, g, w0_a):
    if twist_choice == '0.0':
        return None, None, None, None, None
    else:
        if amp_choice.startswith('X'):
            if 's' in amp_choice:
                Scalar_amp, tVector_amp = 'SsVnn', 'VsVnn'
            else: 
                Scalar_amp, tVector_amp = 'SVnn', 'VVnn'
            m_mother, m_daughter, E_daughter, Three_pt_amp = get_Scalar_stats(Scalar_amp, mass_choice, twist_choice, g, 'G5-G5X')
            Three_p_daughter = calc_Daughter_3Momentum(twist_choice, N_x)
            xVector_amp = g[(amp_choice+'_m'+mass_choice+'_tw'+twist_choice)][0]
            V_matrix = calc_Lattice_Current(m_mother, E_daughter, xVector_amp, mass_choice)
            q_sqrd = calc_q_sqrd(m_mother,m_daughter, E_daughter)
            FF = (V_matrix) / (gv.sqrt(2*m_mother)*Three_p_daughter)
            return gv.sqrt(1/w0_a)*FF, q_sqrd, m_mother, m_daughter, E_daughter
        else:
            print('ERROR: Options for f_perpendicular are "XVnn" and "XsVnn". Cannot apply to {}'.format(amp_choice))
