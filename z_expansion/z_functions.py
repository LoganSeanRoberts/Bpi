import numpy as np
import gvar as gv
from matplotlib import pyplot as plt
import collections
from tabulate import tabulate

######################################
### z-expansion related functions ####
######################################
def calc_Daughter_3Momentum(twist, N_x): #twist, lattice length in lattice units. 
    ap_daughter = float(twist)*np.pi/N_x
    return ap_daughter

### Splitting independant of heavy quark mass.
### Following Functions are the "Higher order" funcitons that include functions within functions.  Those are defined later down the page

def z_expansion(q_sqrd, Hl_mass, pion_mass, t_0=0):  # Main funciton for mapping q^2 to analitical function z
    t_cut = (Hl_mass + pion_mass)**2  ## Regardless of Hpi or HsK, the Hl and pion are the mesons in the branch cut.
    z = (gv.sqrt(t_cut - q_sqrd) - gv.sqrt(t_cut - t_0))/(gv.sqrt(t_cut - q_sqrd) + gv.sqrt(t_cut - t_0))
    return z

def z_expansion_inverse(z, m_mother, m_daughter, t_0=0): #Inverse of above function from https://arxiv.org/pdf/0807.2722.pdf, origin of BCL parameterization
    t_plus = (m_mother + m_daughter)**2
    q_sqrd = t_plus - (t_plus - t_0)*((1+z)/(1-z))**2
    return q_sqrd

def calc_FF_Scalar_of_z(q_sqrd, z, M_pole, N_poly, a_scalar, chi_log, ap_term = 1): #Calc form factor in terms of q^2 and z. 
    pole_term = 1-(q_sqrd/M_pole**2)
    frac, Sum = chi_log/pole_term, 0
    for n in range(N_poly):
        Sum += a_scalar[n] * z**n
    return frac * Sum * ap_term

def calc_FF_Vector_of_z(q_sqrd, z, M_pole, N_poly, a_vector, chi_log, ap_term = 1): #Calc form factor in terms of q^2 and z.  
    pole_term = 1-(q_sqrd/M_pole**2)
    frac, Sum = chi_log/pole_term, 0
    for n in range(N_poly):
        Sum += a_vector[n] * (z**n - (n/N_poly)*(-1)**(n-N_poly) * z**N_poly)
    return frac * Sum * ap_term

def calc_FF_Tensor_of_z(q_sqrd, z, M_pole, N_poly, a_tensor, chi_log, ap_term = 1): #Calc form factor in terms of q^2 and z. 
    pole_term = 1-(q_sqrd/M_pole**2)
    frac, Sum = chi_log/pole_term, 0
    for n in range(N_poly):
        Sum += a_tensor[n] * (z**n - (n/N_poly)*(-1)**(n-N_poly) * z**N_poly)
    return frac * Sum * ap_term

# Now to wrap the three above equations into one, wher ewe chose which current FF we want based on the argument 'curr'
# ap term has been added to maybe account for momentum disc. effects.  is want to include it, ap term should be form (1 + (ap/pi)^2)
def calc_FF_of_z(q_sqrd, z, M_pole, N_poly, a_ns, chi_log, curr, ap_term = 1):
    #pole_term_dict = {}
    if curr == '0': FF_of_z = calc_FF_Scalar_of_z(q_sqrd, z, M_pole, N_poly, a_ns, chi_log, ap_term)
    elif curr in ('+', '+t', '+x'): FF_of_z = calc_FF_Vector_of_z(q_sqrd, z, M_pole, N_poly, a_ns, chi_log, ap_term)
    elif curr == 'T': FF_of_z = calc_FF_Tensor_of_z(q_sqrd, z, M_pole, N_poly, a_ns, chi_log, ap_term)
    else: print('Invalid "curr" argument in calc_FF_of_z function.  "curr" should be in ["0", ["+", or "+t", "+x"], "T"]')
    return FF_of_z

### This set of functions can be used to assign the priors to the coefficients and exponents in the z expansion equations
# For ρn and dijkln we take values of 0.0(1.0) in all cases except for terms which are O(a2). We know such terms are highly suppressed in the HISQ
# action because it is a2-improved [https://arxiv.org/abs/hep-lat/0610092], so we take a reduced width prior of 0.0(0.3) for di10ln and di01ln terms.
# expect curr to be in {0,+,T}
def assign_Priors_d_ijkln(p_dict, Npoly, Nijkl, curr, Prior_widener = 1.0, bulk_prior_width = 1.0, Oa2_prior_width = 0.3, cv_neq_0_debug = False):
    for i in range(Nijkl[0]):
        for j in range(Nijkl[1]):
            for k in range(Nijkl[2]):
                for l in range(Nijkl[3]):
                    for n in range(Npoly):
                        curr_str = curr
                        if n == 0:
                            if curr != 'T':
                                curr_str = '0/+'
                        #reassigning priors for O(a^2) terms
                        #debug line
                        if cv_neq_0_debug == True: cv, cv_bulk = np.random.normal(scale= Oa2_prior_width), np.random.normal(scale= bulk_prior_width)
                        else: cv, cv_bulk = 0.0, 0.0
                        if j == 1 and k == 0:
                            p_dict['d^{5}_{0}{1}{2}{3}{4}'.format(i,j,k,l,n,curr_str)] = gv.gvar(cv, Oa2_prior_width*Prior_widener)
                        elif j== 0 and k == 1:
                            p_dict['d^{5}_{0}{1}{2}{3}{4}'.format(i,j,k,l,n,curr_str)] = gv.gvar(cv, Oa2_prior_width*Prior_widener)
                        else:
                            p_dict['d^{5}_{0}{1}{2}{3}{4}'.format(i,j,k,l,n,curr_str)] = gv.gvar(cv_bulk, bulk_prior_width*Prior_widener)
    return p_dict

# We therefore take a prior P [ζ0] = 1.5(5) as a common prior for the a_0 coefficients but set ζ neq =0 = 0 for the other a_n
def assign_Priors_zeta(p_dict, Npoly, Prior_widener = 1.0, zeta_0_cv = 1.5, zeta_0_width = 0.5, test_n_neq_0 = False):
# ---- forcing zeta to be 1.5 for testing
#def assign_Priors_zeta(p_dict, Npoly, Prior_widener = 1.0, zeta_0_cv = 1.5, zeta_0_width = 0.0000000000001, test_n_neq_0 = False):
    p_dict['zeta_0'] = gv.gvar(zeta_0_cv, zeta_0_width*Prior_widener)
    if test_n_neq_0 == True:
        for n in range(Npoly):
            if n != 0:
                p_dict['zeta_{}'.format(n)] = gv.gvar(zeta_0_cv, zeta_0_width*Prior_widener)
    return p_dict

#Assigning rho priors, use 0.0(1.0).  Can adjust for stability testing
# expect curr to be in {0,+,T}
def assign_Priors_rho(p_dict, Npoly, curr, Prior_widener = 1.0, rho_cv = 0.0, rho_width = 1.0, cv_neq_0_debug = False):
    for n in range(Npoly):
        curr_str = curr
        if n == 0:
            if curr != 'T':
                curr_str = '0/+'
        if cv_neq_0_debug == True: rho_cv = np.random.normal(scale = rho_width)
        p_dict['rho^{0}_{1}'.format(curr_str, n)] = gv.gvar(rho_cv, rho_width*Prior_widener)
    return p_dict

#Assigning the priors for c^{sea,valence}_{l,s,c}
#The prior for the daughter strange quark mistuning parameter in N , cval s,n, is taken as 0.0(1.0) for each n and
# each form factor. This size is based on the variation seen between B → K and B → π form factors [https://arxiv.org/abs/1510.02349]
# We expect smaller effects from sea quark mass mistuning and so take the csea s,n and csea l,n parameters to have priors of 0.0(0.5) and
# the csea c,n parameters to have prior 0.0(1)
# expect curr to be in {0,+,T}
def assign_Priors_c(p_dict, N_poly, curr, Prior_widener = 1.0, cval_s_width = 1.0, csea_sl_width = 0.5, csea_c_width = 0.1, cv_neq_0_debug = False):
    for n in range(N_poly):
        curr_str = curr
        if n == 0:
            if curr != 'T':
                curr_str = '0/+'
        # cv neq 0 debug line for fit funciton testing
        if cv_neq_0_debug == True: 
            p_dict['cval^{0}_s{1}'.format(curr_str,n)] = gv.gvar(np.random.normal(scale = cval_s_width),cval_s_width*Prior_widener)
            p_dict['csea^{0}_s{1}'.format(curr_str,n)] = gv.gvar(np.random.normal(scale = csea_sl_width),csea_sl_width*Prior_widener)
            #dont need seperate cval_l because mval_l = msea_l
            p_dict['csea^{0}_l{1}'.format(curr_str,n)] = gv.gvar(np.random.normal(scale = csea_sl_width),csea_sl_width*Prior_widener)
            p_dict['csea^{0}_c{1}'.format(curr_str,n)] = gv.gvar(np.random.normal(scale = csea_c_width),csea_c_width*Prior_widener)
        else:
            p_dict['cval^{0}_s{1}'.format(curr_str,n)] = gv.gvar(0,cval_s_width*Prior_widener)
            p_dict['csea^{0}_s{1}'.format(curr_str,n)] = gv.gvar(0,csea_sl_width*Prior_widener)
            #dont need seperate cval_l because mval_l = msea_l
            p_dict['csea^{0}_l{1}'.format(curr_str,n)] = gv.gvar(0,csea_sl_width*Prior_widener)
            p_dict['csea^{0}_c{1}'.format(curr_str,n)] = gv.gvar(0,csea_c_width*Prior_widener)
    return p_dict
        
def assign_Priors_g_Coupling(p_dict, Prior_widener = 1.0, g_inf_cv = 0.48, g_inf_width = 0.11, C1_cv = 0.5, C1_width = 1.0, C2_cv = 0.0, C2_width = 3.0, cv_neq_0_debug = False):
    p_dict['g_inf'] =gv.gvar(g_inf_cv, g_inf_width*Prior_widener)
    p_dict['C1'] = gv.gvar(C1_cv, C1_width*Prior_widener)
    #cv neq 0 debug line
    if cv_neq_0_debug == True: C2_cv = np.random.normal(scale = C2_width)
    p_dict['C2'] = gv.gvar(C2_cv, C2_width*Prior_widener)
    return p_dict

#Now to neatly package all the above functions together
def assign_Full_Prior_Set(Npoly, curr, Nijkl):
    p_dict = {}
    p_dict = assign_Priors_d_ijkln(p_dict, Npoly, Nijkl, curr)
    p_dict = assign_Priors_rho(p_dict, Npoly, curr)
    p_dict = assign_Priors_zeta(p_dict, Npoly)
    p_dict = assign_Priors_c(p_dict, Npoly, curr)
    p_dict = assign_Priors_g_Coupling(p_dict)
    return p_dict
#used to pick prior for dijkl coeff from prior collection
def pick_d_from_p_dict(p_dict, i,j,k,l,n, curr):
    picked_d = p_dict['d^{5}_{0}{1}{2}{3}{4}'.format(i,j,k,l,n, curr)]
    return picked_d

### Heavy Quark mass dependant terms
# This includes the chiral logarithm term, L, which is includes meson chiral terms x_meson, coupling term "g" between H and light mesons
# (5.63 * m_strange) is ~ (4 pi f_pi)^2

def calc_X_Meson(M_phys, decay_const):
    return M_phys**2 / (4*np.pi*decay_const)**2

# Since not all of these meson masses are available in our calculation we use leading-order chiral perturbation theory
# to rescale meson masses in proportion to the masses of the quarks they contain
# 'daughter quark is always light, this in inverse of Will's paper
def calc_X_pi(m_light, mtuned_strange):
    return 2 * m_light / (5.63 * mtuned_strange)

#The following equation has the same form as above, but is included to avoid confusion down the road
#function might be defuct with ratio mphys light to mphys strange = 1 / 27.10
def calc_Xphys_pi(mphys_light, mphys_strange):
    return 2 * mphys_light / (5.63 * mphys_strange)

#The follwing equation is the general ratio of pdg quark masses regardless of scheme, will be used to get mtuned_light from mtuned_strange
##Note: the superscript "phys" here does not refer to the mass of the valence quark masses on the f-phys or sf-phys ensembles. 
#def calc_mphys_light(mphys_strange):
#    return mphys_strange / 27.18
#    #return mphys_strange / gv.gvar(27.18,0.10)

'''def calc_X_kaon(m_light, m_spectator, m_strange):
    return  (m_light + m_spectator) / (5.63 * m_strange)'''
#13/06/25 change made to not account for any scaling and instead just be constant meson mass
def calc_X_kaon(m_light, m_strange):
    return  (m_light + m_strange) / (5.63 * m_strange)

'''def calc_X_eta(m_spectator, m_strange):
    return 2 * m_spectator / (5.63 * m_strange)'''
#13/06/25 change made to not account for any scaling and instead just be constant meson mass
def calc_X_eta(m_strange):
    return 2 * m_strange / (5.63 * m_strange)

# Coupling term g(M_H), takes arguments priors C1 and C2 .  Pring g is useful for debugging
def calc_g_coupling(g_inf, M_mother, Lambda_QCD, C1, C2): #, print_g = False):
    #g_inf = gv.gvar(0.48,0.11) #https://arxiv.org/abs/hep-lat/0310050, allow this to be fit parameter
    g = g_inf + C1*Lambda_QCD/M_mother + C2*(Lambda_QCD**2)/(M_mother**2)
    #if print_g == True: print(g)
    return g

'''#chiral logarithm term, L which utilzies above functions, adds finite volume term which comes from eq47 of https://arxiv.org/pdf/hep-lat/0111051 ? 
def calc_BsK_chi_log_L(m_daughter, m_strange, m_spectator, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV): #m_daughter refers to quark mass, as do other m's. M is for meson
    #print('g_inf = {0}, M_mother = {1}, LambdaQCD = {2}, C1 = {3}, C2 = {4}'.format(g_inf, M_mother, Lambda_QCD, C1, C2))
    g = calc_g_coupling(g_inf, M_mother, Lambda_QCD, C1, C2)
    X_pi = calc_X_pi(m_daughter, m_strange)
    X_kaon = calc_X_kaon(m_daughter, m_spectator, m_strange)
    X_eta = calc_X_eta(m_spectator, m_strange)
    L = 1 - ((3/8) * X_pi * (gv.log(X_pi)+delta_FV)) - ((0.25 + 1.5*g**2) * X_kaon * gv.log(X_kaon)) - (((1/24)+0.5*g**2) * X_eta * gv.log(X_eta))
    return L
#old form
# L = 1 - ((9*g**2)/8) * X_pi * (gv.log(X_pi)+delta_FV) - ((0.5 + 0.75*g**2) * X_kaon * gv.log(X_kaon)) - (((1/6)+g**2/8) * X_eta * gv.log(X_eta))
    

#chiral logarithm term, L which utilzies above functions, adds finite volume term which comes from eq47 of https://arxiv.org/pdf/hep-lat/0111051 ? 
def calc_Bpi_chi_log_L(m_daughter, m_strange,m_spectator, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV): #m_daughter refers to quark mass, as do other m's. M is for meson
    #print('g_inf = {0}, M_mother = {1}, LambdaQCD = {2}, C1 = {3}, C2 = {4}'.format(g_inf, M_mother, Lambda_QCD, C1, C2))
    g = calc_g_coupling(g_inf, M_mother, Lambda_QCD, C1, C2)
    X_pi = calc_X_pi(m_daughter, m_strange)
    X_kaon = calc_X_kaon(m_daughter, m_spectator, m_strange)
    X_eta = calc_X_eta(m_spectator, m_strange)
    L = 1 - ((3/8) + (9/8)*g**2) * X_pi * (gv.log(X_pi)+delta_FV)
    #L = 1 - ((3/8) + (9/8)*g**2) * X_pi * (gv.log(X_pi)+delta_FV) - ((0.25 + 0.75*g**2) * X_kaon * gv.log(X_kaon)) - (((1/24)+g**2/8) * X_eta * gv.log(X_eta))
    return L'''
#13/06/25 change made to not account for any scaling and instead just be constant meson mass
#chiral logarithm term, L which utilzies above functions, adds finite volume term which comes from eq47 of https://arxiv.org/pdf/hep-lat/0111051 ? 
def calc_HsK_chi_log_L(X_pi, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV, print_Xes = False): #m_light refers to quark mass, as do other m's. M is for meson
    #print('g_inf = {0}, M_mother = {1}, LambdaQCD = {2}, C1 = {3}, C2 = {4}'.format(g_inf, M_mother, Lambda_QCD, C1, C2))
    #g = calc_g_coupling(g_inf, M_mother, Lambda_QCD, C1, C2)
    #X_pi = calc_X_pi(m_light, m_strange)
    #X_kaon = calc_X_kaon(m_light, m_strange)
    #X_eta = calc_X_eta(m_strange)
    #if print_Xes == True: print('X_pi*logX_pi =', X_pi*gv.log(X_pi), 'X_kaon*logX_kaon =', X_kaon*gv.log(X_kaon), 'X_eta*logX_eta =', X_eta*gv.log(X_eta), )
    #if print_Xes == True: print('X_pi chunk =', (3/8) * X_pi * (gv.log(X_pi)+delta_FV), 'X_kaon chunk =', (0.25 + 1.5*g**2) * X_kaon * gv.log(X_kaon), 'X_eta chunk =', ((1/24)+0.5*g**2) * X_eta * gv.log(X_eta))
    L = 1 - ((3/8) * X_pi * (gv.log(X_pi)+delta_FV)) #- ((0.25 + 1.5*g**2) * X_kaon * gv.log(X_kaon)) - (((1/24)+0.5*g**2) * X_eta * gv.log(X_eta))
    return L
#old form for B to K ONLY vvv
# L = 1 - ((9*g**2)/8) * X_pi * (gv.log(X_pi)+delta_FV) - ((0.5 + 0.75*g**2) * X_kaon * gv.log(X_kaon)) - (((1/6)+g**2/8) * X_eta * gv.log(X_eta))
    

#chiral logarithm term, L which utilzies above functions, adds finite volume term which comes from eq47 of https://arxiv.org/pdf/hep-lat/0111051 ? 
def calc_Hpi_chi_log_L(X_pi, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV, print_Xes = False): #m_light refers to quark mass, as do other m's. M is for meson
    #print('g_inf = {0}, M_mother = {1}, LambdaQCD = {2}, C1 = {3}, C2 = {4}'.format(g_inf, M_mother, Lambda_QCD, C1, C2))
    g = calc_g_coupling(g_inf, M_mother, Lambda_QCD, C1, C2)
    #X_pi = calc_X_pi(m_light, m_strange)
    #X_kaon = calc_X_kaon(m_light, m_strange) #defunct because we use su2 and arent varying to strange quark
    #X_eta = calc_X_eta(m_strange)
    if print_Xes == True: print('X_pi*logX_pi =', X_pi*gv.log(X_pi))
    L = 1 - ((3/8) + (9/8)*g**2) * X_pi * (gv.log(X_pi)+delta_FV)
    #L = 1 - ((3/8) + (9/8)*g**2) * X_pi * (gv.log(X_pi)+delta_FV) - ((0.25 + 0.75*g**2) * X_kaon * gv.log(X_kaon)) - (((1/24)+g**2/8) * X_eta * gv.log(X_eta))
    return L


#Implementing choice of BsK or Bpi
def calc_chi_log_L(call_str, X_pi, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV, print_Xes = False):
    if call_str == 'Hpi':
        L = calc_Hpi_chi_log_L(X_pi, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV, print_Xes)
    elif call_str == 'HsK':
        L = calc_HsK_chi_log_L(X_pi, g_inf, M_mother, Lambda_QCD, C1, C2, delta_FV, print_Xes)
    else: print('Invalid Call String in chi log function, should be either Hpi or HsK')
    return L

##### Mistuning effects for other quark masses Related Functions
#calc mtuned strange from lattice valence s quark, pdg mass of Eta s physical mass, and eta_s from lattice fit which uses same valence s quark
# get M_eta_s from https://arxiv.org/pdf/2010.07980
def calc_mtuned_strange(m_strange, Mphys_eta_s, M_eta_s):
    return m_strange * (Mphys_eta_s / M_eta_s)**2

#This next equation derives mtuned_light from mtuned_strange in accordance with the same ratio of mphys_strange / mphys_light = 27.18(10)
def calc_mtuned_light(m_strange, Mphys_eta_s, M_eta_s, ms_ml_ratio):
    mtuned_strange = calc_mtuned_strange(m_strange, Mphys_eta_s, M_eta_s)
    mtuned_light = mtuned_strange / ms_ml_ratio # CHANGE MADE JUNE 23 2025
    return mtuned_light

#The following equation calculating quark mistuning effects is valid for both strange and light quarks (assume can use either msea_s or
# mval_s depending on which delta you want - delta^sea_s or delta^val_s)
#From will's B to K: "Dividing by mtuned s here makes this a physical, scale-independent ratio and the factor of 10 matches this 
#                     approximately to the usual expansion parameter in chiral perturbation theory"
def calc_delta_q(m_quark, mtuned_quark, mtuned_strange):
    return ((m_quark - mtuned_quark)/(10*mtuned_strange))

#Following covers delta charm.  msea charm comes from lattice table,
# and mtuned_charm is fixed from the ηc meson mass from https://arxiv.org/pdf/2005.01845
#Otherwise maybe from table II in https://arxiv.org/pdf/1408.4169
#This function might be unnecessaty because "These values, on each ensemble, 
# correspond well with the lowest heavy valence mass that we have use" - Wills paper B to K
def calc_delta_charm_sea(msea_charm, mtuned_charm):
    return ((msea_charm - mtuned_charm)/mtuned_charm)

#Finally the larger "mass-mistuning term, N", which is unique for every n and choice of scalar, vector, and tensor
# delta terms in equation are independent of n and 0,+,T choice, coefficients c are not, and are fir parameters with priors
# csea_l is degenerate with cval_l, so instead just use 2*csea_l.  Meanwhile, we have no valence c quarks so there is no cval_c.
def calc_Mass_Mistune_Ncurr_n(cval_s, csea_s, csea_l, csea_c, deltaval_s, deltasea_s, delta_l, deltasea_c):
    N = cval_s*deltaval_s + csea_s*deltasea_s + 2*csea_l*delta_l + csea_c*deltasea_c
    return N

#For finding the mass of the Heavy vector meson H*, we use the following formula (eq 13 in wills b to k)
#Should check that this equation holds true for both B to pi AND Bs to K.  Might be case that Delta equations 
# need to be changed from MPhysHstar_s to Mphys_Hstar.
## Physical masses should be isospin averages ->  Bphys = (B0 +B±)/2, Dphys = (D0 + D±)/2 
### From PDG
def calc_M_Hstar(M_H, Mphys_D, Mphys_Dstar, Mphys_B, Mphys_Bstar):
    Delta_D = Mphys_Dstar - Mphys_D
    Delta_B = Mphys_Bstar - Mphys_B
    line1 = M_H + (Mphys_D / M_H) * Delta_D
    line2 = (Mphys_B/M_H)*((M_H - Mphys_D)/(Mphys_B - Mphys_D))*(Delta_B-((Mphys_D / Mphys_B)*Delta_D))
    return line1 + line2

### Finally putting these functions together to calculate the coeficcient a for every n, and choice of 0,+,T
#zeta, rho, d_ijkl are priors, and Ntuned contains priors as well
# expect this function to be a loop that looks like:
#   for curr in ['0', '+', 'T']:
#       for n in range(Npoly):
def calc_a_coeff(M_D, M_H, zeta, rho, Ntuned, N_i, N_j, N_k, N_l, p_dict, Lambda_QCD, am_h, X_pi, Xphys_pi, n, a, curr_str, print_parts = False):
    #print(X_pi - Xphys_pi)
    line1 = (M_D/M_H)**zeta * (1 + rho*gv.log(M_H/M_D)) * (1 + Ntuned)
    #altline = (M_D/M_H)**zeta * (1 + rho*gv.log(M_H/M_D)) * (1 + Ntuned)
    Sigma_sum = 0
    altsum = 0
    for i in range(N_i):
        for j in range(N_j):
            for k in range(N_k):
                for l in range(N_l):
                    d = pick_d_from_p_dict(p_dict, i,j,k,l,n, curr_str)
                    Sigma_sum += d * (Lambda_QCD/M_H)**i * (am_h/np.pi)**(2*j) * (a*Lambda_QCD/np.pi)**(2*k) * (X_pi - Xphys_pi)**l
                    #altsum +=  d * (Lambda_QCD/M_H)**i * (am_h/np.pi)**(2*j) * (a*Lambda_QCD/np.pi)**(2*k) * (X_pi - Xphys_pi)**l
    
    #if print_parts == True: 
        #for l in range(N_l): print ((X_pi - Xphys_pi)**l)
        #print(X_pi, Xphys_pi, Ntuned)
        #print(gv.evalcorr(np.array([line1, altsum])))
        #print('line1 = {}, Sigma_sum = {}, a^+_0 = {}'.format(altline, altsum, altline*altsum))
        #print(' [line1,       Sigma_sum] Correlation Matrix')
        #print(gv.evalcorr(np.array([altline,altsum])))
        #print('###############################################')
    return line1 * Sigma_sum

#This function is used to caclulate the continuois version of the above function
#note Dphys argument should be Dsphys for BsK

#Function now defuct cus a_phys is handled within main fit func
'''def calc_a_coeff_phys(M_Dphys, M_H, zeta, rho, N_i, Lambda_QCD, p_dict, n, curr_str):
    line1 = (M_Dphys/M_H)**zeta * (1+rho*gv.log(M_H/M_Dphys))
    Sigma_sum = 0
    for i in range(N_i):
        d = pick_d_from_p_dict(p_dict, i, 0, 0, 0, n, curr_str)
        temp_term = d * (Lambda_QCD/M_H)**i 
        Sigma_sum += temp_term
    return line1 * Sigma_sum'''

#### function to read in specific pickle file.  File location might be something like "form_factors/FF_gvdumps"
# Expect call_tag to be "Hpi" or "HsK".  Expect ensemble tag to be F, Fp, SF, SFp, or UF
def load_gvdump(file_loc, call_str, ensemble_str):
    filename = './{0}/FF-gvdump_{1}_{2}.pickle'.format(file_loc, call_str, ensemble_str)
    g = gv.load(filename)
    return g


#We can use this function to add physical  meson masses to an ensemble mass dictionary.  Useful in above functions
## Physical masses should be isospin averages ->  Bphys = (B0 +B±)/2, Dphys = (D0 + D±)/2 
def add_Physical_Masses_in_GeV():
    mass_dict = {}
    #pi and K phys are used in continuum z expansion
    mass_dict['D_phys'] = (gv.gvar('1.86966(5)') + gv.gvar('1.86484(5)'))/2 #GeV
    mass_dict['D*_phys'] = gv.gvar('2.00685(5)') #GeV
    #mass_dict['D*_s_phys'] = gv.gvar('2.1122(4)') #GeV
    mass_dict['B_phys'] = (gv.gvar('5.27941(7)') + gv.gvar('5.27972(8)'))/2 #GeV
    mass_dict['B*_phys'] = gv.gvar('5.32475(20)') #GeV
    #mass_dict['B*_s_phys'] = gv.gvar('5.4158(15)') #GeV
    #mass_dict['eta_s_phys'] = gv.gvar('0.6885(22)') #GeV
    #mass_dict['D*_0s_phys'] = gv.gvar('2.3178(5)') #GeV
    mass_dict['D*_0_phys'] = gv.gvar('2.343(10)') #GeV
    mass_dict['Delta_M_H'] = mass_dict['D*_0_phys'] - mass_dict['D_phys'] #GeV
    mass_dict['pi_phys'] = (gv.gvar('139.57039(18)') + gv.gvar('134.9768(5)'))/(2*1000) #Taking average of the two and converting to GeV
    mass_dict['K_phys'] = (gv.gvar('493.677(15)') + gv.gvar('497.611(13)'))/(2*1000) #Taking average of the two and converting to GeV
    #mass_dict['D_phys'] = (gv.gvar('1.86966(5)') + gv.gvar('1.86484(5)'))/2
    mass_dict['Ds_phys'] = gv.gvar('1.96835(7)') #There is no Neutral charged Ds meson.
    #mass_dict['B_phys'] = (gv.gvar('5.27941(7)') + gv.gvar('5.27972(8)'))/2 #GeV
    mass_dict['Bs_phys'] = gv.gvar('5.36693(10)')
    return mass_dict


# Working print results function for simple z fit script
def print_results(fit, Npoly, Ni, currs3, call_str, Lambda_QCD):#, params):
    print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
    print('')
    #Error budgt comparison
    '''p = fit.p
    inputs = {}
    Bprint_dict = {}
    Dprint_dict = {}
    Btable_print = []
    Dtable_print = []'''

    '''print('-'*28)
    print('B{} phys Eval Corr Table...'.format(s[0]))
    #correlation matrix printing
    evalcorr = gv.evalcorr(Bprint_dict)
    numlist, keylist = [], []
    for key in Bprint_dict:
        keylist.append(key)
    for entry in evalcorr:
        numlist.append(round(evalcorr[entry][0][0],5))
    a = len(Bprint_dict)
    arr = np.empty((a,a+1), dtype = object)
    etick = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if j == 0: arr[i, j] = keylist[i]
            else:
                arr[i, j] = numlist[etick]
                etick+=1

    print(tabulate(arr,headers=['-']+keylist,tablefmt="simple_grid"))
    print(gv.fmt_errorbudget(Bprint_dict, p, percent=True))'''
def format_corr_mtrx(corr_matrix_dict):
    new_dict = {}
    seq_dict = {}
    seq = ['a^+/0_0', 'a^0_1', 'a^0_2', 'a^+_1', 'a^+_2', 'a^T_0', 'a^T_1', 'a^T_2', 'H*0', 'H*', 'chi_log']
    for key in corr_matrix_dict:
        nkey = key.split('-')[1]
        if type(corr_matrix_dict[key]) != type([]):
            new_dict[nkey] = corr_matrix_dict[key]
        else:
            i = 0
            for item in corr_matrix_dict[key]:
                new_dict['{}_{}'.format(nkey,i)] = corr_matrix_dict[key][i]
                i += 1
    new_dict['a^+/0_0'] = new_dict['a^0_0']
    new_dict.pop('a^0_0')
    new_dict.pop('a^+_0')
    for key in seq:
        seq_dict[key] = new_dict[key]
    corr_matrix_dict = seq_dict
    evalcorr = gv.evalcorr(corr_matrix_dict)
    numlist, keylist = [], []
    for key in corr_matrix_dict:
        keylist.append(key)
    for entry in evalcorr:
        numlist.append(round(evalcorr[entry][0][0],5))
    a = len(corr_matrix_dict)
    arr = np.empty((a,a+1), dtype = object)
    etick = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if j == 0: arr[i, j] = keylist[i]
            else:
                arr[i, j] = numlist[etick]
                etick+=1
    table = tabulate(arr,headers=['-']+keylist,tablefmt="simple_grid")
    return table


#DEFUNCT FUNCTION - Now continuum fucntionality is built into main fit function
'''#physical FF funtions for either Hpi or HsK.  Should be function of M_H, among other things
#Use seperate funtions for either Hpi or HsK
#function works like eq 24 and 25 in wills BtoK
def Hpi_FFcont(M_H, curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, post_dict):
    FFofq2, polexFFofz = {}, {} #Output dictionaries
    #First we can calculate chi log term, which is funciton of M_H. the values m_light and m_strange are only used in a ration, so we can say..
    m_strange = post_dict['ms-ml_phys_ratio'] #reminder, its just the ratio that matters in the chi log calc
    m_light = 1/m_strange
    chi_log =  calc_chi_log_L('Hpi', m_light, m_strange, post_dict['g_inf'], M_H , Lambda_QCD, post_dict['C1'], post_dict['C2'], 0.0)    
    for curr_tuple in curr_tuple_list:
        curr = curr_tuple[0]
        #polemass is always H*, regardless of Hpi or HsK
        if curr == '0': polemass = M_H + post_dict['Delta_M_H']
        else: polemass = calc_M_Hstar(M_H, post_dict['D_phys'], post_dict['D*_phys'], post_dict['B_phys'], post_dict['B*_phys'])
        #calculating a coeffs
        a_ns = []
        for n in range(Npoly):
            curr_str = curr
            if n == 0:
                zeta = post_dict['zeta_0']
                if curr != 'T':
                    curr_str = '0/+'
            else: zeta = 0
            a_n = calc_a_coeff_phys(post_dict['D_phys'], M_H, zeta, post_dict['rho^{0}_{1}'.format(curr_str,n)], N_i, Lambda_QCD, post_dict, n, curr_str)
            a_ns.append(a_n)
        #calculating FF values, which are the y-axis points against either z or q2
        temp_FF_list = []
        z_range = []
        for q2 in q2_range:
            z = z_expansion(q2, M_H, post_dict['pi_phys'])
            z_range.append(z.mean) #this is just to get our xvar on polemass*FF against z plot
            temp_FF = calc_FF_of_z(q2, z, polemass, Npoly, a_ns, chi_log, curr)
            temp_FF_list.append(temp_FF)
        z_range = np.array(z_range)#making this an np array for handy plotting purposes
        FFofq2['FFcont {}-curr'.format(curr_tuple[1])] = np.array(temp_FF_list)
        polexFFofz['polexFFcont {}-curr'.format(curr_tuple[1])] = np.array(temp_FF_list) * (1 - q2_range/polemass**2)
    return FFofq2, polexFFofz, z_range  
    
#physical FF funtions for either Hpi or HsK.  Should be function of M_H, among other things
#Use seperate funtions for either Hpi or HsK
#function works like eq 24 and 25 in wills BtoK
#Is a funciton of both M_H and M_Hs
def HsK_FFcont(M_H, M_Hs, curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, post_dict):
    FFofq2, polexFFofz = {}, {} #Output dictionaries
    #First we can calculate chi log term, which is funciton of M_H. the values m_light and m_strange are only used in a ration, so we can say..
    m_strange = post_dict['ms-ml_phys_ratio'] #reminder, its just the ratio that matters in the chi log calc
    m_light = 1/m_strange
    chi_log =  calc_chi_log_L('HsK', m_light, m_strange, post_dict['g_inf'], M_H , Lambda_QCD, post_dict['C1'], post_dict['C2'], 0.0)    
    for curr_tuple in curr_tuple_list:
        curr = curr_tuple[0]
        #polemass is always H*, regardless of Hpi or HsK
        if curr == '0': polemass = M_H + post_dict['Delta_M_H']
        else: polemass = calc_M_Hstar(M_H, post_dict['D_phys'], post_dict['D*_phys'], post_dict['B_phys'], post_dict['B*_phys'])
        #calculating a coeffs
        a_ns = []
        for n in range(Npoly):
            curr_str = curr
            if n == 0:
                zeta = post_dict['zeta_0']
                if curr != 'T':
                    curr_str = '0/+'
            else: zeta = 0
            a_n = calc_a_coeff_phys(post_dict['Ds_phys'], M_Hs, zeta, post_dict['rho^{0}_{1}'.format(curr_str,n)], N_i, Lambda_QCD, post_dict, n, curr_str)
            a_ns.append(a_n)
        #calculating FF values, which are the y-axis points against either z or q2
        temp_FF_list = []
        z_range = []
        for q2 in q2_range:
            z = z_expansion(q2, M_H, post_dict['pi_phys']) # again, the z mapping comes from t_cut, which comes from Hpi
            z_range.append(z.mean) #this is just to get our xvar on polemass*FF against z plot
            temp_FF = calc_FF_of_z(q2, z, polemass, Npoly, a_ns, chi_log, curr)
            temp_FF_list.append(temp_FF)
        z_range = np.array(z_range)#making this an np array for handy plotting purposes
        FFofq2['FFcont {}-curr'.format(curr_tuple[1])] = np.array(temp_FF_list)
        polexFFofz['polexFFcont {}-curr'.format(curr_tuple[1])] = np.array(temp_FF_list) * (1 - q2_range/polemass**2)
    return FFofq2, polexFFofz, z_range  

#Nesting the above functions for fewer "if" statements in main script
def FFcont(call_str, M_H, curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, post_dict, M_Hs = None):
    if call_str == 'Hpi': 
        FFofq2, polexFFofz, z_range = Hpi_FFcont(M_H, curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, post_dict)
    elif call_str == 'HsK':
        FFofq2, polexFFofz, z_range = HsK_FFcont(M_H, M_Hs, curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, post_dict)
    else: print('Invalid call_str')
    return FFofq2, polexFFofz, z_range
'''

#func not used/needed currently
def calc_Daughter_3Momentum(twist, N_x): #twist, lattice length in lattice units. 
    ap_daughter = float(twist)*np.pi/N_x
    return ap_daughter

#Now we can emplement some q_sqrd equations that either use the fit posterior E_daughter for non zero twist(more uncertaint)...
# or use the full lattice dispersion relation, which just uses the daughter meson mass, a twist, and the spacial length of the lattice
def calc_q_sqrd_v1(M_mother,M_daughter, E_daughter):#,twist, N_x):
    #q_sqrd = ((m_mother-E_daughter)**2)-(np.sqrt(3)*calc_Daughter_3Momentum(twist, N_x)**2)
    q_sqrd = M_mother**2 + M_daughter**2 - 2*M_mother*E_daughter
    return q_sqrd

def calc_q_sqrd_v2(M_mother,M_daughter, twist, N_x):
    if twist == '0.0': E_daughter = M_daughter
    else: E_daughter = gv.arccosh(1 + 0.5*(M_daughter**2) + 3*(1 - gv.cos(float(twist)*np.pi/N_x)))
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

# Gvar splitter
def gvar_splitter(gvars):
    means, errs = [] , []
    for y in gvars:
        means.append(y.mean)
        errs.append(abs(y.sdev))
    return means, errs

#########################################################
#Plotting functions for physical continuous form factors#
#########################################################
def get_Dpi_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p):
    q2_max = (p['D_phys'] - p['pi_phys'])**2
    q2_range = np.linspace(0.000001, q2_max.mean, 1000)
    FFofq2, polexFFofz, z_range = FFcont('Hpi', p['D_phys'], curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, p)
    return  FFofq2, q2_range, polexFFofz, z_range

def get_Bpi_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p):
    q2_max = (p['B_phys'] - p['pi_phys'])**2
    q2_range = np.linspace(0.000001, q2_max.mean, 1000)
    FFofq2, polexFFofz, z_range = FFcont('Hpi', p['B_phys'], curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, p)
    return  FFofq2, q2_range, polexFFofz, z_range

def get_DsK_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p):
    q2_max = (p['Ds_phys'] - p['K_phys'])**2
    q2_range = np.linspace(0.000001, q2_max.mean, 1000)
    FFofq2, polexFFofz, z_range = FFcont('HsK', p['D_phys'], curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, p, M_Hs = p['Ds_phys'])
    return  FFofq2, q2_range, polexFFofz, z_range

def get_BsK_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p):
    q2_max = (p['Bs_phys'] - p['K_phys'])**2
    q2_range = np.linspace(0.000001, q2_max.mean, 1000)
    FFofq2, polexFFofz, z_range = FFcont('HsK', p['B_phys'], curr_tuple_list, q2_range, Npoly, N_i, Lambda_QCD, p, M_Hs = p['Bs_phys'])
    return  FFofq2, q2_range, polexFFofz, z_range

def get_phys_cont_dicts(call_str, M, curr_tuple_list, Npoly, N_i, Lambda_QCD, p):
    if call_str == 'Hpi':
        if M == 'D':
            FFofq2, q2_range, polexFFofz, z_range = get_Dpi_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p)
            phys_tag = 'Dpi Phys.'
        elif M == 'B':
            FFofq2, q2_range, polexFFofz, z_range = get_Bpi_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p)
            phys_tag = 'Bpi Phys.'
        else: print('Invalid "M" arg in get_phys_cont_dict()')
    elif call_str == 'HsK':
        if M == 'D':
            FFofq2, q2_range, polexFFofz, z_range = get_Dpi_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p)
            phys_tag = 'DsK Phys.'
        elif M == 'B':
            FFofq2, q2_range, polexFFofz, z_range = get_Bpi_phys_continuous_dicts(curr_tuple_list, Npoly, N_i, Lambda_QCD, p)
            phys_tag = 'BsK Phys.'
        else: print('Invalid "M" arg in get_phys_cont_dict()')
    else: print('Invalid call_str in get_phys_cont_dict()')
    return FFofq2, q2_range, polexFFofz, z_range, phys_tag



########################################################
#Plotting functions for continuous lattice form factors#
########################################################
#This function gets the lines for each ensemble, current, and quark mass option and bundles them up nicely in a dicitonary
#   This dictionary is of the right sntax to feed into the fit funciton found in the main script
## REcent change to force eventual q2 range to go to zero
def get_lattice_aE_dict(p, ens_list, plot_scale_expander = (0.0,0.0)):
    aE_dict = {}
    for ens in ens_list:
        aE_min = p['E-daughters_{}'.format(ens)][0].mean
        aE_max = p['E-daughters_{}'.format(ens)][-1].mean
        aE_dict['aE_space_{}'.format(ens)] = np.linspace(aE_min*(1-plot_scale_expander[0]), aE_max*(1+plot_scale_expander[1]), 1000)#1000 here is enough for a smooth curve
    #The existence of this key lets the fit function know that we are generating a linspace of daughter meson energies, 
    # not using distinct corr. fit posteriors ascociated with distinct twists
    aE_dict['trigger'] = 0
    #adding key that triggers fit func to expand lattice q2 linespace to q2 = 0 if it doesnt already
    return aE_dict  

def get_Hpi_lattice_FFofz_dicts(ens_list, curr_tuple_list, twist_dict, q2_dict, aE_dict, Fit_dict, FF_data_dict, post_dict, non_gv_dict):
    z_lattice_disc_dict = {}
    polexFFofz_lattice_disc_dict = {}
    z_lattice_cont_dict = {}
    polexFFofz_lattice_cont_dict = {}
    q2s_linspace_dict = {}
    for ens in ens_list:
        a = non_gv_dict['{}_a'.format(ens)]
        for curr_tuple in curr_tuple_list:
            curr = curr_tuple[0]
            if curr == '0': tw_range = 0
            else: tw_range = 1
            mi = 0
            for mass in non_gv_dict['{}_masses'.format(ens)]:
                #continuous line implementation
                aM_mother = post_dict['M-mother_{}_{}'.format(ens, curr_tuple[1])][mi]
                aM_daughter = post_dict['E-daughters_{}'.format(ens)][0]
                if curr == '0': polemass = aM_mother/a + post_dict['Delta_M_H']
                else: polemass = calc_M_Hstar(aM_mother/a, post_dict['D_phys'], post_dict['D*_phys'], post_dict['B_phys'], post_dict['B*_phys'])
                zs_lattice = []
                polexFFofzs_lattice = []
                #discrete point implementation for polexFFofz plot.  
                #All we are doing is mapping q2 dict to z, and then taking FF_data_dict points and multiplying by the poleterm
                tw = 0
                for i in range(len(twist_dict['{0}_twists'.format(ens)]))[tw_range:]:
                    aq2 = q2_dict['q2s_{}_{}_m{}'.format(ens, curr_tuple[1], mass)][tw] 
                    q2 = aq2 / (a**2)
                    z = z_expansion(aq2, aM_mother, aM_daughter) #aq2 is fine here cus aq2 and and (aM_mother + aM_daughter)^2 are in same units
                    zs_lattice.append(z)
                    #print('pole term:', (1 - q2/polemass**2))
                    polexFFofz_lattice = FF_data_dict['FFs_{0}_{1}_m{2}'.format(ens, curr_tuple[1], mass)][tw] * (1 - q2/polemass**2)
                    polexFFofzs_lattice.append(polexFFofz_lattice)
                    tw += 1

                z_lattice_disc_dict['z_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = zs_lattice
                polexFFofz_lattice_disc_dict['pole*FF_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = polexFFofzs_lattice

                #continuous line implementation
                q2s_linspace = []
                zs_linspace = []
                polexFFofzs_cont = []
                for i in range(len(aE_dict['aE_space_{}'.format(ens)])):
                    aE_daughter = aE_dict['aE_space_{}'.format(ens)][i]
                    aq2 = calc_q_sqrd(aM_mother, aM_daughter, aE_daughter)
                    q2 = (aq2/a**2)
                    q2s_linspace.append(q2.mean)
                    polexFFofz_cont = Fit_dict['FFs_{0}_{1}_m{2}'.format(ens, curr_tuple[1], mass)][i] * (1 - q2/polemass**2)
                    polexFFofzs_cont.append(polexFFofz_cont.mean)
                    z = z_expansion(aq2, aM_mother, aM_daughter) #aq2 is fine here cus aq2 and and (aM_mother + aM_daughter)^2 are in same units
                    zs_linspace.append(z.mean)
                q2s_linspace_dict['q2s_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = q2s_linspace
                z_lattice_cont_dict['z_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = zs_linspace
                polexFFofz_lattice_cont_dict['pole*FF_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = polexFFofzs_cont
                mi += 1
    return polexFFofz_lattice_disc_dict, z_lattice_disc_dict, polexFFofz_lattice_cont_dict, z_lattice_cont_dict, q2s_linspace_dict

def get_HsK_lattice_FFofz_dicts(ens_list, curr_tuple_list, twist_dict, q2_dict, aE_dict, Fit_dict, FF_data_dict, post_dict, non_gv_dict):
    z_lattice_disc_dict = {}
    polexFFofz_lattice_disc_dict = {}
    z_lattice_cont_dict = {}
    polexFFofz_lattice_cont_dict = {}
    q2s_linspace_dict = {}
    for ens in ens_list:
        a = non_gv_dict['{}_a'.format(ens)]
        for curr_tuple in curr_tuple_list:
            curr = curr_tuple[0]
            if curr == '0': tw_range = 0
            else: tw_range = 1
            mi = 0
            for mass in non_gv_dict['{}_masses'.format(ens)]:
                #continuous line implementation
                aM_mother = post_dict['M-mother_{}_{}'.format(ens, curr_tuple[1])][mi]
                aM_daughter = post_dict['E-daughters_{}'.format(ens)][0]
                aHl_mass, apion_mass = post_dict['Hpi-M-mother_{}_{}'.format(ens, curr_tuple[1])][mi], post_dict['Hpi-E-daughters_{}'.format(ens)]
                if curr == '0': polemass = aHl_mass/a + post_dict['Delta_M_H']
                else: polemass = calc_M_Hstar(aHl_mass/a, post_dict['D_phys'], post_dict['D*_phys'], post_dict['B_phys'], post_dict['B*_phys'])
                zs_lattice = []
                polexFFofzs_lattice = []
                #discrete point implementation for polexFFofz plot.  
                #All we are doing is mapping q2 dict to z, and then taking FF_data_dict points and multiplying by the poleterm
                tw = 0
                for i in range(len(twist_dict['{0}_twists'.format(ens)]))[tw_range:]:
                    aq2 = q2_dict['q2s_{}_{}_m{}'.format(ens, curr_tuple[1], mass)][tw]
                    q2 = aq2 / (a**2)
                    z = z_expansion(aq2, aHl_mass, apion_mass) #aq2 is fine here cus aq2 and and (aM_mother + aM_daughter)^2 are in same units
                    zs_lattice.append(z)
                    polexFFofz_lattice = FF_data_dict['FFs_{0}_{1}_m{2}'.format(ens, curr_tuple[1], mass)][tw] * (1 - q2/polemass**2)
                    polexFFofzs_lattice.append(polexFFofz_lattice)
                    tw += 1

                z_lattice_disc_dict['z_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = zs_lattice
                polexFFofz_lattice_disc_dict['pole*FF_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = polexFFofzs_lattice

                #continuous line implementation
                q2s_linspace = []
                zs_linspace = []
                polexFFofzs_cont = []
                for i in range(len(aE_dict['aE_space_{}'.format(ens)])):
                    aE_daughter = aE_dict['aE_space_{}'.format(ens)][i]
                    aq2 = calc_q_sqrd(aM_mother, aM_daughter, aE_daughter)
                    q2 = (aq2/a**2)
                    q2s_linspace.append(q2.mean)
                    polexFFofz_cont = Fit_dict['FFs_{0}_{1}_m{2}'.format(ens, curr_tuple[1], mass)][i] * (1 - q2/polemass**2)
                    polexFFofzs_cont.append(polexFFofz_cont.mean)
                    z = z_expansion(aq2, aHl_mass, apion_mass) #aq2 is fine here cus aq2 and and (aM_mother + aM_daughter)^2 are in same units
                    zs_linspace.append(z.mean)
                q2s_linspace_dict['q2s_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = q2s_linspace
                z_lattice_cont_dict['z_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = zs_linspace
                polexFFofz_lattice_cont_dict['pole*FF_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = polexFFofzs_cont
                mi += 1
    return polexFFofz_lattice_disc_dict, z_lattice_disc_dict, polexFFofz_lattice_cont_dict, z_lattice_cont_dict, q2s_linspace_dict

#Similarly as above, we can bundle to two functions above together.  Need this function to get FFofq2 lattice linespaces as well
def get_lattice_FFofz_dicts(call_str, ens_list, curr_tuple_list, twist_dict, q2_dict, aE_dict, Fit_dict, FF_data_dict, post_dict, non_gv_dict):
    if call_str == 'Hpi':
        polexFFofz_lattice_disc_dict, z_lattice_disc_dict, polexFFofz_lattice_cont_dict, z_lattice_cont_dict, q2s_linspace_dict = get_Hpi_lattice_FFofz_dicts(ens_list, curr_tuple_list, twist_dict, q2_dict, aE_dict, Fit_dict, FF_data_dict, post_dict, non_gv_dict)
    elif call_str == 'HsK':
        polexFFofz_lattice_disc_dict, z_lattice_disc_dict, polexFFofz_lattice_cont_dict, z_lattice_cont_dict, q2s_linspace_dict = get_Hpi_lattice_FFofz_dicts(ens_list, curr_tuple_list, twist_dict, q2_dict, aE_dict, Fit_dict, FF_data_dict, post_dict, non_gv_dict)
    else: print('Invalid call_Str in get_lattice_FFofz_dicts')
    return polexFFofz_lattice_disc_dict, z_lattice_disc_dict, polexFFofz_lattice_cont_dict, z_lattice_cont_dict, q2s_linspace_dict

def plot_FFofq2_lattice(call_str, ens_list, curr_tuple_list, q2_dict, FF_data_dict, q2s_linspace_dict, Fit_dict, non_gv_dict, directory, colors, markers, plot_phys = False, incld_errbnds = False):
    if call_str =='Hpi': texstr = r'$H \rightarrow \pi$'
    elif call_str == 'HsK': texstr = r'$H_s \rightarrow K$'
    for ens in ens_list:
        if ens.endswith('p') == True: ratio = 27
        else: ratio = 5
        a = non_gv_dict['{}_a'.format(ens)]
        filename = '{}_{}_FFofq2'.format(call_str, ens)
        plt.figure(figsize=(12,9))
        plt.tight_layout(pad=0)
        alt_plot_order = [1,2,4,3]
        curr_tick = 0
        UF_masses = [0.21, 0.43, 0.64, 0.86]
        ffs_str = [r'$f_0$', r'$f_+(V^0)$',r'$f_+(V^1)$',r'$f_T$']
        for curr_pair in curr_tuple_list:
            mi = 0
            for mass in non_gv_dict['{}_masses'.format(ens)]:
                '''if curr_tick in (1,2):
                    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
                    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
                else:
                    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
                    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True'''
                plt.subplot(2,2,alt_plot_order[curr_tick])
                plt.tight_layout(pad=0.2)
                q2_linspace = q2s_linspace_dict['q2s_{}_{}_m{}'.format(ens, curr_pair[1], mass)]
                FF_latt_cont, FF_errs = gvar_splitter(Fit_dict['FFs_{0}_{1}_m{2}'.format(ens, curr_pair[1], mass)]) 
                x_means, x_errs = gvar_splitter(np.array(q2_dict['q2s_{}_{}_m{}'.format(ens, curr_pair[1], mass)])/(a**2))
                y_means, y_errs = gvar_splitter(FF_data_dict['FFs_{0}_{1}_m{2}'.format(ens, curr_pair[1], mass)])
                #if ens == 'UF': plt.errorbar(x_means,y_means,yerr=y_errs, label = r'${} \times m_b^\mathrm{{phys}}$'.format(UF_masses[mi]), ms = 8, marker = markers[mi], capsize = 2, linestyle = '', color = colors[mi], mfc='white')
                #else: 
                plt.errorbar(x_means,y_means,yerr=y_errs, label = r'$am_h={}$'.format(mass), ms = 8, marker = markers[mi], capsize = 2, linestyle = '', color = colors[mi], mfc='white')
                #if curr_pair[0] == '0':
                #    print(call_str, ens, mass, round((float(mass)/non_gv_dict['{}_a'.format(ens)])/4.18, 2))
                if incld_errbnds == False:
                    plt.plot(q2_linspace, FF_latt_cont, color = colors[mi], linestyle = ':')
                elif incld_errbnds == True:
                    plt.fill_between(q2_linspace, np.array(FF_latt_cont)+np.array(FF_errs), np.array(FF_latt_cont)-np.array(FF_errs), alpha = 0.2, color = colors[mi])
                mi += 1
            if plot_phys != False:
                FFphys_means, FFphys_errs = gvar_splitter(plot_phys[0]['FFcont {}-curr'.format(curr_pair[1])])
                q2_phys_linspace = plot_phys[1]
                plt.plot(q2_phys_linspace, FFphys_means, label = plot_phys[4], color = 'mediumslateblue', linestyle = ':')
            if curr_pair[0] == 'T': plt.legend(fontsize='18', frameon=False, handletextpad=0, loc = 'upper left')
            plt.minorticks_on()
            if curr_tick == 1:
                #plt.ylabel(r'$f_{}(q^2)$ $(V^0)$'.format(curr_pair[0]),labelpad=30, rotation=270)
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.tick_right()
            elif curr_tick == 2:
                #plt.ylabel(r'$f_{}(q^2)$ $(V^1)$'.format(curr_pair[0]),labelpad=30, rotation=270)
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.tick_right()
            plt.text(0.95, 0.05, ffs_str[curr_tick], ha="right", va="bottom", transform=plt.gca().transAxes, fontsize='20')
            #plt.ylabel(r'$f_{}(q^2)$'.format(curr_pair[0]))
            '''if ens == 'UF': 
                xtix = [-4, 0, 4, 8, 12, 16, 20]
                ytix = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                
                plt.xlim([-7,20.5])
                plt.xticks(xtix, xtix)
                if curr_tick >= 2: plt.xlabel(r'$q^2$ [GeV$^2$]')
                else: plt.xticks(xtix, [])
                if curr_tick != 0: 
                    plt.ylim([0.1,3.25])
                    plt.yticks(ytix, ytix)'''
            curr_tick +=1
        plt.suptitle(texstr+r' lattice form factors: {}-ensemble ($a={}$ fm, $m_s/m_l \approx {}$)'.format(non_gv_dict['{}_alt'.format(ens)], non_gv_dict['{}_a/fm'.format(ens)], ratio))
        plt.subplots_adjust(top=0.93)
        
        plt.savefig('./{}/{}.pdf'.format(directory, filename),format = 'pdf')
        print('{} plot saved'.format(filename))
        plt.close()

'''def plot_FFofq2_phys(curr_tuple_list, Mphys_dicts, colors, directory):
    labels = ['f_0', 'f_+', 'f_+x', 'f_T']
    colors = ['blue', 'red', 'purple', 'green']
    plt.figure(figsize=(10,6))
    plt.tight_layout()
    ci = 0
    for curr_tuple in curr_tuple_list:
        if ci != 2:
            Fphys_means, FFphys_errs = gvar_splitter(Mphys_dicts[0]['FFcont {}-curr'.format(curr_tuple[1])])
            q2_linspace = Mphys_dicts[1]
            upperlim, lowerlim = np.array(Fphys_means) + np.array(FFphys_errs), np.array(Fphys_means) - np.array(FFphys_errs)
            plt.plot(q2_linspace, Fphys_means, color = colors[ci], label = labels[ci])
            plt.fill_between(q2_linspace, upperlim, lowerlim, alpha = 0.3, color = colors[ci])
        ci += 1
    plt.xlabel('q² [GeV]')
    plt.minorticks_on()
    ax = plt.gca()
    plt.tick_params(axis='y', which='both', labelright=True, right=True)
    plt.legend(frameon=False)
    plt.savefig('./{}/{}.pdf'.format(directory, Mphys_dicts[4]),format = 'pdf')
    print('{} plot saved'.format(Mphys_dicts[4]))'''

#new version to use with sept 23 fork
def plot_FFofq2_phys(curr_tuple_list, FF_set, q2_phys_range, tag, Title, labels, colors, directory):
    plt.figure(figsize=(8,6))
    plt.tight_layout(pad=0)
    ci = 0
    for curr_tuple in curr_tuple_list:
        if ci != 2:
            FFphys_means, FFphys_errs = gvar_splitter(FF_set['FFs_{}_{}'.format(tag, curr_tuple[1])])
            upperlim, lowerlim = np.array(FFphys_means) + np.array(FFphys_errs), np.array(FFphys_means) - np.array(FFphys_errs)
            plt.plot(q2_phys_range, FFphys_means, color = colors[ci], label = labels[ci], linewidth = 1)
            plt.fill_between(q2_phys_range, upperlim, lowerlim, alpha = 0.3, color = colors[ci])
        ci += 1
    plt.xlabel(r'$q^2$ [GeV$^2$]')
    ax = plt.gca()
    plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
    plt.tick_params(axis = 'x',  which= 'both', top = True, direction = 'in')
    plt.legend(frameon=False)
    plt.title(Title)
    plt.minorticks_on()
    plt.savefig('./{}/{}.pdf'.format(directory, tag),format = 'pdf')
    print('{} plot saved'.format(tag))

def plot_allcurrs_FFofz(call_str, curr_tuple_list, Meson_tuple, colors, labels, directory, phys_title):
    plt.figure(figsize=(8,6))
    plt.tight_layout(pad=0)
    FF_set, poleterm_dict, z_range, Title, phys_str = Meson_tuple[0], Meson_tuple[1], Meson_tuple[2], Meson_tuple[3], Meson_tuple[4]
    ci = 0
    for curr_tuple in curr_tuple_list:
        if ci != 2:
            polexFFs = []
            FFi = 0
            for FF in FF_set['FFs_{}_{}'.format(phys_str, curr_tuple[1])]:
                polexFF = FF*poleterm_dict['poleterms_{}_{}'.format(phys_str, curr_tuple[1])][FFi]
                polexFFs.append(polexFF)
                FFi += 1
            polexFFs_means, polexFFs_errs = gvar_splitter(polexFFs)
            z_means, z_errs = gvar_splitter(z_range)
            #print(z_means, polexFFs_means)
            plt.plot(z_means, polexFFs_means, color = colors[ci], label = labels[ci])
            plt.fill_between(z_means, np.array(polexFFs_means) + np.array(polexFFs_errs), np.array(polexFFs_means) - np.array(polexFFs_errs), alpha=0.3, color = colors[ci])
        ci +=1
    plt.xlabel(r'$z(q^2, t_0)$')
    #manually doing titles to avoid $$ inside of $$
    if phys_str == 'Bpi': plt.ylabel(r'$B(q^2)f^{B \rightarrow \pi}(q^2)$')
    elif phys_str == 'Dpi': plt.ylabel(r'$B(q^2)f^{D \rightarrow \pi}(q^2)$')
    elif phys_str == 'BsK': plt.ylabel(r'$B(q^2)f^{B_s \rightarrow K}(q^2)$')
    elif phys_str == 'DsK': plt.ylabel(r'$B(q^2)f^{D_s \rightarrow K}(q^2)$')
    else: plt.ylabel(r'$B(q^2)f^{{{}}}(q^2)$'.format(phys_str))
    plt.legend(frameon=False)
    #plt.ylim([-0.1, 1.0])
    ax = plt.gca()
    plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
    plt.tick_params(axis = 'x',  which= 'both', top = True, direction = 'in')
    plt.minorticks_on()
    filename = '{}_FFofz'.format(phys_str)
    plt.savefig('./{}/{}.pdf'.format(directory, filename),format = 'pdf')
    plt.close()
    print('f(z) plot saved to ./{}/{}.pdf'.format(directory, filename))

def plot_FFofz(call_str, ens_list, curr_tuple_list, polexFFofz_lattice_disc_dict, z_lattice_disc_dict, polexFFofz_lattice_cont_dict, z_lattice_cont_dict, non_gv_dict, directory, colors, markers, plot_phys = False):
    colors = ['blue', 'red', 'green', 'purple']
    for curr_tuple in curr_tuple_list:
        curr0, curr1 = curr_tuple[0], curr_tuple[1]
        plt.figure(figsize=(14,10))
        plt.tight_layout()
        ens_i = 0
        for ens in ens_list:
            mi = 0
            for mass in non_gv_dict['{}_masses'.format(ens)]:
                #discrete part
                x_means, x_errs = gvar_splitter(z_lattice_disc_dict['z_{}_{}_m{}'.format(ens, curr1, mass)])
                y_means, y_errs = gvar_splitter(polexFFofz_lattice_disc_dict['pole*FF_{0}_{1}_m{2}'.format(ens, curr1, mass)])
                plt.errorbar(x_means,y_means, xerr=x_errs,yerr=y_errs, capsize = 2, linestyle = '', color = colors[mi])
                plt.scatter(x_means,y_means, label = '{} am={}'.format(ens, mass), s = 70, facecolors='none', marker=markers[ens_i], color = colors[mi])
                xs = z_lattice_cont_dict['z_{}_{}_m{}'.format(ens, curr1, mass)]
                ys = polexFFofz_lattice_cont_dict['pole*FF_{}_{}_m{}'.format(ens, curr1, mass)]
                plt.plot(xs, ys, linestyle = ':', color = colors[mi])
                mi += 1
            ens_i += 1
        #implementing physical D and B plotting
        if plot_phys != False:
            mcolors = ['purple', 'blue']
            m = 0
            for Meson_tuple in plot_phys:
                FF_set, poleterm_dict, z_range, Title, phys_str = Meson_tuple[0], Meson_tuple[1], Meson_tuple[2], Meson_tuple[3], Meson_tuple[4]
                polexFFs = []
                FFi = 0
                for FF in FF_set['FFs_{}_{}'.format(phys_str, curr_tuple[1])]:
                    polexFF = FF*poleterm_dict['poleterms_{}_{}'.format(phys_str, curr_tuple[1])][FFi]
                    polexFFs.append(polexFF)
                    FFi += 1
                polexFFs_means, polexFFs_errs = gvar_splitter(polexFFs)
                z_means, z_errs = gvar_splitter(z_range)
                #print(z_means, polexFFs_means)
                plt.plot(z_means, polexFFs_means, color = mcolors[m], label = Title)
                plt.fill_between(z_means, np.array(polexFFs_means) + np.array(polexFFs_errs), np.array(polexFFs_means) - np.array(polexFFs_errs), alpha=0.3, color = mcolors[m])
                m +=1
        plt.xlabel('z')
        if curr0 == '0': plt.ylabel(r'$\left(1-\frac{{q^2}}{{M^2_{{H^*_0}}}}\right)f_{}(z)$'.format(curr0))
        else: plt.ylabel(r'$\left(1-\frac{{q^2}}{{M^2_{{H^*}}}}\right)f_{}(z)$'.format(curr0))
        plt.legend(frameon=False)
        ax = plt.gca()
        plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
        plt.tick_params(axis = 'x',  which= 'both', top = True, direction = 'in')
        plt.minorticks_on()
        plt.ylim([-0.1, 1.0])
        filename = '{}_{}_FFofz'.format(call_str, curr1)
        plt.savefig('./{}/{}.pdf'.format(directory, filename),format = 'pdf')
        print('f(z) plot saved to ./{}/{}.pdf'.format(directory, filename))

def plot_FofM_H(call_str,  FofM_H_dict, M_space, directory):
    plt.figure(figsize=(8,6))
    plt.tight_layout(pad=0)
    F0_0, F0_max, Fplus_max, FT_0, FT_max = [], [], [], [], []
    colors = ['black', 'blue', 'red', 'green', 'purple']
    labels = [r'$f_{0/+}(0)$', r'$f_0(q^2_\mathrm{max})$', r'$f_+(q^2_\mathrm{max})$', r'$f_{T}(0, \mu=4.8)$', r'$f_T(q^2_\mathrm{max}, \mu=4.8)$']
    for mi in range(len(M_space)):
        F0_0.append(FofM_H_dict['FFs_{}_S_mi={}'.format(call_str, mi)][0])
        F0_max.append(FofM_H_dict['FFs_{}_S_mi={}'.format(call_str, mi)][1])
        #Fplus_0.append(FofM_H_dict['FFs_Hpi_X_mi={}'.format(mi)][0])
        Fplus_max.append(FofM_H_dict['FFs_{}_X_mi={}'.format(call_str, mi)][1])
        FT_0.append(FofM_H_dict['FFs_{}_T_mi={}'.format(call_str, mi)][0])
        FT_max.append(FofM_H_dict['FFs_{}_T_mi={}'.format(call_str, mi)][1])
    i = 0
    for arr in [F0_0, F0_max, Fplus_max, FT_0, FT_max]:
        means, errs = gvar_splitter(np.array(arr))
        plt.plot(M_space, means, color = colors[i], linewidth=0.5)
        plt.fill_between(M_space, np.array(means)+np.array(errs), np.array(means)-np.array(errs), label = labels[i], color = colors[i], alpha=0.3)
        i += 1 
    plt.axvline(x = M_space[0], color = 'grey', linestyle = ':')
    plt.axvline(x = M_space[-1], color = 'grey', linestyle = ':')
    plt.legend(frameon=False)
    ax = plt.gca()
    plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
    plt.tick_params(axis = 'x',  which= 'both', top = True, direction = 'in')
    plt.minorticks_on()
    plt.xlabel(r'$M_H[\mathrm{GeV}]$')
    if call_str == 'Hpi':
        plt.text(M_space[0], -0.6, r'$M_D$', horizontalalignment='center', verticalalignment='center')
        plt.text(M_space[-1], -0.6, r'$M_B$', horizontalalignment='center', verticalalignment='center')
    elif call_str == 'HsK':
        plt.text(M_space[0], -0.2, r'$M_{D_s}$', horizontalalignment='center', verticalalignment='center')
        plt.text(M_space[-1], -0.2, r'$M_{B_s}$', horizontalalignment='center', verticalalignment='center')
    plt.savefig('./{}/{}.pdf'.format(directory, '{}-FofM_H.pdf'.format(call_str)),format = 'pdf')
    print('Saved FofM_H plot to ./{}/{}.pdf'.format(directory, '{}-FofM_H.pdf'.format(call_str)))
