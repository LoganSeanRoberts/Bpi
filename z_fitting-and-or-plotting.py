import collections
import gvar as gv
import numpy as np
import lsqfit
import time
import matplotlib.pyplot as plt
from z_expansion.z_functions import *
from z_expansion.z_ensemble_dicts import *
from tabulate import tabulate
#import pickle
#import datetime
plt.rc("font",**{"size":14})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# This scipt has two main functions that can be done seperately or together dpending on the following inputs
# The functions are to 1) do the z expansion fit
#                  and 2) make z-expansion plots
# Defining the fit funciton is necessary for either functions
do_fit = True
do_plots = True
add_keyword = 'Dec08chi2scaling' #Extra keyword you can add to file name of gv dump to help pick it out in directory
save_fit = False #If true, will save fit in directory given below
save_corr_matrix_dict = True  #save_corr_matrix_dict requires do_plots = True and save_fit = True, 

#If we just want to plot without having to rerun the fit, we need to include the filename of the gvdump we want to pull from.
#File should be a .pickle file in the following defined directory:
output_dir = 'z_expansion/fit_outputs'
z_fit_gvdump_file = 'Hpi_Npoly=3_Nikjl=(3, 3, 2, 2)_chi2-by-dof=0.374_Q=1.0_logGBF=600_Oct22.pickle'
#z_fit_gvdump_file = 'HsK_Npoly=3_Nikjl=(3, 3, 2, 2)_chi2-by-dof=1.382_Q=0.0_logGBF=706_Sep3-tensor-fix.pickle'
#z_fit_gvdump_file = ''

# Control Panel #
#call_str, alt_call_str, phys_strs,  s, Title = 'Hpi', 'H → π' , ('Dpi', 'Bpi'), '', (r'$D \rightarrow \pi$', r'$B \rightarrow \pi$')
call_str, alt_call_str, phys_strs, s, Title = 'HsK', 'Hs → K' , ('DsK', 'BsK'), 's', (r'$Ds \rightarrow K$', r'$B_s \rightarrow K$')
Npoly = 3 #This is something to change in stability testing
Nijkl = (3,3,2,2) #The change in stability testing
Lambda_QCD = 0.5
max_iter = 5000 #max iterations do cycle fit function
FLAG_dispersion = False #Use lattice dispersion relation to get non zero twist energies rather than fit posteriors...
                       # Note the fit will not go right if FLAG_dispersion does not match the same flag in FF_control.py
incld_errbnds = True #IF true, plots errorbands on f(q2)lattice plots rather than just mean (note, =true looks better! use it)
############################################################
#Twists from control.py in correlator fitting
twist_dict = {'F_twists' : ['0.0','0.4281','1.282','2.1410','2.570'],
              'Fp_twists' : ['0.0','0.58','0.87','1.13','3.000','5.311'],
              'SF_twists' : ['0.0','1.261','2.108','2.666','5.059'],
              'SFp_twists' : ['0.0','2.522','4.216','7.94','10.118'],
              'UF_twists' : ['0.0','0.706','1.529','2.235','4.705']}

#Initializing fit parameter / prior dictionary
gvdump_dict = {}
ens_list =['F', 'Fp', 'SF', 'SFp', 'UF'] ## Add to this for adding additional ensembles 
#making change to try and deal wiht both vector data sets
curr_tuple_list = [('0', 'S'), ('+', 'V'), ('+','X'), ('T','T')]
currs3 = ['0', '+', 'T'] #calling this currs three to remind ourselves that this list only has three entries
#Buildung parameter dictionary by reading in gv dump files, and adding FF, q2, energies and masses to p_dict
for ens in ens_list: #for ensemble in ensemble list
    g = load_gvdump('form_factors/FF_gvdumps', call_str, ens)
    gvdump_dict.update(g)
#Removing the FF data from p_dict and putting it in FF_data_dict...
#And removing perp and par for now...
FF_data_dict = {}
p_dict = {}
q2_dict = {} #This is used for plotting corrfitt derrived form factors on same plots as continuous z expansion curves
for key in gvdump_dict:
    if key.startswith('FFs_') == True:
        FF_data_dict[key] = gvdump_dict[key]
    elif key.startswith('M') == True:
        p_dict[key] = gvdump_dict[key]
    elif key.startswith('E') == True:
        p_dict[key] = np.array(gvdump_dict[key])
    elif key.startswith('q2s') == True:
        q2_dict[key] = gvdump_dict[key]

#Creating special gvload for Hpi only. Need meson masses to calculate t_cut in z-mapping, and to get pole mass in pole term
if call_str == 'HsK':
    Hpi_gvdump = {}
    Hpi_dict = {}
    for ens in ens_list: #for ensemble in ensemble list
        Hpi_g = load_gvdump('form_factors/FF_gvdumps', 'Hpi', ens)
        Hpi_gvdump.update(Hpi_g)
    #
    for key in Hpi_gvdump:
        if key.startswith('M') == True:
            Hpi_dict['Hpi-{}'.format(key)] = Hpi_gvdump[key]
        elif key.startswith('E') == True:
            Hpi_dict['Hpi-{}'.format(key)] = Hpi_gvdump[key][0] #the [0] here is to only grab the pion mass E_0
    p_dict.update(Hpi_dict)

latt_dict, non_gv_dict = import_Ensemble_Dictionaries()

ms_tuned = 0
#p_dict.update(params)
for ens in ens_list:
    ms_tuned += calc_mtuned_strange(non_gv_dict['{}_strange_val'.format(ens)], non_gv_dict['eta_s_phys'], non_gv_dict['{}_eta_s'.format(ens)]) / non_gv_dict['{}_a'.format(ens)]
    for curr in currs3:
        p_dict['ap2_{}_{}'.format(ens, curr)] = gv.gvar('0.0(1.0)')
        #p_dict['ap4_{}_{}'.format(ens, curr)] = gv.gvar('0.0(1.0)')
non_gv_dict['ms_phys_avg'] = ms_tuned / len(ens_list)
p_dict['ms-ml_phys_ratio'] = gv.gvar('27.18(10)')
non_gv_dict['ms-ml_phys_ratio'] = 27.18
p_dict.update(add_Physical_Masses_in_GeV())
p_dict.update(latt_dict)
#for curr_tuple in curr_tuple_list:
#    p_dict.update(assign_Full_Prior_Set(Npoly, curr_tuple[0], Nijkl))
for curr in currs3:
    p_dict.update(assign_Full_Prior_Set(Npoly, curr, Nijkl))

#Defining fit function...
#non gv dict has all non gvar lattice info
## a_set = None presumes that we are fitting lattice data with ensemble specific a_latt. 
### Otherwise phys tuple can be set as (customkey, M_H, a, M_Hs)
def fit_func(arg, p_dict, non_gv_dict = non_gv_dict, ens_list = ens_list, call_str = call_str, phys_tuple = None, troubleshoot = False, ap_disc = True, FofM_H = False):
    FF_dict = {} #This is the dicitonary that is returned by the function
    poleterm_dict = {} #Used for continuum limit plotting along z axis
    corr_matrix_dict = {} #dict of terms that we would want a correlation matrix for: a_ns, H*, H*0, chilog
    #Used for plotting FofM_H plot, only used if FofM_H = True. FofM_H is reset to false is phys tuple is none, because we still need a=0 and such
    if phys_tuple == None: FofM_H = False
    if FofM_H == True:
        M_Dphys, M_Bphys = 1.86966, 5.27941
        M_Dsphys, M_Bsphys = 1.96835, 5.36693
        M_space = np.linspace(M_Dphys, M_Bphys, 30)
        Ms_space = np.linspace(M_Dsphys, M_Bsphys, 30)
        #FS_0, FS_max, FX_0, FX_max, FT_0, FT_max = [], [], [], [], [], []
        #FofM_H_dict['FS_0']
    for curr_tuple in curr_tuple_list:
        curr = curr_tuple[0]
        if phys_tuple != None: #phys_tuple implementation
            physkey = phys_tuple[0]
            ens_list = [physkey]
            if FofM_H == True:
                ens_list = [call_str]
        for ens in ens_list:
            if phys_tuple != None: #phys_tuple implementation
                a = phys_tuple[2]
            else: a = non_gv_dict['{}_a'.format(ens)] #lattice length in GeV units
            #if phys_tuple != None:
            #    
            #else: M_D = p_dict['M-mother_{}_{}'.format(ens, curr_tuple[1])][0]/a #This is the mass of the D(s) meson for each ensemble
            M_D = p_dict['D{}_phys'.format(s)]
            #M_D = p_dict['D_phys']
            # The following statements help us deal with the fact that we have a scalar form factor at 0-twist but not for the other currents
            # Additionally, a few lines down we define the poll mass term as a tuple, where polemass[0] = M_H*_0, and polemass[1] = M_H*
            if curr == '0': 
                tw_range = 0
            else: 
                tw_range = 1
            mi = 0 #This ticker is how we cycle through mass option in the following loop
            if phys_tuple != None: 
                mass_list = ['0.0'] #phys_tuple implementation - only one mass option.  assume am_h = 1. only used in acoeff calc where we do a->0
                if FofM_H == True: 
                    if call_str == 'Hpi': mass_list = M_space
                    elif call_str == 'HsK': mass_list = Ms_space
            else: mass_list = non_gv_dict['{}_masses'.format(ens)]
            for mass in mass_list:
                if phys_tuple != None: #phys_tuple implementation
                    if call_str == 'Hpi': 
                        if FofM_H == True: heavy = mass
                        else: heavy = phys_tuple[1] #not actually a string here
                        M_mother, M_daughter, Hl_mass, pion_mass = heavy, p_dict['pi_phys'], heavy, p_dict['pi_phys']
                    elif call_str == 'HsK':
                        if FofM_H ==True: heavy, heavy_strange = M_space[mi], mass
                        else:  heavy, heavy_strange = phys_tuple[1], phys_tuple[3]
                        M_mother, M_daughter, Hl_mass, pion_mass = heavy_strange, p_dict['K_phys'], heavy, p_dict['pi_phys']
                else:
                    M_mother = p_dict['M-mother_{}_{}'.format(ens, curr_tuple[1])][mi]/a
                    M_daughter = p_dict['E-daughters_{}'.format(ens)][0]/a
                    ##Now to enforce that for BsK, Hpi is used for pole mass and z expansiont t_cut
                    if call_str == 'Hpi' : Hl_mass, pion_mass = M_mother, M_daughter
                    elif call_str == 'HsK' : Hl_mass, pion_mass = p_dict['Hpi-M-mother_{}_{}'.format(ens, curr_tuple[1])][mi]/a, p_dict['Hpi-E-daughters_{}'.format(ens)]/a
                    else: print('Invalid call_str, must be "Hpi" or "HsK"')
                
                if phys_tuple == None:
                    mtuned_strange = calc_mtuned_strange(non_gv_dict['{}_strange_val'.format(ens)], non_gv_dict['eta_s_phys'], non_gv_dict['{}_eta_s'.format(ens)])
                    alt_mtuned_strange = non_gv_dict['ms_phys_avg']*a
                    mtuned_light = calc_mtuned_light(non_gv_dict['{}_strange_val'.format(ens)], non_gv_dict['eta_s_phys'], non_gv_dict['{}_eta_s'.format(ens)], non_gv_dict['ms-ml_phys_ratio'])
                    #if 'trigger' in arg: print('mtuned_light: ', round(mtuned_light/a,5), 'mtuned_strange: ', round(mtuned_strange/a,5)) #troubleshooting mtuned values
                #getting polemass term
                Xphys_pi = 1/p_dict['ms-ml_phys_ratio'] * (2/5.63)
                if phys_tuple != None: X_pi = 1.00000001*Xphys_pi #phys_tuple implementation
                else: X_pi = calc_X_pi(non_gv_dict['{}_light'.format(ens)], alt_mtuned_strange)

                polemass = (Hl_mass + p_dict['Delta_M_H'], calc_M_Hstar(Hl_mass, p_dict['D_phys'], p_dict['D*_phys'], p_dict['B_phys'], p_dict['B*_phys']))
                
                #ChiPT logarithm term
                if phys_tuple != None: delta_FV = 0 #phys_tuple implementation
                else: delta_FV = p_dict['{}_delta_FV'.format(ens)]
                chi_log = calc_chi_log_L(call_str, X_pi, p_dict['g_inf'], M_mother, Lambda_QCD, p_dict['C1'], p_dict['C2'], delta_FV)            
                #now to calculate a coefficients for a given current, ensemble, and n in Npoly.  a's are not momentum dependent.
                
                #Sept25 fixed lines below: arguments were not ordered correctly
                #Ntuned related functions - for phys assume all physical quarks and Nterm goes to 0
                if phys_tuple == None:
                    deltaval_s = calc_delta_q(non_gv_dict['{}_strange_val'.format(ens)], mtuned_strange, mtuned_strange)
                    deltasea_s = calc_delta_q(non_gv_dict['{}_strange_sea'.format(ens)], mtuned_strange, mtuned_strange)
                    delta_l    = calc_delta_q(non_gv_dict['{}_light'.format(ens)],       mtuned_light,   mtuned_strange)
                    deltasea_c = calc_delta_charm_sea(non_gv_dict['{}_charm_sea'.format(ens)], non_gv_dict['{}_charm_tuned'.format(ens)])
                
                #With these prerequisites defined we can now calculate the a coefficients used in the z-expansion
                a_ns = []
                for n in range(Npoly):
                    curr_str = curr
                    if n == 0:
                        zeta = p_dict['zeta_0']
                        if curr != 'T':
                            curr_str = '0/+'
                    else: zeta = 0
                    am_h = mass
                    if phys_tuple != None: Ntuned, am_h = 0 , 0
                    else: Ntuned = calc_Mass_Mistune_Ncurr_n(p_dict['cval^{0}_s{1}'.format(curr_str,n)], p_dict['csea^{0}_s{1}'.format(curr_str,n)], p_dict['csea^{0}_l{1}'.format(curr_str,n)], p_dict['csea^{0}_c{1}'.format(curr_str,n)], deltaval_s, deltasea_s, delta_l, deltasea_c)
                    print_parts = False
                    '''if troubleshoot == True:
                        if ens == 'UF':
                            if mi == 3:
                                if n == 0:
                                    print_parts = True #a_ns trouble shooting
                                    print('Lattice a_0s breakdown for UF, am_h=0.8, {}-current:'.format(curr_tuple[1]))
                        if phys_tuple != None: 
                            if n == 0:
                                print_parts = True
                                print('Physical limit a_0s breakdown for {}-current:'.format(curr_tuple[1]))'''
                    a_n = calc_a_coeff(M_D, M_mother, zeta, p_dict['rho^{0}_{1}'.format(curr_str,n)],Ntuned, Nijkl[0], Nijkl[1], Nijkl[2], Nijkl[3], p_dict, Lambda_QCD, float(am_h), X_pi, Xphys_pi, n, a, curr_str, print_parts)
                    a_ns.append(a_n)
                
                #if phys_tuple != None: print(a_ns) #debugging line
                temp_FF_list = []
                #Debug print statement vvv
                #print('a coeffs for {}-ensemble, {}-current, and {}-mass'.format(ens, curr_tuple[1], mass), a_ns)
                #Below we implement the ability to use the fit function using distinct pion/kaon energies which are posterior outputs form the corr. fit,...
                #   or a linespace of daughter meson energies that are useful for plotting.  If you want the funcion to accept a continuous linespace...
                #   for each ensemble, the dictionary 'arg' (function argument) must have a key 'trigger' -> arg['trigger'] must exist.  Otherise the ...
                #   fitter assumes you are passing a dicitonary of distinct energies at distinct twists
                if phys_tuple == None:
                    if 'trigger' in arg:
                        for i in range(len(arg['aE_space_{}'.format(ens)])):
                            E_daughter = arg['aE_space_{}'.format(ens)][i]/a
                            q2 = calc_q_sqrd(M_mother, M_daughter, E_daughter)#, twist_dict['{}_twists'.format(ens)][i], non_gv_dict['{}_L'.format(ens)])
                            z = z_expansion(q2, Hl_mass, pion_mass)                 
                            #using polemass[tw_range] here is out of convenience, to get a value 0 or 1 based on being a scalar or non scalar FF. 
                            temp_FF = calc_FF_of_z(q2, z, polemass[tw_range], Npoly, a_ns, chi_log, curr_tuple[0])
                            temp_FF_list.append(temp_FF)
                        FF_dict['FFs_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = np.array(temp_FF_list)

                    else:
                        #twist_dict = arg
                        for i in range(len(arg['{0}_twists'.format(ens)]))[tw_range:]:
                            twist_str = arg['{0}_twists'.format(ens)][i]
                            if ap_disc == True:
                                ap_daughter = calc_Daughter_3Momentum(twist_str, non_gv_dict['{}_L'.format(ens)])
                                ap_term = (1 + p_dict['ap2_{}_{}'.format(ens, curr)]*(ap_daughter/np.pi)**2) # + p_dict['ap4_{}_{}'.format(ens, curr)]*(ap_daughter/np.pi)**4)
                            else: ap_term = 1
                            E_daughter = p_dict['E-daughters_{}'.format(ens)][i]/a
                            q2 = calc_q_sqrd(M_mother, M_daughter, E_daughter, FLAG_dispersion = FLAG_dispersion, twist=twist_str, N_x = non_gv_dict['{}_L'.format(ens)])#, twist_dict['{}_twists'.format(ens)][i], non_gv_dict['{}_L'.format(ens)])
                            z = z_expansion(q2, Hl_mass, pion_mass)                 
                            #using polemass[tw_range] here is out of convenience, to get a value 0 or 1 based on being a scalar or non scalar FF. 
                            temp_FF = calc_FF_of_z(q2, z, polemass[tw_range], Npoly, a_ns, chi_log, curr_tuple[0], ap_term)
                            temp_FF_list.append(temp_FF)
                        FF_dict['FFs_{}_{}_m{}'.format(ens, curr_tuple[1], mass)] = np.array(temp_FF_list)
                
                else: ##under this condition (where presumably we want continuos physical mesons), we ignore the "arg" argument in fit funtcion
                    corr_matrix_dict['{}-a^{}'.format(ens, curr)] = a_ns 
                    corr_matrix_dict['{}-chi_log'.format(ens)] = chi_log
                    if curr == '0': corr_matrix_dict['{}-H*0'.format(ens)] = polemass[0]
                    else: corr_matrix_dict['{}-H*'.format(ens)] = polemass[1]
                    q2max = ((M_mother - M_daughter)**2).mean
                    q2min = 0.00000001
                    if FofM_H == True: q2_phys_range = np.array([q2min, q2max]) #just use q2 = 0 and q2 max for FofM_H plotting
                    else: q2_phys_range = np.linspace(q2min, q2max, 1000)
                    poleterm_range = 1-(q2_phys_range/polemass[tw_range]**2)
                    z_phys_range = z_expansion(q2_phys_range, Hl_mass, pion_mass)
                    for i in range(len(z_phys_range)):
                        temp_FF = calc_FF_of_z(q2_phys_range[i], z_phys_range[i], polemass[tw_range], Npoly, a_ns, chi_log, curr_tuple[0])
                        temp_FF_list.append(temp_FF)
                    if FofM_H == True: FF_dict['FFs_{}_{}_mi={}'.format(ens, curr_tuple[1], mi)] = np.array(temp_FF_list)
                    else:
                        poleterm_dict['poleterms_{}_{}'.format(ens, curr_tuple[1])] = poleterm_range
                        #implementing 1.0773(17) scaling for mu (2GeV) for f_T
                        mu_scalar = 1
                        if 'D' in phys_tuple[0]:
                            if curr == 'T':
                                mu_scalar = gv.gvar('1.0773(17)')
                        FF_dict['FFs_{}_{}'.format(ens, curr_tuple[1])] = np.array(temp_FF_list)*mu_scalar
                mi += 1

    #Comment out following line if you don't want your terminal output filled up.  But its nice to see that the fitter is working rather than being stuck
    print('Successfully run iteration of fit func at time =', time.strftime("%H:%M:%S", time.localtime()))
    if phys_tuple != None:
        if FofM_H ==True: 
            if call_str == 'Hpi': return FF_dict, M_space
            elif call_str == 'HsK': return FF_dict, Ms_space
        else:
            #corr_matrix_dict = dict(sorted(corr_matrix_dict.items()))
            for key in corr_matrix_dict:
                print('{}: {}'.format(key, corr_matrix_dict[key]))
            if save_corr_matrix_dict == True:
                filename = 'corr_matrix_dict_{}_{}'.format(ens, add_keyword)
                gv.dump(corr_matrix_dict,'{}/{}.pickle'.format(output_dir, filename))
                f = open('{}/{}.txt'.format(output_dir, filename),'w')
                for key in corr_matrix_dict:
                    f.write('{}: {}\n'.format(key, corr_matrix_dict[key]))
                f.write('\n' * 3)
                f.write(format_corr_mtrx(corr_matrix_dict))
                f.close()
            return FF_dict, q2_phys_range, poleterm_dict, z_phys_range
    else: return FF_dict
    
#Implementing the use of the fit function
if do_fit == True:
    print('Beginning fit...')
    fit = lsqfit.nonlinear_fit(data = (twist_dict, FF_data_dict), fcn=fit_func, prior = p_dict, maxit = max_iter)
    print(fit)
    #Dtable_print, Btable_print = print_results(fit, Npoly, Nijkl[0], currs3, call_str, Lambda_QCD)
    ### Saving z fit outputs for plotting purposes
    if save_fit == True:
        filename = '{}_Npoly={}_Nikjl={}_chi2-by-dof={}_Q={}_logGBF={}_{}'.format(call_str, Npoly, Nijkl, round(fit.chi2/fit.dof, 3), round(fit.Q,3), round(fit.logGBF), add_keyword)
        print('Starting gv dump...')
        #Dont need to create pickle file until fit finalized, then uncomment following line
        gv.gdump(fit.p,'{}/{}.pickle'.format(output_dir, filename))
        f = open('{}/{}.txt'.format(output_dir, filename),'w')
        f.write(fit.format(pstyle='v'))
        f.write('\nchi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}\n'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        #f.write('{}\n{} Composite Posteriors\n{}'.format('-'*28, phys_strs[0], tabulate(Dtable_print, headers=['Key', 'Value'])))
        #f.write('\n{}\n{} Composite Posteriors\n{}'.format('-'*28, phys_strs[1], tabulate(Btable_print, headers=['Key', 'Value'])))
        f.close()
        print('Completed gv dump to {}/{}.pickle'.format(output_dir, filename))

#Doing plotting.  Want to add functionality to plot the ouput of do_fit if do_fit == True
if do_plots == True:
    #If do_fit == True, the we want to plot directly from that, we can define post_dict = fit.p. Else we gv.load from the dir and file specified at top of script
    #We use the dicitonary post_dict to not confuse ourself with p_dict; we are using parameters from the psoterior fit results of the z-fit
    if do_fit == True: post_dict = fit.p
    else: post_dict = gv.load('{}/{}'.format(output_dir, z_fit_gvdump_file))
    
    # We ca plot both a the linspace and distinct points on the same plots
    #Creating aE arrays - continuous linspace spanning q2 range that matches lattice data
    #IMPORTANT -> if you want the q2 range to be the energy limits from correlator fit, use p_dict as 0th argument below...
    #          -> if you want the q2 range to be the energy limits from the posteriors of the z-fit, use post dict
    #               -> This has a minor effect and only matters to plotting. only really visible at high twist/low q2 for dispersion relation violating meson energies
    aE_dict = get_lattice_aE_dict(post_dict, ens_list, plot_scale_expander=(0.0,0.36))
    #creating our y-variable linespace from aE dict above.  the output of the fit func is a dictionary containing n keys where n = number of ensembles.
    Fit_dict = fit_func(aE_dict, post_dict, troubleshoot = False)   #
    #Fit_FF_dict = fit_func(twist_dict, post_dict) # Distinc points os f(q2) where q2 values come from twistst

    #implementing f(z) plot functionality where all mass/ensmeble variations are shown on same plot for either scalar, vector, tensor
    #ideally the keys in both the following dicitonaries shoud be for xvar z or yvar pole*FF -> dict['z/pole*FF_{}_{}_m{}'.format(ens, mass)] = [...] -> gvars in descrete case, list of floats in cont. case

    #plotting params
    #colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf'] #ideally these are color blind friendly?
    colors = ['blue', 'red', 'green', 'purple']
    
    markers = ['^', '>', 'v', '<', 'D']

    dict0, dict1, dict2, dict3, dict4 = get_lattice_FFofz_dicts(call_str, ens_list, curr_tuple_list, twist_dict, q2_dict, aE_dict, Fit_dict, FF_data_dict, post_dict, non_gv_dict)
    
    directory = 'z_expansion/FFofq_plots'
    
    plot_FFofq2_lattice(call_str, ens_list, curr_tuple_list, q2_dict, FF_data_dict, dict4, Fit_dict, non_gv_dict, directory, colors, markers, plot_phys = False, incld_errbnds = incld_errbnds)
    z_directory = 'z_expansion/FFofz_plots'

    #continuum implemntation
    phys_directory = 'z_expansion/Phys_Cont_Plots'
   
    colors = ['blue', 'red', 'red', 'green']

    #Testing M_H dependence
    Bscalar = 1.0
    Dscalar = 1.0
      
    #B or Bs phys
    labels = [r'$f_0(q^2)$', r'$f_{+}(q^2)$', r'$f_{+}(q^2)$', r'$f_{T}(q^2, \mu = 4.8 \mathrm{GeV})$']
    phys_tuple = (phys_strs[1], 5.27941*Bscalar, 0.1715/(10000*0.1973), 5.36693*Bscalar)
    B_FF_set, B_q2_phys_range, B_poleterm_dict, B_z_phys_range = fit_func(0, post_dict, non_gv_dict = non_gv_dict, ens_list = ens_list, call_str = call_str, phys_tuple = phys_tuple, troubleshoot = False)
    plot_FFofq2_phys(curr_tuple_list, B_FF_set, B_q2_phys_range, phys_tuple[0], Title[1], labels, colors, phys_directory)

    #D or Ds phys
    labels = [r'$f_0(q^2)$', r'$f_{+}(q^2)$', r'$f_{+}(q^2)$', r'$f_{T}(q^2, \mu = 2.0 \mathrm{GeV})$']
    phys_tuple = (phys_strs[0], 1.86966*Dscalar, 0.1715/(10000*0.1973), 1.96835*Dscalar)
    D_FF_set, D_q2_phys_range, D_poleterm_dict, D_z_phys_range = fit_func(0, post_dict, non_gv_dict = non_gv_dict, ens_list = ens_list, call_str = call_str, phys_tuple = phys_tuple, troubleshoot = False)
    plot_FFofq2_phys(curr_tuple_list, D_FF_set, D_q2_phys_range, phys_tuple[0], Title[0], labels, colors, phys_directory)

    z_directory = 'z_expansion/FFofz_plots'
    #plot_FFofz(call_str, ens_list, curr_tuple_list, dict0, dict1, dict2, dict3, non_gv_dict, z_directory, colors, markers)
    
    #B end tuple
    B_tuple = (B_FF_set, B_poleterm_dict, B_z_phys_range, Title[1], phys_strs[1])
    #D end tuple
    D_tuple = (D_FF_set, D_poleterm_dict, D_z_phys_range, Title[0], phys_strs[0])

    plot_FFofz(call_str, ens_list, curr_tuple_list, dict0, dict1, dict2, dict3, non_gv_dict, z_directory, colors, markers, plot_phys = (B_tuple, D_tuple))
    
    ## phys combination where 0,+,T all on same z plot
    plot_allcurrs_FFofz(call_str, curr_tuple_list, B_tuple, colors, labels, z_directory, 'B{}cont'.format(s))
    plot_allcurrs_FFofz(call_str, curr_tuple_list, D_tuple, colors, labels, z_directory, 'D{}cont'.format(s))
    
    phys_tuple = (phys_strs[1], 5.27941*Bscalar, 0.1715/(10000*0.1973), 5.36693*Bscalar)
    FofM_H_dict, M_space = fit_func(0, post_dict, non_gv_dict = non_gv_dict, ens_list = ens_list, call_str = call_str, phys_tuple = phys_tuple, troubleshoot = False, FofM_H = True)
    #print(FofM_H_dict)

    plot_FofM_H(call_str,  FofM_H_dict, M_space, phys_directory)




    