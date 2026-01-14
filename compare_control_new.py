### Changed as of 13/02/2024 to use new pickle fit method
### Added functionality to get priors for 3 point amps

import numpy as np
import gvar as gv
import math
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from comparing.compare_functions import *
#from comparing.compare_fit_picker import *
#from fit_choice_catalog_by_mass import *
#from fit_choice_catalog_by_current_type import *
import collections
import datetime

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

### Ensemble Specs
F = collections.OrderedDict()
F['tag'] = 'F'
F['alt_tag'] = 'f5'
F['avg'] = './comparing/datafile_avg_folder/f5_avg.txt'
F['masses'] = ['0.450','0.55','0.675','0.8']
F['twists'] = ['0.0','0.4281','1.282','2.1410','2.570']
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['Ts'] = [15,18,21,24]
F['tp'] = 96
F['L'] = 32
F['a'] = 0.1715/(1.9006*0.1973)
F['tmin_meson'] = 4
F['tmax_meson'] = 48
F['tmin_3pt'] = 2

Fp = collections.OrderedDict()
Fp['tag'] = 'Fp'
Fp['avg'] = './comparing/datafile_avg_folder/fp_avg.txt'
Fp['masses'] = ['0.433','0.555','0.678','0.8']
Fp['twists'] = ['0.0','0.58','0.87','1.13','3.000','5.311']
Fp['m_l'] = '0.0012'
Fp['m_s'] = '0.036'
Fp['Ts'] = [15,18,21,24]
Fp['tp'] = 96
Fp['L'] = 64
Fp['a'] = 0.1715/(1.9518*0.1973)
Fp['tmin_meson'] = 4
Fp['tmax_meson'] = 48
Fp['tmin_3pt'] = 2

SF = collections.OrderedDict()
SF['tag'] = 'SF'
SF['avg'] = './comparing/datafile_avg_folder/sf5_avg.txt'
SF['masses'] = ['0.274','0.5','0.65','0.8']
SF['twists'] = ['0.0','1.261','2.108','2.666','5.059']
SF['m_l'] = '0.0048'
SF['m_s'] = '0.0234'
SF['Ts'] = [22,25,28,31]
SF['tp'] = 144
SF['L'] = 48
SF['a'] = 0.1715/(2.896*0.1973)
SF['tmin_meson'] = 7
SF['tmax_meson'] = 72
SF['tmin_3pt'] = 3

SFp = collections.OrderedDict()
SFp['tag'] = 'SFp'
SFp['avg'] = './comparing/datafile_avg_folder/sfp_avg.txt'
SFp['masses'] = ['0.2585','0.5','0.65','0.8']
SFp['twists'] = ['0.0','2.522','4.216','7.94','10.118']
SFp['m_l'] = '0.0008'
SFp['m_s'] = '0.0219'
SFp['Ts'] = [22,25,28,31]
SFp['tp'] = 192
SFp['L'] = 96
SFp['a'] = 0.1715/(3.0170*0.1973) #from 2207.04765
SFp['tmin_meson'] = 7
SFp['tmax_meson'] = 72
SFp['tmin_3pt'] = 2


UF = collections.OrderedDict()
UF['tag'] = 'UF'
UF['alt_tag'] = 'uf5'
UF['avg'] = './comparing/datafile_avg_folder/uf5_avg.txt'
UF['masses'] = ['0.194','0.4','0.6','0.8']
UF['twists'] = ['0.0','0.706','1.529','2.235','4.705']
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['Ts'] = [29,34,39,44]
UF['tp'] = 192
UF['L'] = 64
UF['a'] = 0.1715/(3.892*0.1973)
UF['tmin_meson'] = 7
UF['tmin_3pt'] = 2
UF['tmax_meson'] = 96


#####################################################################
###################### Control Panel ################################
#####################################################################
                                                                    #
ensemble = F #Fp, SF, SFp, UF                                       #
strange = False  #if False -> B to pi, if True -> Bs to K             #
mass_choice = 2 #[0,1,2,3]                                          #    
twist_choice = 0 #[0,1,2,3,4]              
#ci = 0 #current choice -> Scalar, TVec, XVec, Tensor                         #
width_choice = 3 #[0,1,2,3]                                         #
Nexp = 4          
function_output_choice = 2  #0 = compare, 1 = twopt meffs, 2 = 3pt prior, 3 = 2pt amps , 4 = aEeff plots
                                                                    #
#####################################################################
#####################################################################
#####################################################################

### Automatic
corr_txt_file = ensemble['avg']
mass = ensemble['masses'][mass_choice]
twist = ensemble['twists'][twist_choice]
width  = ensemble['Ts'][width_choice]
tmin_meson = ensemble['tmin_meson']
tmin_3pt = ensemble['tmin_3pt']
ens = ensemble

currs = ['Scalar', 'Temporal Vector', 'Spacial Vector', 'Tensor']

Nt = ensemble['tp']

t_space = np.arange(0, Nt, 1)

if strange == True:
    spectator, daughter = 's', 'kaon'
    s_tag, s3_tag = 'true' , 'HsK'
else: spectator, daughter, s_tag, s3_tag = 'l', 'pion', 'false', 'Hpi'

#loaded_PKL_file = gv.load(fit_Pick(ensemble['tag'],mass))
#g_real = gv.load(fit_Pick_by_Current_Type(ensemble['tag'], 'real'))
#g_imaginary = gv.load(fit_Pick_by_Current_Type(ensemble['tag'], 'imaginary'))

'''def compare_plots():
    # Reconstructing 2pt correlator, masses, and amps
    posteriors_2pt_mother, posteriors_2pt_daughter = get_2pt_posteriors(g_real, mass, twist, spectator, daughter, Nexp)

    # corrs
    recon_corr_mother = recon_2pt_corr(posteriors_2pt_mother, Nexp, Nt)
    recon_corr_daughter = recon_2pt_corr(posteriors_2pt_daughter, Nexp, Nt)
    # masses
    aM_eff_mother_recon = calc_aM_eff(recon_corr_mother)
    aM_eff_daughter_recon = calc_aM_eff(recon_corr_daughter)
    # amplitudes
    Amp_eff_mother_recon = calc_Amp_eff(recon_corr_mother, aM_eff_mother_recon, Nt)
    Amp_eff_daughter_recon = calc_Amp_eff(recon_corr_daughter, aM_eff_daughter_recon, Nt)
    # means for plotting
    recon_corr_mother_means, recon_corr_mother_errs = gvar_splitter(recon_corr_mother)
    recon_corr_daughter_means, recon_corr_daughter_errs = gvar_splitter(recon_corr_daughter)
    Amp_eff_mother_recon_means, Amp_eff_mother_recon_errs = gvar_splitter(Amp_eff_mother_recon)
    Amp_eff_daughter_recon_means, Amp_eff_daughter_recon_errs = gvar_splitter(Amp_eff_daughter_recon)

    # Loading in gpl to read correlator and calc masses and amps
    tag_array, corr_array = load_Corr_txt(corr_txt_file)
    tag_mother, corr_mother, tag_daughter, corr_daughter = Corr2pt_picker(strange, mass, twist, tag_array, corr_array)

    # amplitudes and masses
    aM_eff_mother, aM_eff_daughter = calc_aM_eff(corr_mother), calc_aM_eff(corr_daughter)
    Amp_eff_mother, Amp_eff_daughter = calc_Amp_eff(corr_mother, aM_eff_mother, Nt), calc_Amp_eff(corr_daughter, aM_eff_daughter, Nt)

    # Means for plotting
    corr_mother_means, corr_mother_errs = gvar_splitter(corr_mother)
    corr_daughter_means, corr_daughter_errs = gvar_splitter(corr_daughter)

    aM_eff_mother_means, aM_eff_mother_errs = gvar_splitter(aM_eff_mother)
    aM_eff_daughter_means, aM_eff_daughter_errs = gvar_splitter(aM_eff_daughter)

    Amp_eff_mother_means, Amp_eff_mother_errs = gvar_splitter(Amp_eff_mother)
    Amp_eff_daughter_means, Amp_eff_daughter_errs = gvar_splitter(Amp_eff_daughter)
    


    ################################################## PLotting 2pt Effective Amps

    plt.figure(figsize=(10, 3))
    plt.subplot(1,2,1)
    plt.errorbar(t_space, Amp_eff_mother_means, yerr= Amp_eff_mother_errs, label = 'Corr Amp_eff', capsize=2, linestyle = '')
    plt.errorbar(t_space, Amp_eff_mother_recon_means, yerr= Amp_eff_mother_recon_errs, label = 'Recon Amp_eff', capsize = 2, linestyle = '')
    plt.title('Amp_eff {} {}'.format(ensemble['tag'] ,tag_mother))
    plt.axvline(x=tmin_meson, color = 'black', linestyle = '--', label = 'tmin')
    plt.ylim([0,0.50])
    #plt.xlim([0,Nt/2])
    plt.legend()

    plt.subplot(1,2,2)
    plt.errorbar(t_space, Amp_eff_daughter_means, yerr= Amp_eff_daughter_errs, label = 'Corr Amp_eff', capsize=2, linestyle = '')
    plt.errorbar(t_space, Amp_eff_daughter_recon_means, yerr= Amp_eff_daughter_recon_errs, label = 'Recon Amp_eff', capsize = 2, linestyle = '')
    plt.title('Amp_eff {} {}'.format(ensemble['tag'] ,tag_daughter))
    plt.axvline(x=tmin_meson, color = 'black', linestyle = '--', label = 'tmin')
    plt.ylim([-0.5,2])
    #plt.xlim([0,Nt/2])
    plt.legend()


    plt.savefig('./comparing/compare_plots/amp_eff_2pt_plot.png')
    print('Plot successfully saved to /comparing/compare_plots/amp_eff_2pt_plot.png')

#################################################### Plotting 2pt Correlators
    mother_ratio_means, mother_ratio_errs= gvar_splitter(recon_corr_mother/ corr_mother)
    daughter_ratio_means, daughter_ratio_errs= gvar_splitter(recon_corr_daughter/ corr_daughter)

    log_recon_corr_mother_means, log_recon_corr_mother_errs = gvar_splitter(gv.log(recon_corr_mother))
    log_recon_corr_daughter_means, log_recon_corr_daughter_errs = gvar_splitter(gv.log(recon_corr_daughter))

    log_corr_mother_means, log_corr_mother_errs = gvar_splitter(gv.log(corr_mother))
    log_corr_daughter_means, log_corr_daughter_errs = gvar_splitter(gv.log(corr_daughter))

    plt.figure(figsize=(10, 8))
    plt.subplot(2,2,1)
    plt.errorbar(t_space, log_corr_mother_means, yerr=log_corr_mother_errs, label = 'Original Corr', capsize=2, linestyle = '')
    plt.errorbar(t_space, log_recon_corr_mother_means, log_recon_corr_mother_errs, label = 'Recon Corr', capsize=2, linestyle = '')
    plt.title('ln {} {}'.format(ensemble['tag'] ,tag_mother))
    plt.axvline(x=tmin_meson, color = 'black', linestyle = '--', label = 'tmin')
    plt.legend()
    plt.xlim([0,Nt/2])
    plt.ylim([-60,0])

    plt.subplot(2,2,2)
    plt.errorbar(t_space, log_corr_daughter_means, yerr=log_corr_daughter_errs, label = 'Original Corr', capsize=2, linestyle = '')
    plt.errorbar(t_space, log_recon_corr_daughter_means, log_recon_corr_daughter_errs, label = 'Recon Corr', capsize=2, linestyle = '')
    plt.title('ln {} {}'.format(ensemble['tag'] ,tag_daughter))
    plt.axvline(x=tmin_meson, color = 'black', linestyle = '--', label = 'tmin')
    plt.legend()
    plt.xlim([0,Nt/2])
    plt.ylim([-30,0])

    plt.subplot(2,2,3)
    plt.errorbar(t_space, mother_ratio_means, yerr=mother_ratio_errs, color = 'red', capsize = 1.5, fmt = 'none')
    plt.axvline(x=tmin_meson, color = 'black', linestyle = '--')
    plt.axvline(x= Nt - tmin_meson, color = 'black', linestyle = '--')
    plt.axhline(y = 1, color = 'blue', linestyle = '--')
    plt.title('Recon/Original Corr {} {}'.format(ensemble['tag'] ,tag_mother))
    plt.ylim([0.8, 1.2])


    plt.subplot(2,2,4)
    plt.errorbar(t_space, daughter_ratio_means, yerr=daughter_ratio_errs, color = 'red', capsize = 1.5, fmt = 'none')
    plt.axvline(x=tmin_meson, color = 'black', linestyle = '--')
    plt.axvline(x= Nt - tmin_meson, color = 'black', linestyle = '--')
    plt.axhline(y = 1, color = 'blue', linestyle = '--')
    plt.title('Recon/Original Corr {} {}'.format(ensemble['tag'] ,tag_daughter))
    plt.ylim([0.8, 1.2])


    plt.savefig('./comparing/compare_plots/corr_2pt_plot.png')
    print('Plot successfully saved to /comparing/compare_plots/corr_2pt_plot.png')


###################################################
# 3 point processing and plotting
###################################################
    # Corr Data from gpl
    tags_3pt, corrs_3pt = Corr3pt_picker(strange, mass, twist, width, tag_array, corr_array)
    corrs_3pt_means, corrs_3pt_errs = [], []
    for corr in corrs_3pt:
        #corr = gv.log(zero_setter(corr))
        corr = gv.log(corr)
        corr_mean,  corr_err = gvar_splitter(corr)
        corrs_3pt_means.append(corr_mean)
        corrs_3pt_errs.append(corr_err)

    #getting posteriors
    posteriors_3pt, order_string = get_3pt_posteriors(g_real, g_imaginary, strange, mass, twist)
    
    reconned_3pt_corrs = recon_3pt_corr(posteriors_3pt, posteriors_2pt_mother, posteriors_2pt_daughter, width, Nexp, Nt)
    reconned_3pt_means, reconned_3pt_errs = [], []
    for comp in reconned_3pt_corrs:
        #comp =gv.log(zero_setter(comp))
        comp =gv.log(comp)
        comp_mean, comp_err = gvar_splitter(comp)
        reconned_3pt_means.append(comp_mean)
        reconned_3pt_errs.append(comp_err)

##############
#####plotting#
##############
    plt.figure(figsize=(10, 8))

    plt.subplot(2,2,1)
    plt.errorbar(t_space, corrs_3pt_means[0], yerr= corrs_3pt_errs[0], label = 'Original Corr',capsize = 4, linestyle = '')
    plt.errorbar(t_space, reconned_3pt_means[0], yerr= reconned_3pt_errs[0], label = 'Recon Corr', capsize = 4, linestyle = '')
    plt.title('ln {} {}'.format(ensemble['tag'], tags_3pt[0]))
    plt.xlim([0,width+1])
    plt.ylim([-60,0])
    plt.axvline(x= tmin_3pt, color = 'black', linestyle = '--')
    plt.legend()

    plt.subplot(2,2,2)
    plt.errorbar(t_space, corrs_3pt_means[1], yerr= corrs_3pt_errs[1], label = 'Original Corr',capsize = 4, linestyle = '')
    plt.errorbar(t_space, reconned_3pt_means[1], yerr= reconned_3pt_errs[1], label = 'Recon Corr', capsize = 4,linestyle = '')
    plt.title('ln {} {}'.format(ensemble['tag'], tags_3pt[1]))
    plt.xlim([0,width+1])
    plt.ylim([-60,0])
    plt.axvline(x= tmin_3pt, color = 'black', linestyle = '--')
    plt.legend()

    plt.subplot(2,2,3)
    plt.errorbar(t_space, corrs_3pt_means[2], yerr= corrs_3pt_errs[2], label = 'Original Corr',capsize = 4, linestyle = '')
    plt.errorbar(t_space, reconned_3pt_means[2], yerr= reconned_3pt_errs[2], label = 'Recon Corr', capsize = 4,linestyle = '')
    plt.title('ln {} {}'.format(ensemble['tag'], tags_3pt[2]))
    plt.xlim([0,width+1])
    plt.ylim([-60,0])
    plt.axvline(x= tmin_3pt, color = 'black', linestyle = '--')
    plt.legend()

    plt.subplot(2,2,4)
    plt.errorbar(t_space, corrs_3pt_means[3], yerr= corrs_3pt_errs[3], label = 'Original Corr',capsize = 4, linestyle = '')
    plt.errorbar(t_space, reconned_3pt_means[3], yerr= reconned_3pt_errs[3], label = 'Recon Corr', capsize = 4,linestyle = '')
    plt.title('ln {} {}'.format(ensemble['tag'], tags_3pt[3]))
    plt.xlim([0,width+1])
    plt.ylim([-60,0])
    plt.axvline(x= tmin_3pt, color = 'black', linestyle = '--')
    plt.legend()

    plt.savefig('./comparing/compare_plots/corr_3pt_plot.png')
    print('Plot successfully saved to /comparing/compare_plots/corr_3pt_plot.png')


############# 3pt Effective Amplitudes ###############
    # From GPL
    J_effs = []
    for corr in corrs_3pt:
        J = calc_J_eff(corr, aM_eff_mother, aM_eff_daughter, Amp_eff_mother, Amp_eff_daughter, width, Nt)
        J_eff_mean, junk = gvar_splitter(J)
        J_effs.append(J_eff_mean)

    # Reconstructed J_effs
    reconned_J_effs =[]
    for corr in reconned_3pt_corrs:
        J = calc_J_eff(corr, aM_eff_mother_recon, aM_eff_daughter_recon, Amp_eff_mother_recon, Amp_eff_daughter_recon, width, Nt)
        J_eff_mean, junk = gvar_splitter(J)
        reconned_J_effs.append(J_eff_mean)

    ### Plotting

    plt.figure(figsize=(14, 10))

    plt.subplot(2,2,1)
    plt.scatter(t_space, J_effs[0], marker = 'x', label = 'Original J_eff')
    plt.scatter(t_space, reconned_J_effs[0], marker = '+', label = 'Recon J_eff')
    plt.title('J_eff {} {}'.format(ensemble['tag'], tags_3pt[0]))
    plt.ylim([-1,5])
    plt.xlim([0,width])
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.legend()

    plt.subplot(2,2,2)
    plt.scatter(t_space, J_effs[1], marker = 'x', label = 'Original J_eff')
    plt.scatter(t_space, reconned_J_effs[1], marker = '+', label = 'Recon J_eff')
    plt.title('J_eff {} {}'.format(ensemble['tag'], tags_3pt[1]))
    plt.ylim([-1,5])
    plt.xlim([0,width])
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.legend()

    plt.subplot(2,2,3)
    plt.scatter(t_space, J_effs[2], marker = 'x', label = 'Original J_eff')
    plt.scatter(t_space, reconned_J_effs[2], marker = '+', label = 'Recon J_eff')
    plt.title('J_eff {} {}'.format(ensemble['tag'], tags_3pt[2]))
    plt.ylim([-1,5])
    plt.xlim([0,width])
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.legend()

    plt.subplot(2,2,4)
    plt.scatter(t_space, J_effs[3], marker = 'x', label = 'Original J_eff')
    plt.scatter(t_space, reconned_J_effs[3], marker = '+', label = 'Recon J_eff')
    plt.title('J_eff {} {}'.format(ensemble['tag'], tags_3pt[3]))
    plt.ylim([-1,5])
    plt.xlim([0,width])
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.legend()
    plt.savefig('./comparing/compare_plots/J_eff_3pt_plot.png')
    print('Plot successfully saved to /comparing/compare_plots/J_eff_3pt_plot.png')
#########################################
# Making error comparison plots
#########################################
# 2pt Error comparison
    plt.figure(figsize=(14, 10))
    #mother meson
    plt.subplot(3,2,1)
    plt.plot(t_space,(np.array(log_corr_mother_errs) - np.array(log_recon_corr_mother_errs)))
    plt.title('Mother Meson Corr - Recon errs')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.ylim([-0.2,1])
    plt.xlim([0,Nt])

    #daughter meson
    plt.subplot(3,2,2)
    plt.plot(t_space,(np.array(log_corr_daughter_errs) - np.array(log_recon_corr_daughter_errs)))
    plt.title('Daughter Meson Corr - Recon errs')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.ylim([-0.2,1])
    plt.xlim([0,Nt])

    #scalar comp
    plt.subplot(3,2,3)
    plt.plot(t_space,(np.array(corrs_3pt_errs[0]) - np.array(reconned_3pt_errs[0])))
    plt.title('3pt Scalar Comp Corr - Recon errs')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.axvline(x = width, color = 'red',linestyle = '--')
    plt.ylim([-10,10])
    plt.xlim([0,width])

    #temporal vector comp
    plt.subplot(3,2,4)
    plt.plot(t_space,(np.array(corrs_3pt_errs[1]) - np.array(reconned_3pt_errs[1])))
    plt.title('3pt Temporal Vector Comp Corr - Recon errs')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.axvline(x = width, color = 'red',linestyle = '--')
    plt.ylim([-10,10])
    plt.xlim([0,width])

    #spacial vector comp
    plt.subplot(3,2,5)
    plt.plot(t_space,(np.array(corrs_3pt_errs[2]) - np.array(reconned_3pt_errs[2])))
    plt.title('3pt Spacial Vector Comp Corr - Recon errs')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.axvline(x = width, color = 'red',linestyle = '--')
    plt.ylim([-10,10])
    plt.xlim([0,width])

    #Tensor vector comp
    plt.subplot(3,2,6)
    plt.plot(t_space,(np.array(corrs_3pt_errs[3]) - np.array(reconned_3pt_errs[3])))
    plt.title('3pt Tensor Comp Corr - Recon errs')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.axvline(x = width, color = 'red',linestyle = '--')
    plt.ylim([-10,10])
    plt.xlim([0,width])

    plt.savefig('./comparing/compare_plots/error_comparison.png')
    print('Plot successfully saved to /comparing/compare_plots/error_comparison.png')'''
################################################################################################################################
#added jul 11 2025
def aE_effective_plotting(ens, strange):
    if strange == True:
        spectator, daughter = 's', 'kaon'
        s_tag, s3_tag = 'true' , 'HsK'
    else: spectator, daughter, s_tag, s3_tag = 'l', 'pion', 'false', 'Hpi'
    corr_txt_file = ens['avg']
    Nt = ens['tp']
    t_space = np.arange(0, Nt, 1)
    colors = ['red', 'blue', 'green', 'purple', 'tab:orange', 'tab:cyan']
    plt.figure(figsize = (10,5))
    tag_array, corr_array = load_Corr_txt(corr_txt_file)
    tw = 0
    for twist in ens['twists']:
        tag_mother, corr_mother, tag_daughter, corr_daughter = Corr2pt_picker(strange, ens['masses'][mass_choice], twist, tag_array, corr_array)
        aM_eff_mother, aE_eff_daughter = calc_aM_eff(corr_mother), calc_aM_eff(corr_daughter)
        aE_eff_daughter_means, aE_eff_daughter_errs = gvar_splitter(aE_eff_daughter)
        roll_avg = []
        for i in range(len(aE_eff_daughter_means)):
            if i + 1 == Nt:
                end = -1
            else: end = i+1
            avg = (aE_eff_daughter_means[i-1] + aE_eff_daughter_means[i] + aE_eff_daughter_means[end])/3
            roll_avg.append(avg)
        plt.errorbar(t_space, aE_eff_daughter_means, yerr= aE_eff_daughter_errs, capsize=2, linestyle = '', label = 'aE_eff tw{}'.format(twist), color = colors[tw])
        plt.plot(t_space, roll_avg, linestyle = ':', color = colors[tw])
        tw +=1
    plt.ylim([0,1])
    plt.yticks(np.linspace(0,1,21))
    plt.grid(visible=True, axis= 'y', which= 'both')
    plt.xlim([ens['tmin_meson'],Nt/3])
    plt.title('{} Effective Energy {}-ensmble '.format(s3_tag, ens['tag']))
    plt.legend(loc = 1)
    plt.savefig('./comparing/aEeff_plots/aE_{}_{}.pdf'.format(ens['tag'],s3_tag), format = 'pdf')
    print('Saved to ./comparing/aEeff_plots/aE_{}_{}.pdf'.format(ens['tag'],s3_tag))
################################################################################################################################


def two_point_priors():
    plt.figure(figsize=(10, 5))
    tag_array, corr_array = load_Corr_txt(corr_txt_file)
    ticker, colors = 0, ['red','blue', 'green', 'purple']
    for mass in ensemble['masses']:    
        tag_mother, corr_mother, tag_daughter, corr_daughter = Corr2pt_picker(strange, mass, twist, tag_array, corr_array)
        aM_eff_mother, aM_eff_daughter = calc_aM_eff(corr_mother), calc_aM_eff(corr_daughter)
        aM_eff_mother_means, aM_eff_mother_errs = gvar_splitter(aM_eff_mother)
        aM_eff_daughter_means, aM_eff_daughter_errs = gvar_splitter(aM_eff_daughter)
        ################################################## PLotting 2pt Effective Masses
        '''plt.subplot(1,2,1)
        plt.errorbar(t_space, aM_eff_mother_means, yerr= aM_eff_mother_errs, label = 'am = {}'.format(mass), capsize=2, linestyle = '', color = colors[ticker])
        plt.title('{}-ensemble {} mother meson effective masses'.format(ensemble['tag'], s3_tag))
        plt.axvline(x=tmin_meson, color = 'black', linestyle = '--')
        plt.ylabel('a*M [unit-less]')
        plt.xlabel('t/a')
        #plt.ylim([0,0.50])
        plt.xlim([0,Nt/2])
        plt.ylim([0.35,1.45])
        ticker += 1
        plt.legend()
        plt.yticks(np.linspace(0.35,1.5,24))
        plt.grid(visible=True, axis= 'y', which= 'both')'''
    #plt.subplot(1,2,2)
    plt.errorbar(t_space, aM_eff_daughter_means, yerr= aM_eff_daughter_errs, capsize=2, linestyle = '', label = 'Meff(t)')
    plt.title('{} {} daughter meson effective mass'.format(ensemble['tag'], s3_tag))
    #plt.axvline(x=tmin_meson, color = 'black', linestyle = '--')
    plt.axhline(y = 0.0572, color = 'red', linestyle = '--')
    plt.axhline(y = 0.05471, color = 'green', linestyle = '--')
    plt.axhline(y = 0.057229, color = 'purple', linestyle = '--')
    #(23)
    plt.fill_between(t_space, 0.0572 + 0.0029, 0.0572 - 0.0029, color = 'red', alpha = 0.2, label = 'Prior bounds on pion mass')
    plt.fill_between(t_space, 0.05471 + 0.00012, 0.05471 - 0.00012, color = 'green', alpha = 0.2, label = 'Global post. pion mass')
    plt.fill_between(t_space, 0.057229 + 0.000023, 0.057229 - 0.000023, color = 'purple', alpha = 0.4, label = '2pt Only post. pion mass')
    plt.xlim([0,Nt/2])
    plt.ylim([0.0475,0.0675])
    plt.xlabel('t/a')
    plt.ylabel('a*M_pi')
    plt.yticks(np.linspace(0.0475,0.0675, 17))
    plt.grid(visible=True, axis= 'y', which= 'both')
    plt.legend()

    filename = '{}_daughter={}_aMeffs_{}.pdf'.format(ensemble['tag'], daughter, datetime.datetime.now())
    plt.savefig('./comparing/aMeff_plots/{}'.format(filename), format = 'pdf')
    print('Plot successfully saved to /comparing/aMeff_plots/{}'.format(filename))





    print('Control complete')

'''def three_point_priors():
    plt.figure(figsize=(10, 10))
    tag_array, corr_array = load_Corr_txt(corr_txt_file)
    ticker, colors = 0, ['red','blue', 'green', 'purple']
    for mass in ensemble['masses']:
        for twist in ensemble['twists']:
            tag_mother, corr_mother, tag_daughter, corr_daughter = Corr2pt_picker(strange, mass, twist, tag_array, corr_array)
            aM_eff_mother, aM_eff_daughter = calc_aM_eff(corr_mother), calc_aM_eff(corr_daughter)
            Amp_eff_mother, Amp_eff_daughter = calc_Amp_eff(corr_mother, aM_eff_mother, Nt), calc_Amp_eff(corr_daughter, aM_eff_daughter, Nt)
            tags_3pt, corrs_3pt = Corr3pt_picker(strange, mass, twist, width, tag_array, corr_array)
            J_effs, J_errs = [], []
            for corr in corrs_3pt:
                J = calc_J_eff(corr, aM_eff_mother, aM_eff_daughter, Amp_eff_mother, Amp_eff_daughter, width, Nt)
                J_eff_mean, J_err = gvar_splitter(J)
                J_effs.append(J_eff_mean)
                J_errs.append(J_err)
            plt.subplot(2,2,1)
            plt.errorbar(t_space, J_effs[0], yerr=J_errs[0], capsize=2, linestyle = '', color = colors[ticker])
            plt.ylim([-1,10])
            plt.xlim([0,width])
            plt.xlabel('t/a')
            plt.ylabel('A_eff')
            plt.grid(visible=True, axis= 'y', which= 'both')
            plt.axvline(x=tmin_3pt, color = 'black', linestyle = '--')
            plt.axhline(y = 0, color = 'black', linestyle = '--')
            plt.title('{} {} Scalar A_eff'.format(ensemble['tag'], s3_tag))

            plt.subplot(2,2,2)
            plt.errorbar(t_space, J_effs[1], yerr=J_errs[1], capsize=2, linestyle = '', color = colors[ticker])
            plt.ylim([-8,10])
            plt.xlim([0,width])
            plt.xlabel('t/a')
            plt.ylabel('A_eff')
            plt.grid(visible=True, axis= 'y', which= 'both')
            plt.axvline(x=tmin_3pt, color = 'black', linestyle = '--')
            plt.axhline(y = 0, color = 'black', linestyle = '--')
            plt.title('{} {} Temporal Vector A_eff'.format(ensemble['tag'], s3_tag))

            plt.subplot(2,2,3)
            plt.errorbar(t_space, J_effs[2], yerr=J_errs[2], capsize=2, linestyle = '', color = colors[ticker])
            plt.ylim([-1,10])
            plt.xlim([0,width])
            plt.xlabel('t/a')
            plt.ylabel('A_eff')
            plt.grid(visible=True, axis= 'y', which= 'both')
            plt.axvline(x=tmin_3pt, color = 'black', linestyle = '--')
            plt.axhline(y = 0, color = 'black', linestyle = '--')
            plt.title('{} {} Spacial Vector A_eff'.format(ensemble['tag'], s3_tag))

            plt.subplot(2,2,4)
            plt.errorbar(t_space, J_effs[3], yerr=J_errs[3], capsize=2, linestyle = '', color = colors[ticker])
            plt.ylim([-1,10])
            plt.xlim([0,width])
            plt.xlabel('t/a')
            plt.ylabel('A_eff')
            plt.grid(visible=True, axis= 'y', which= 'both')
            plt.axvline(x=tmin_3pt, color = 'black', linestyle = '--')
            plt.axhline(y = 0, color = 'black', linestyle = '--')
            plt.title('{} {} Tensor A_eff'.format(ensemble['tag'], s3_tag))
        ticker += 1
    filename = '{}_{}_Vamps_{}.svg'.format(ensemble['tag'], s3_tag ,datetime.datetime.now())
    plt.savefig('./comparing/3pt_prior_plots/{}'.format(filename), format = 'svg')
    print('Plot successfully saved to /comparing/3pt_prior_plots/{}'.format(filename))'''

def three_point_priors(tw, ci):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    tag_array, corr_array = load_Corr_txt(corr_txt_file)
    ticker, colors = 0, ['red','blue', 'green', 'purple']
    twist_tick = -0.25
    for mass in ensemble['masses']:
        for twist in ensemble['twists']:
            tag_mother, corr_mother, tag_daughter, corr_daughter = Corr2pt_picker(strange, mass, twist, tag_array, corr_array)
            aM_eff_mother, aM_eff_daughter = calc_aM_eff(corr_mother), calc_aM_eff(corr_daughter)
            Amp_eff_mother, Amp_eff_daughter = calc_Amp_eff(corr_mother, aM_eff_mother, Nt), calc_Amp_eff(corr_daughter, aM_eff_daughter, Nt)
            tags_3pt, corrs_3pt = Corr3pt_picker(strange, mass, twist, width, tag_array, corr_array)
            J_effs, J_errs = [], []
            for corr in corrs_3pt:
                J = calc_J_eff(corr, aM_eff_mother, aM_eff_daughter, Amp_eff_mother, Amp_eff_daughter, width, Nt)
                J_eff_mean, J_err = gvar_splitter(J)
                J_effs.append(J_eff_mean)
                J_errs.append(J_err)
            if twist == ensemble['twists'][tw]:
                plt.errorbar(t_space+twist_tick, J_effs[ci], yerr=J_errs[ci], capsize=4, linestyle = '', color = colors[ticker], marker = 'x',label = 'amₕ={}'.format(mass))
                #
                ravg = [] #plotting ravg of Veff
                for i in range(len(J_effs[ci][1:-3])):
                    ravg.append((J_effs[ci][i-1] + J_effs[ci][i] + J_effs[ci][i+1] + J_effs[ci][i+2])/4)
                #print(ravg)
                plt.plot(t_space[1:-3]+twist_tick, ravg, color = colors[ticker], linestyle = ':')
            if ci > 1: 
                plt.ylim([-0.1,1.0])
                plt.yticks(np.linspace(-.1, 1.0, 23))
            else: 
                plt.ylim([-0.25,6])
                plt.yticks(np.linspace(-.25, 6.0, 26))
            plt.xlim([ensemble['tmin_3pt']+5,width+0.5])
            plt.xlabel('t/a')
            plt.ylabel('Effective Amplitude')
            plt.grid(visible=True, axis= 'y', which= 'both')
            plt.axvline(x=tmin_3pt, color = 'black', linestyle = '--')
            plt.title('{} {} {} Veff, θ = {}'.format(ensemble['tag'], s3_tag, currs[ci],ensemble['twists'][tw]))
            plt.legend()
        twist_tick+=0.125
        ticker += 1
    filename = '{}_{}_Vamps_{}_{}.pdf'.format(ensemble['tag'], s3_tag, currs[ci], tw) #,datetime.datetime.now())
    plt.savefig('./comparing/3pt_prior_plots/{}_{}/{}'.format(ensemble['tag'], s3_tag, filename), format = 'pdf')
    print('Plot successfully saved to /comparing/3pt_prior_plots/{}_{}/{}'.format(ensemble['tag'], s3_tag,filename))

def plot2pt0twE():
    plt.rcParams["font.family"]= 'serif'
    ticker = 0
    #plt.figure(figsize=(25, 25))
    plt.figure(figsize=(6, 6)) #custom plotting for thesis
    #for ens in [F, Fp, SF, SFp, UF]:
    for ens in [F, UF]: #custom plotting for thesis
        corr_txt_file = ens['avg']
        tag_array, corr_array = load_Corr_txt(corr_txt_file)
        #for meson in ['pion', 'kaon']:
        for meson in ['pion']: #custom plotting for thesis
            #plt.subplot(5,2,ticker+1)
            plt.subplot(2,1,ticker+1) #custom plotting for thesis
            plt.locator_params(axis='x', integer=True)
            plt.xlim(0, ens['tp']/3)
            C2_tag = '2pt_{}G5-G5_th0.0'.format(meson)
            index = list(tag_array).index(C2_tag)
            C2 = corr_array[index]

            M_effs = []
            for t in range(2,len(C2)-2):
                thing  = (C2[t-2] + C2[t+2])/(2*C2[t]) 
                if thing >= 1:
                    M_effs.append(gv.arccosh(thing)/2)
            #M_effs is all positive masses, we now take a rolling average of 4, and find where this changes the least
            rav = []
            for i in range(len(M_effs)-4):
                rav.append((M_effs[i] + M_effs[i+1] + M_effs[i+2] + M_effs[i+3])/4)
            M_eff = rav[0]
            diff = abs((rav[1] - rav[0]).mean)
            for i in range(1,len(rav)-1):
                if abs((rav[i+1]-rav[i]).mean) < diff:
                    diff = abs((rav[i+1]-rav[i]).mean)
                    M_eff = (rav[i] + rav[i+1])/2 

            means, errs = gvar_splitter(M_effs)
            rav_means, rav_errs = gvar_splitter(rav)
            t_space = np.arange(0, ens['tp'], 1)[2:-2]
            plt.errorbar(t_space, means, yerr=errs, capsize=2, fmt = 'o', mfc = 'none', linestyle = '', color = 'blue', label = r'$C_2^\pi$ $aM_{{\mathrm{{eff}}}}(t)$')
            plt.plot(t_space[0:-4], rav_means, color = 'green', linestyle = '--', label = r'R.Avg$(t)$')
            plt.ylim([M_eff.mean-M_eff.mean*0.6,M_eff.mean+M_eff.mean*0.6])
            plt.xlim([ens['tp']/30, ens['tp']/5])
            
            plt.axhline(M_eff.mean, linestyle = ':', color = 'red', label = r'$\Delta_\mathrm{{min}}$ of $\mathrm{{R.Avg}}(t)$')
            plt.axhspan(M_eff.mean - M_eff.mean*0.3, M_eff.mean + M_eff.mean*0.3, color = 'red', alpha = 0.1)
            plt.axhspan(M_eff.mean - M_eff.mean*0.25, M_eff.mean + M_eff.mean*0.25, color = 'red', alpha = 0.1)
            plt.axhspan(M_eff.mean - M_eff.mean*0.2, M_eff.mean + M_eff.mean*0.2, color = 'red', alpha = 0.1)
            plt.axhspan(M_eff.mean - M_eff.mean*0.15, M_eff.mean + M_eff.mean*0.15, color = 'red', alpha = 0.1)
            plt.axhspan(M_eff.mean - M_eff.mean*0.1, M_eff.mean + M_eff.mean*0.1, color = 'red', alpha = 0.1)
            plt.axhspan(M_eff.mean - M_eff.mean*0.05, M_eff.mean + M_eff.mean*0.05, color = 'red', alpha = 0.1, label = r'$\Delta_\mathrm{{min}}$ of $\mathrm{{R.Avg}}(t)$ $\pm 5 \%$')
            #plt.scatter(t_slice, A_eff.mean, color = 'green', marker='x')
            
            plt.xlabel(r'$t/a$')
            plt.ylabel(r'$aM_{{\mathrm{{eff}}}}(t)$')
            #plt.title('{} ensemble {} effective mass'.format(ens['tag'], meson))
            plt.title(r'{} ensemble pion $M_{{\mathrm{{eff}}}}(t)$'.format(ens['alt_tag'])) #custom plotting for thesis
            plt.minorticks_on()
            plt.legend(frameon=False, loc = 3, ncols = 2)
            #plt.savefig('./comparing/2pt_amps/{}_{}_0tw.pdf'.format(ens['tag'], meson))
            #plt.close()
            ax = plt.gca()
            plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
            plt.tick_params(axis = 'x',  which= 'both', top = False, direction = 'in')
            
            ticker +=1
    plt.tight_layout()
    plt.savefig('./comparing/2pt_amps/CustomEs.pdf')
    print('Done!')    


def plot_2pt0twAmp(Flat_Meff = False):
    plt.rcParams["font.family"]= 'serif'
    ticker = 0
    plt.figure(figsize=(6, 6))
    #for ens in [F, Fp, SF, SFp, UF]:
    for ens in [F, UF]:
        corr_txt_file = ens['avg']
        tag_array, corr_array = load_Corr_txt(corr_txt_file)
        #for meson in ['pion', 'kaon']:
        for meson in ['pion']:
            plt.subplot(2,1,ticker+1)
            plt.xlim([ens['tp']/30, ens['tp']/5])
            C2_tag = '2pt_{}G5-G5_th0.0'.format(meson)
            index = list(tag_array).index(C2_tag)
            C2 = corr_array[index]
            aMeff = calc_aM_eff(C2)

            M_effs = []
            for t in range(2,len(C2)-2):
                thing  = (C2[t-2] + C2[t+2])/(2*C2[t]) 
                if thing >= 1:
                    M_effs.append(gv.arccosh(thing)/2)
            #M_effs is all positive masses, we now take a rolling average of 4, and find where this changes the least
            rav = []
            for i in range(len(M_effs)-4):
                rav.append((M_effs[i] + M_effs[i+1] + M_effs[i+2] + M_effs[i+3])/4)
            M_eff = rav[0]
            diff = abs((rav[1] - rav[0]).mean)
            for i in range(1,len(rav)-1):
                if abs((rav[i+1]-rav[i]).mean) < diff:
                    diff = abs((rav[i+1]-rav[i]).mean)
                    M_eff = (rav[i] + rav[i+1])/2 

            if Flat_Meff == False: M_eff = aMeff #aMeff(t) here

            A_effs = calc_Amp_eff(C2, M_eff, ens['tp'])#[ens['tmin_meson']:ens['tmax_meson']]
            
            #rav
            rav = []
            for i in range(len(A_effs)-4):
                rav.append((A_effs[i] + A_effs[i+1] + A_effs[i+2] + A_effs[i+3])/4)
            A_eff = rav[0]
            diff = abs((rav[1] - rav[0]).mean)
            for i in range(1,len(rav)-1):
                if abs((rav[i+1]-rav[i]).mean) < diff:
                    diff = abs((rav[i+1]-rav[i]).mean)
                    A_eff = (rav[i] + rav[i+1])/2
                    t_slice = i
            #print(t_slice)

            #print(Aeff)
            means, errs = gvar_splitter(A_effs)
            rav_means, rav_errs = gvar_splitter(rav)
            t_space = np.arange(0, ens['tp'], 1)#[ens['tmin_meson']:ens['tmax_meson']]
            plt.errorbar(t_space, means, yerr=errs, capsize=2, fmt = 'o', mfc = 'none', linestyle = '', color = 'blue', label = r'$C_2^\pi$ $A_{{\mathrm{{eff}}}}(t)$')
            plt.plot(t_space[0:-4], rav_means, color = 'green', linestyle = '--', label = r'R.Avg$(t)$')
            plt.ylim([A_eff.mean-A_eff.mean*0.7,A_eff.mean+A_eff.mean*0.7])
            
            plt.axhline(A_eff.mean, linestyle = ':', color = 'red', label = r'$\Delta_\mathrm{{min}}$ of $\mathrm{{R.Avg}}(t)$')
            plt.axhspan(A_eff.mean - A_eff.mean*0.3, A_eff.mean + A_eff.mean*0.3, color = 'red', alpha = 0.1)
            plt.axhspan(A_eff.mean - A_eff.mean*0.25, A_eff.mean + A_eff.mean*0.25, color = 'red', alpha = 0.1)
            plt.axhspan(A_eff.mean - A_eff.mean*0.2, A_eff.mean + A_eff.mean*0.2, color = 'red', alpha = 0.1)
            plt.axhspan(A_eff.mean - A_eff.mean*0.15, A_eff.mean + A_eff.mean*0.15, color = 'red', alpha = 0.1)
            plt.axhspan(A_eff.mean - A_eff.mean*0.1, A_eff.mean + A_eff.mean*0.1, color = 'red', alpha = 0.1)
            plt.axhspan(A_eff.mean - A_eff.mean*0.05, A_eff.mean + A_eff.mean*0.05, color = 'red', alpha = 0.1, label = r'$\Delta_\mathrm{{min}}$ of $\mathrm{{R.Avg}}(t)$ $\pm 5 \%$')
            #plt.scatter(t_slice, A_eff.mean, color = 'green', marker='x')
            
            plt.xlabel(r'$t/a$')
            plt.ylabel(r'$A_{{\mathrm{{eff}}}}(t)$')
            #plt.title('{} ensemble {} effective mass'.format(ens['tag'], meson))
            plt.title(r'{} ensemble pion $A_{{\mathrm{{eff}}}}(t)$'.format(ens['alt_tag']))
            plt.minorticks_on()
            plt.legend(frameon=False, loc = 3, ncols = 2)
            #plt.savefig('./comparing/2pt_amps/{}_{}_0tw.pdf'.format(ens['tag'], meson))
            #plt.close()
            ax = plt.gca()
            plt.tick_params(axis='y', which='both', labelright=True, right=True, direction = 'in')
            plt.tick_params(axis = 'x',  which= 'both', top = False, direction = 'in')
            ticker +=1
    plt.tight_layout()
    plt.savefig('./comparing/2pt_amps/CustomAmps_Flat_Meff={}.pdf'.format(Flat_Meff))
    print('Done!')

def get_all_2pt_amps():
    plt.figure(figsize=(10, 5))
    ticker, colors = 0, ['red','blue', 'green', 'purple']
    tag_array, corr_array = load_Corr_txt(corr_txt_file)
    for mass in ensemble['masses']:
        for twist in ensemble['twists'][0]:
            tag_mother, corr_mother, tag_daughter, corr_daughter = Corr2pt_picker(strange, mass, twist, tag_array, corr_array)
            aM_eff_mother, aM_eff_daughter = calc_aM_eff(corr_mother), calc_aM_eff(corr_daughter)
            Amp_eff_mother, Amp_eff_daughter = calc_Amp_eff(corr_mother, aM_eff_mother, Nt), calc_Amp_eff(corr_daughter, aM_eff_daughter, Nt)
            Amp_eff_mother_means, Amp_eff_mother_errs = gvar_splitter(Amp_eff_mother)
            Amp_eff_daughter_means, Amp_eff_daughter_errs = gvar_splitter(Amp_eff_daughter)
            plt.errorbar(t_space, Amp_eff_mother_means, yerr=Amp_eff_mother_errs, capsize=2, linestyle = '', color = colors[ticker])
            plt.errorbar(t_space, Amp_eff_daughter_means, yerr=Amp_eff_daughter_errs, capsize=2, linestyle = '', color = colors[ticker])
        ticker += 1
    plt.title('{} {} 2pt Effective Amplitudes'.format(ensemble['tag'],s3_tag))
    filename = '{}_{}_2pt_amps_{}.pdf'.format(ensemble['tag'], s3_tag ,datetime.datetime.now())
    plt.grid(visible=True, axis= 'y', which= 'both')
    plt.yticks(np.linspace(-0.25,1.5,15))
    plt.xlim([0,Nt/2])
    plt.ylim([-0.25,1.5])
    plt.xlabel('t/a')
    plt.ylabel('A_eff')

    plt.savefig('./comparing/2pt_amps/{}'.format(filename), format = 'pdf')
    print('Plot successfully saved to /comparing/2pt_amps/{}'.format(filename))

def function_choice(choice):
    if choice == 0:
        pass
    elif choice == 1:
        two_point_priors()
    elif choice == 2:
        for i in range(4):
            ci = i
            for tw in range(len(ensemble['twists'])):
                three_point_priors(tw, ci)
    elif choice == 3:
        get_all_2pt_amps()
    elif choice ==4:
        aE_effective_plotting()
    else: print('invalid function choice')

#function_choice(function_output_choice)
#plot2pt0twE()
plot_2pt0twAmp(Flat_Meff=False)

'''for ens in (F, Fp, SF, SFp, UF):
    #corr_txt_file = ens['avg']
    strange = False
    aE_effective_plotting(ens, strange)
    strange = True
    aE_effective_plotting(ens, strange)'''





