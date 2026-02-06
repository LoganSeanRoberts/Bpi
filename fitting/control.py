import collections
import sys
import h5py
import gvar as gv
import numpy as np
import corrfitter as cf
import csv
#import corrbayes
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from functions import *
from plotting import *
plt.rc("font",**{"size":18})
import os.path
import pickle
import datetime
#import pvc_fitter as vc
#Global tmin adjuster for stability testing
tmin_2pt_adjuster = 0
tmin_3pt_adjuster = 0
################# F #######################################
F = collections.OrderedDict()
F['conf'] = 'F'
F['filename'] = "binned_f5_all"
F['file_location'] = "./gpls/"
F['masses'] = ['0.450','0.55','0.675','0.8']
F['twists'] = ['0.0','0.4281','1.282','2.1410','2.570']
F['m_l'] = '0.0074'
F['m_s'] = '0.0376'
F['Ts'] = [15,18,21,24]
F['tp'] = 96
F['L'] = 32
F['tmaxesB5'] = [48,48,48,48]
F['tmaxesB5T'] = [48,48,48,48]
F['tmaxesB5X'] = [48,48,48,48]
F['tmaxesBYZ'] = [48,48,48,48]
F['tmaxesBs5'] = [48,48,48,48]
F['tmaxesBs5T'] = [48,48,48,48]
F['tmaxesBs5X'] = [48,48,48,48]
F['tmaxesBsYZ'] = [48,48,48,48]
#F['tmaxesK'] = [48,48,48,35,30]
#F['tmaxespi'] = [48,48,48,30,25]
F['tmaxesK'] = [48,48,48,32,28] #adjusted Jul 11 2025
F['tmaxespi'] = [48,48,32,25,20]
#########################
F['tmin_2pt'] = 4 + tmin_2pt_adjuster       #sets tmin for all 2 pts
F['tmin_3pt'] = 2 + tmin_3pt_adjuster      #sets tmin for all 3 pts
#########################
F['tminB5'] = F['tmin_2pt'] 
F['tminB5T'] = F['tmin_2pt'] 
F['tminB5X'] = F['tmin_2pt'] 
F['tminBYZ'] = F['tmin_2pt'] 
F['tminBs5'] = F['tmin_2pt'] 
F['tminBs5T'] = F['tmin_2pt'] 
F['tminBs5X'] = F['tmin_2pt'] 
F['tminBsYZ'] = F['tmin_2pt'] 
F['tminK'] = F['tmin_2pt'] 
F['tminpi'] = F['tmin_2pt'] 
F['Stmin'] = F['tmin_3pt']
F['Vtmin'] = F['tmin_3pt']
F['Ttmin'] = F['tmin_3pt']
F['Xtmin'] = F['tmin_3pt']
F['Sstmin'] = F['tmin_3pt']
F['Vstmin'] = F['tmin_3pt']
F['Tstmin'] = F['tmin_3pt']
F['Xstmin'] = F['tmin_3pt']
#########################
F['En_Width_Fraction'] = 0.5
######################### 
F['Mloosener'] = 0.05          #Loosener on ground state mass H meson only!
F['oMloosener'] = 0.2                        #Loosener on oscillating ground state mass #H AND pi/K
F['Api_loosener'] = 0.05    #0twist pion effective amp plot derrived
F['AK_loosener'] = 0.03
F['Aeff_0_loosener'] = 0.11     #H meson only
F['m3/m0_width_rat'] = 2        #H meson amp scaling going from lowest mass to highest out of [0,1,2,3]
F['Aeff_n_loosener'] = 5.0
F['V10_widener'] = 1.8
F['Vloosener'] = 1                          #Specific prior widenor that only affects all 3pts
F['a'] = 0.1715/(1.9006*0.1973)  #wilson coefficient w_o divided by (w_o/a) = a.  Then divided by 0.1973 -> a in units of GeV
F['aE_pion_loosner'] = 0.05
F['aE_kaon_loosner'] = 0.02
#########################
F['an'] = '{}({})'.format(0.2,0.2)
F['aon'] = '{}({})'.format(0.03, 0.03)   
F['SVnn0'] = '1.6({})'.format(1.6)      #updated                     #Prior for SV_nn[0][0]
F['SVnn0'] = ['2.6(2.6)', '2.4(2.4)', '1.5(1.5)', '1.0(1.0)', '0.8(0.8)']
F['SVn'] = '0.0({})'.format(0.5) #0.5                           #Prior for SV_??[n][n]
F['SV0'] = '0.0({})'.format(1.75) #1.0                             #Prior for SV_no[0][0] etc
F['VVnn0'] = '1.1({})'.format(1.1)   
F['VVnn0'] = ['1.0(1.0)', '0.9(0.9)' , '0.6(0.6)', '0.5(0.5)', '0.4(0.4)']
F['VVn'] = '0.0({})'.format(0.7) #0.5  
F['VV0'] = '0.0({})'.format(3.5) #3.5
F['XVnn0'] = '0.25({})'.format(0.10)   
F['XVnn0'] = ['0(0)', '0.15(0.15)', '0.22(0.22)', '0.2(0.2)', '0.2(0.2)']
F['XVn'] = '0.0({})'.format(0.10) #0.5 
F['XV0'] = '0.0({})'.format(0.22) #1.0
F['TVnn0'] = '0.21({})'.format(0.10)
F['TVnn0'] = ['0(0)', '0.1(0.1)', '0.15(0.15)', '0.12(0.12)', '0.13(0.13)']
F['TVn'] = '0.0({})'.format(0.07) #0.5
F['TV0'] = '0.0({})'.format(0.12)  #1.0
F['SsVnn0'] = '1.3({})'.format(1.3 * 0.5)                 #Prior for SV_nn[0][0]
F['SsVnn0'] = ['1.8(1.8)', '1.75(1.75)', '1.35(1.35)', '1.0(1.0)', '0.85(0.85)']
F['SsVn'] = '0.0({})'.format(0.43) #0.5                            #Prior for SV_??[n][n]
F['SsV0'] = '0.0({})'.format(1.1) #1.0                                #Prior for SV_no[0][0] etc
F['VsVnn0'] = '0.9({})'.format(0.5)       
F['VsVnn0'] = ['0.8(0.8)', '0.75(0.75)', '0.6(0.6)', '0.5(0.5)', '0.5(0.5)']
F['VsVn'] = '0.0({})'.format(0.75) #0.5
F['VsV0'] = '0.0({})'.format(2.0) #1.0  
F['XsVnn0'] = '0.18({})'.format(0.10)   
F['XsVnn0'] = ['', '0.1(0.1)', '0.2(0.2)', '0.22(0.22)', '0.3(0.3)']
F['XsVn'] = '0.0({})'.format(0.12)  #0.5 
#F['XsV0'] = '0.0({})'.format(0.24)  #1.0
F['XsV0'] = '0.0({})'.format(1.0)
F['TsVnn0'] = '0.16({})'.format(0.10)  
F['TsVnn0'] = ['', '0.07(0.07)', '0.12(0.12)', '0.14(0.14)', '0.15(0.15)']
F['TsVn'] = '0.0({})'.format(0.8) #0.5 
#F['TsV0'] = '0.0({})'.format(0.24) #1.0
F['TsV0'] = '0.0({})'.format(1.0) #1.0
#########################
F['B5-Tag'] = '2pt_HlG5-G5_m{0}_th0.0'
F['B5X-Tag'] = '2pt_HlG5-G5X_m{0}_th0.0'
F['B5T-Tag'] = '2pt_HlG5T-G5T_m{0}_th0.0'
F['BYZ-Tag'] = '2pt_HlG5T-GYZ_m{0}_th0.0'
F['Bs5-Tag'] = '2pt_HsG5-G5_m{0}_th0.0'
F['Bs5X-Tag'] = '2pt_HsG5-G5X_m{0}_th0.0'
F['Bs5T-Tag'] = '2pt_HsG5T-G5T_m{0}_th0.0'
F['BsYZ-Tag'] = '2pt_HsG5T-GYZ_m{0}_th0.0'
F['K-Tag'] = '2pt_kaonG5-G5_th{0}'
F['pi-Tag'] = '2pt_pionG5-G5_th{0}'
F['threePtTagS'] = '3pt_scalar_Hlpi_width{0}_m{2}_th{4}'
F['threePtTagV'] = '3pt_tempVec_Hlpi_width{0}_m{2}_th{4}'
F['threePtTagX'] = '3pt_spatVec_Hlpi_width{0}_m{2}_th{4}'
F['threePtTagT'] = '3pt_tensor_Hlpi_width{0}_m{2}_th{4}'  
F['threePtTagSs'] = '3pt_scalar_HsK_width{0}_m{2}_th{4}'
F['threePtTagVs'] = '3pt_tempVec_HsK_width{0}_m{2}_th{4}'
F['threePtTagXs'] = '3pt_spatVec_HsK_width{0}_m{2}_th{4}'
F['threePtTagTs'] = '3pt_tensor_HsK_width{0}_m{2}_th{4}'  
##########  
F['svd'] = 0.02
F['binsize'] = 1

################# Fp #######################################
Fp = collections.OrderedDict()
Fp['conf'] = 'Fp'
Fp['filename'] = "binned_fp_all"
Fp['file_location'] = "./gpls/"
Fp['masses'] = ['0.433','0.555','0.678','0.8']
Fp['twists'] = ['0.0','0.58','0.87','1.13','3.000','5.311']
Fp['m_l'] = '0.0012'
Fp['m_s'] = '0.036'
Fp['Ts'] = [15,18,21,24]
Fp['tp'] = 96
Fp['L'] = 64
Fp['tmaxesB5'] = [48,48,48,48]
Fp['tmaxesB5T'] = [48,48,48,48]
Fp['tmaxesB5X'] = [48,48,48,48]
Fp['tmaxesBYZ'] = [48,48,48,48]
Fp['tmaxesBs5'] = [48,48,48,48]
Fp['tmaxesBs5T'] = [48,48,48,48]
Fp['tmaxesBs5X'] = [48,48,48,48]
Fp['tmaxesBsYZ'] = [48,48,48,48]
#Fp['tmaxesK'] = [48,48,48,48,48,25]
#Fp['tmaxespi'] = [48,48,48,48,35,15]
Fp['tmaxesK'] = [48,48,48,48,40,18]
Fp['tmaxespi'] = [48,48,48,48,25,14]
#########################
Fp['tmin_2pt'] = 4 + tmin_2pt_adjuster   #sets tmin for all 2 pts
Fp['tmin_3pt'] = 2 + tmin_3pt_adjuster     #sets tmin for all 3 pts
#########################
Fp['tminB5'] = Fp['tmin_2pt']
Fp['tminB5T'] = Fp['tmin_2pt']
Fp['tminB5X'] = Fp['tmin_2pt']
Fp['tminBYZ'] = Fp['tmin_2pt']
Fp['tminBs5'] = Fp['tmin_2pt']
Fp['tminBs5T'] = Fp['tmin_2pt']
Fp['tminBs5X'] = Fp['tmin_2pt']
Fp['tminBsYZ'] = Fp['tmin_2pt']
Fp['tminK'] = Fp['tmin_2pt']
Fp['tminpi'] = Fp['tmin_2pt'] 
Fp['Stmin'] = Fp['tmin_3pt']
Fp['Vtmin'] = Fp['tmin_3pt']
Fp['Ttmin'] = Fp['tmin_3pt']
Fp['Xtmin'] = Fp['tmin_3pt']
Fp['Sstmin'] = Fp['tmin_3pt']
Fp['Vstmin'] = Fp['tmin_3pt']
Fp['Tstmin'] = Fp['tmin_3pt']
Fp['Xstmin'] = Fp['tmin_3pt']
#########################
Fp['En_Width_Fraction'] = 0.5
#############################
Fp['Mloosener'] = 0.05                        #Loosener on ground state mass
Fp['oMloosener'] = 0.25 ##                   #Loosener on oscillating ground state mass
Fp['Api_loosener'] = 0.05
Fp['AK_loosener'] = 0.03
Fp['Aeff_0_loosener'] = 0.22# <-Hpi # HsK -> 0.38#0.18
Fp['m3/m0_width_rat'] = 2.0
Fp['Aeff_n_loosener'] = 6.1 #6.2 #3.4
#Fp['Aeff_n_loosener'] = 1.0
Fp['V10_widener'] = 2.2
Fp['Vloosener'] = 1                       # 11/04/24 now only scales width of 3pt amp prior widths
Fp['a'] = 0.1715/(1.9518*0.1973)
Fp['aE_pion_loosner'] = 0.10
Fp['aE_kaon_loosner'] = 0.02
#########################
#Fp['aE_pi']= gv.gvar()
#########################
Fp['an'] = '{}({})'.format(0.2,0.2)         #defuct, now use automated method
Fp['aon'] = '{}({})'.format(0.02, 0.02)   
Fp['SVnn0'] = '3.0({})'.format(3.0)                      #Prior for SV_nn[0][0]
Fp['SVnn0'] = ['4.75(4.75)', '3.75(3.75)', '3.25(3.25)', '2.75(2.75)', '1.50(1.50)', '0.5(0.5)']
Fp['SVn'] = '0.0({})'.format(1.43)                         #Prior for SV_??[n][n]
Fp['SV0'] = '0.0({})'.format(4.4)                          #Prior for SV_no[0][0] etc
####
Fp['VVnn0'] = '1.8({})'.format(1.5)
Fp['VVnn0'] = ['3.0(3.0)', '2.5(2.5)', '2.0(2.0)', '1.5(1.5)', '0.75(0.75)', '0.25(0.25)']
Fp['VVn'] = '0.0({})'.format(1.0)
Fp['VV0'] = '0.0({})'.format(10.0)
####
Fp['XVnn0'] = '0.30({})'.format(0.30)
Fp['XVnn0'] = ['', '0.2(0.2)', '0.25(0.25)', '0.3(0.3)', '0.25(0.25)', '0.15(0.15)']
Fp['XVn'] = '0.0({})'.format(0.21)
Fp['XV0'] = '0.0({})'.format(0.06)
Fp['TVnn0'] = '0.25({})'.format(0.21)
Fp['TVnn0'] = ['', '0.15(0.15)', '0.18(0.18)', '0.15(0.15)', '0.15(0.15)', '0.1(0.1)']
Fp['TVn'] = '0.0({})'.format(0.08)
Fp['TV0'] = '0.0({})'.format(0.12)
Fp['SsVnn0'] = '1.2({})'.format(0.9)                        #Prior for SV_nn[0][0]
Fp['SsVnn0'] = ['1.90(1.90)', '1.75(1.75)', '1.75(1.75)', '1.75(1.75)', '1.25(1.25)', '0.5(0.5)']
Fp['SsVn'] = '0.0({})'.format(0.42)                          #Prior for SV_??[n][n]
Fp['SsV0'] = '0.0({})'.format(1.52)                         #Prior for SV_no[0][0] etc
Fp['VsVnn0'] = '0.9({})'.format(0.6)  
Fp['VsVnn0'] = ['1.0(1.0)', '0.9(0.9)', '0.85(0.85)', '0.8(0.8)', '0.6(0.6)', '0.3(0.3)']
Fp['VsVn'] = '0.0({})'.format(1.0) 
Fp['VsV0'] = '0.0({})'.format(3.3) 
Fp['XsVnn0'] = '0.16({})'.format(0.10) 
Fp['XsVnn0'] = ['', '0.1(0.1)', '0.12(0.12)', '0.15(0.15)', '0.2(0.2)', '0.15(0.15)']
Fp['XsVn'] = '0.0({})'.format(0.11) 
Fp['XsV0'] = '0.0({})'.format(0.2) 
Fp['TsVnn0'] = '0.15({})'.format(0.10) 
Fp['TsVnn0'] = ['', '0.05(0.05)', '0.08(0.08)', '0.1(0.1)', '0.15(0.15)', '0.1(0.1)']
Fp['TsVn'] = '0.0({})'.format(0.06) 
Fp['TsV0'] = '0.0({})'.format(0.22) 

#########################
Fp['B5-Tag'] = '2pt_HlG5-G5_m{0}_th0.0'
Fp['B5X-Tag'] = '2pt_HlG5-G5X_m{0}_th0.0'
Fp['B5T-Tag'] = '2pt_HlG5T-G5T_m{0}_th0.0'
Fp['BYZ-Tag'] = '2pt_HlG5T-GYZ_m{0}_th0.0'
Fp['Bs5-Tag'] = '2pt_HsG5-G5_m{0}_th0.0'
Fp['Bs5X-Tag'] = '2pt_HsG5-G5X_m{0}_th0.0'
Fp['Bs5T-Tag'] = '2pt_HsG5T-G5T_m{0}_th0.0'
Fp['BsYZ-Tag'] = '2pt_HsG5T-GYZ_m{0}_th0.0'
Fp['K-Tag'] = '2pt_kaonG5-G5_th{0}'
Fp['pi-Tag'] = '2pt_pionG5-G5_th{0}'
Fp['threePtTagS'] = '3pt_scalar_Hlpi_width{0}_m{2}_th{4}'
Fp['threePtTagV'] = '3pt_tempVec_Hlpi_width{0}_m{2}_th{4}'
Fp['threePtTagX'] = '3pt_spatVec_Hlpi_width{0}_m{2}_th{4}'
Fp['threePtTagT'] = '3pt_tensor_Hlpi_width{0}_m{2}_th{4}'  
Fp['threePtTagSs'] = '3pt_scalar_HsK_width{0}_m{2}_th{4}'
Fp['threePtTagVs'] = '3pt_tempVec_HsK_width{0}_m{2}_th{4}'
Fp['threePtTagXs'] = '3pt_spatVec_HsK_width{0}_m{2}_th{4}'
Fp['threePtTagTs'] = '3pt_tensor_HsK_width{0}_m{2}_th{4}'  
##########  
Fp['svd'] = 0.05
Fp['binsize'] = 1

#################SF #######################################
SF = collections.OrderedDict()
SF['conf'] = 'SF'
SF['filename'] = "binned_sf5_all"
SF['file_location'] = "./gpls/"
SF['masses'] = ['0.274','0.5','0.65','0.8']
SF['twists'] = ['0.0','1.261','2.108','2.666','5.059']
SF['m_l'] = '0.0048'
SF['m_s'] = '0.0234'
SF['Ts'] = [22,25,28,31]
SF['tp'] = 144
SF['L'] = 48
SF['tmaxesB5'] = [72,72,72,72]
SF['tmaxesB5T'] = [72,72,72,72]
SF['tmaxesB5X'] = [72,72,72,72]
SF['tmaxesBYZ'] = [72,72,72,72]
SF['tmaxesBs5'] = [72,72,72,72]
SF['tmaxesBs5T'] = [72,72,72,72]
SF['tmaxesBs5X'] = [72,72,72,72]
SF['tmaxesBsYZ'] = [72,72,72,72]
#SF['tmaxesK'] = [72,72,50,35,20]
#SF['tmaxespi'] = [72,72,40,30,20]
SF['tmaxesK'] = [72,72,45,35,28]
SF['tmaxespi'] = [72,72,32,28,16]
#########################
SF['tmin_2pt'] = 7 + tmin_2pt_adjuster                     #sets tmin for all 2 pts
SF['tmin_3pt'] = 3 + tmin_3pt_adjuster      #sets tmin for all 3 pts
#########################
SF['tminB5'] = SF['tmin_2pt']
SF['tminB5T'] = SF['tmin_2pt']
SF['tminB5X'] = SF['tmin_2pt']
SF['tminBYZ'] = SF['tmin_2pt']
SF['tminBs5'] = SF['tmin_2pt']
SF['tminBs5T'] = SF['tmin_2pt']
SF['tminBs5X'] = SF['tmin_2pt']
SF['tminBsYZ'] = SF['tmin_2pt']
SF['tminK'] = SF['tmin_2pt']
SF['tminpi'] = SF['tmin_2pt']
SF['Stmin'] = SF['tmin_3pt']
SF['Vtmin'] = SF['tmin_3pt']
SF['Ttmin'] = SF['tmin_3pt']
SF['Xtmin'] = SF['tmin_3pt']
SF['Sstmin'] = SF['tmin_3pt']
SF['Vstmin'] = SF['tmin_3pt']
SF['Tstmin'] = SF['tmin_3pt']
SF['Xstmin'] = SF['tmin_3pt']
############################
SF['2pt_amp'] = 0.120
SF['En_Width_Fraction'] = 0.5
######################### 
SF['Mloosener'] = 0.05                        #Loosener on ground state
SF['oMloosener'] = 0.2                       #Loosener on oscillating ground state
SF['Aeff_0_loosener'] = 0.1
SF['m3/m0_width_rat'] = 4
SF['Api_loosener'] = 0.10
SF['AK_loosener'] = 0.08
SF['Aeff_n_loosener'] = 2.6
SF['V10_widener'] = 4.0
SF['Vloosener'] = 1                      #Loosener on Vnn[0][0]
SF['a'] = 0.1715/(2.896*0.1973)
SF['aE_pion_loosner'] = 0.10
SF['aE_kaon_loosner'] = 0.05
############################
SF['an'] = '{}({})'.format(0.15,0.15)
SF['aon'] = '{}({})'.format(0.03, 0.03)   
#SF['SVnn0'] = '2.0({})'.format(2.4) 
#vvv Changes made on 2025/05/29 to try and fix SF Hpi
SF['SVnn0'] = '1.5({})'.format(1.5) 
SF['SVnn0'] = ['2.5(2.5)', '1.75(1.75)', '1.25(1.25)', '1.0(1.0)', '1.0(5.0)'] #special note of big prior at end here
SF['SVn'] = '0.0({})'.format(0.32)                         #Prior for SV_??[n][n]
SF['SV0'] = '0.0({})'.format(1.4)                         #Prior for SV_no[0][0] etc
#SF['VVnn0'] = '1.0({})'.format(4.0) 
#vvv Changes made on 2025/05/29 to try and fix SF Hpi
SF['VVnn0'] = '1.0({})'.format(2.0)  
SF['VVnn0'] = ['2.0(2.0)', '1.0(1.0)', '0.65(0.65)', '0.6(0.6)', '1.0(4.0)']
SF['VVn'] = '0.0({})'.format(0.46)  
SF['VV0'] = '0.0({})'.format(3.0)  
#SF['XVnn0'] = '0.25({})'.format(1.1) 
#vvv Changes made on 2025/05/29 to try and fix SF Hpi
SF['XVnn0'] = '0.25({})'.format(0.40)  
SF['XVnn0'] = ['', '0.3(0.3)', '0.25(0.25)', '0.25(0.25)', '1.0(4.0)']
SF['XVn'] = '0.0({})'.format(0.10) 
SF['XV0'] = '0.0({})'.format(0.23) 
#SF['TVnn0'] = '0.22({})'.format(0.80) 
#vvv Changes made on 2025/05/29 to try and fix SF Hpi
SF['TVnn0'] = '0.22({})'.format(0.35)
SF['TVnn0'] = ['', '0.2(0.2)', '0.2(0.2)', '0.2(0.2)', '0.2(3.0)']
SF['TVn'] = '0.0({})'.format(0.05) 
SF['TV0'] = '0.0({})'.format(0.25) 
#SF['SsVnn0'] = '1.2({})'.format(1.1)                        #Prior for SV_nn[0][0]
#vvv Changes made on 2025/05/29 to try and fix SF HsK
SF['SsVnn0'] = '1.3({})'.format(0.8)
SF['SsVnn0'] = ['2.0(2.0)', '1.5(1.5)', '1.2(1.2)', '1.0(1.0)', '1.0(1.0)']  #Implementing new prior scheme November 2025    
SF['SsVn'] = '0.0({})'.format(0.24)                         #Prior for SV_??[n][n]
SF['SsV0'] = '0.0({})'.format(0.55)                      #Prior for SV_no[0][0] etc
SF['VsVnn0'] = '0.7({})'.format(1.0) 
SF['VsVnn0'] = ['1.0(1.0)', '0.75(0.75)', '0.6(0.6)', '0.6(0.6)', '0.75(0.75)']  #Implementing new prior scheme November 2025   
SF['VsVn'] = '0.0({})'.format(0.26)  
SF['VsV0'] = '0.0({})'.format(0.86)  
#SF['XsVnn0'] = '0.20({})'.format(0.40)  
#vvv Changes made on 2025/05/29 to try and fix SF HsK
SF['XsVnn0'] = '0.20({})'.format(0.25)  
SF['XsVnn0'] = ['', '0.25(0.25)', '0.3(0.3)', '0.3(0.3)', '0.25(0.50)']
SF['XsVn'] = '0.0({})'.format(0.1)  
SF['XsV0'] = '0.0({})'.format(0.25)  
SF['TsVnn0'] = '0.23({})'.format(0.20)  
SF['TsVnn0'] = ['', '0.2(0.2)', '0.2(0.2)', '0.2(0.2)', '0.2(0.2)']
SF['TsVn'] = '0.0({})'.format(0.01)  
#SF['TsV0'] = '0.0({})'.format(0.04)  ###change to make much wider, hopefully help chidof?
SF['TsV0'] = '0.0({})'.format(0.3)
#########################
SF['B5-Tag'] = '2pt_HlG5-G5_m{0}_th0.0'
SF['B5X-Tag'] = '2pt_HlG5-G5X_m{0}_th0.0'
SF['B5T-Tag'] = '2pt_HlG5T-G5T_m{0}_th0.0'
SF['BYZ-Tag'] = '2pt_HlG5T-GYZ_m{0}_th0.0'
SF['Bs5-Tag'] = '2pt_HsG5-G5_m{0}_th0.0'
SF['Bs5X-Tag'] = '2pt_HsG5-G5X_m{0}_th0.0'
SF['Bs5T-Tag'] = '2pt_HsG5T-G5T_m{0}_th0.0'
SF['BsYZ-Tag'] = '2pt_HsG5T-GYZ_m{0}_th0.0'
SF['K-Tag'] = '2pt_kaonG5-G5_th{0}'
SF['pi-Tag'] = '2pt_pionG5-G5_th{0}'
SF['threePtTagS'] = '3pt_scalar_Hlpi_width{0}_m{2}_th{4}'
SF['threePtTagV'] = '3pt_tempVec_Hlpi_width{0}_m{2}_th{4}'
SF['threePtTagX'] = '3pt_spatVec_Hlpi_width{0}_m{2}_th{4}'
SF['threePtTagT'] = '3pt_tensor_Hlpi_width{0}_m{2}_th{4}'  
SF['threePtTagSs'] = '3pt_scalar_HsK_width{0}_m{2}_th{4}'
SF['threePtTagVs'] = '3pt_tempVec_HsK_width{0}_m{2}_th{4}'
SF['threePtTagXs'] = '3pt_spatVec_HsK_width{0}_m{2}_th{4}'
SF['threePtTagTs'] = '3pt_tensor_HsK_width{0}_m{2}_th{4}'  
##########  
SF['svd'] = 0.04
SF['binsize'] = 1 
#################SFp #######################################
SFp = collections.OrderedDict()
SFp['conf'] = 'SFp'
SFp['filename'] = "binned_sfp_all"
SFp['file_location'] = "./gpls/"
SFp['masses'] = ['0.2585','0.5','0.65','0.8']
SFp['twists'] = ['0.0','2.522','4.216','7.94','10.118']
SFp['m_l'] = '0.0008'
SFp['m_s'] = '0.0219'
SFp['Ts'] = [22,25,28,31]
SFp['tp'] = 192
SFp['L'] = 96
SFp['tmaxesB5'] = [72,72,72,72]
SFp['tmaxesB5T'] = [72,72,72,72]
SFp['tmaxesB5X'] = [72,72,72,72]
SFp['tmaxesBYZ'] = [72,72,72,72]
SFp['tmaxesBs5'] = [72,72,72,72]
SFp['tmaxesBs5T'] = [72,72,72,72]
SFp['tmaxesBs5X'] = [72,72,72,72]
SFp['tmaxesBsYZ'] = [72,72,72,72]
#SFp['tmaxesK'] = [72,72,50,35,20]
#SFp['tmaxespi'] = [72,72,40,30,20]
SFp['tmaxesK'] = [72,72,42,22,28]
SFp['tmaxespi'] = [72,46,30,25,16]
#########################
SFp['tmin_2pt'] = 7 + tmin_2pt_adjuster      #sets tmin for all 2 pts
SFp['tmin_3pt'] = 2 + tmin_3pt_adjuster      #sets tmin for all 3 pts
#########################
SFp['tminB5'] = SFp['tmin_2pt']
SFp['tminB5T'] = SFp['tmin_2pt']
SFp['tminB5X'] = SFp['tmin_2pt']
SFp['tminBYZ'] = SFp['tmin_2pt']
SFp['tminBs5'] = SFp['tmin_2pt']
SFp['tminBs5T'] = SFp['tmin_2pt']
SFp['tminBs5X'] = SFp['tmin_2pt']
SFp['tminBsYZ'] = SFp['tmin_2pt']
SFp['tminK'] = SFp['tmin_2pt']
SFp['tminpi'] = SFp['tmin_2pt']
SFp['Stmin'] = SFp['tmin_3pt']
SFp['Vtmin'] = SFp['tmin_3pt']
SFp['Ttmin'] = SFp['tmin_3pt']
SFp['Xtmin'] = SFp['tmin_3pt']
SFp['Sstmin'] = SFp['tmin_3pt']
SFp['Vstmin'] = SFp['tmin_3pt']
SFp['Tstmin'] = SFp['tmin_3pt']
SFp['Xstmin'] = SFp['tmin_3pt']
#########################
SFp['En_Width_Fraction'] = 0.5
######################### 
SFp['Mloosener'] = 0.1                        #Loosener on ground state
SFp['oMloosener'] = 0.2                       #Loosener on oscillating ground state
SFp['Vloosener'] = 1                          #Loosener on Vnn[0][0]
SFp['Api_loosener'] = 0.30
SFp['AK_loosener'] = 0.15
SFp['Aeff_0_loosener'] = 0.65
SFp['m3/m0_width_rat'] = 2
SFp['Aeff_n_loosener'] = 2.0
SFp['V10_widener'] = 5.5
SFp['a'] = 0.1715/(3.0170*0.1973) #from 2207.04765
SFp['aE_pion_loosner'] = 0.60
SFp['aE_kaon_loosner'] = 0.05
#########################
SFp['an'] = '{}({})'.format(0.15,0.15)
SFp['aon'] = '{}({})'.format(0.015,0.015)  
SFp['SVnn0'] = '2.0({})'.format(3.5)                        #Prior for SV_nn[0][0]
SFp['SVnn0'] = ['4.75(4.75)', '1.75(1.75)', '1.5(1.5)', '1.0(1.5)', '1.0(3.0)']
SFp['SVn'] = '0.0({})'.format(0.45)                          #Prior for SV_??[n][n]
SFp['SV0'] = '0.0({})'.format(3.0)                           #Prior for SV_no[0][0] etc
#SFp['VVnn0'] = '1.0({})'.format(3.0)  #updated
#vvv Changes made on 2025/05/29 to try and fix SFp Hpi
SFp['VVnn0'] = '2.0({})'.format(2.0)
SFp['VVnn0'] = ['4.0(4.0)', '1.25(1.25)', '1.0(1.0)', '0.5(1.0)', '0.5(2.0)']
SFp['VVn'] = '0.0({})'.format(0.55)
SFp['VV0'] = '0.0({})'.format(5.0)
#SFp['XVnn0'] = '0.7({})'.format(2.9)
#vvv Changes made on 2025/05/29 to try and fix SFp Hpi
SFp['XVnn0'] = '1.0({})'.format(1.0)
SFp['XVnn0'] = ['', '0.35(0.35)', '0.35(0.35)', '0.3(0.3)', '0.3(1.0)']
SFp['XVn'] = '0.0({})'.format(0.04)
SFp['XV0'] = '0.0({})'.format(0.5)
SFp['TVnn0'] = '0.3({})'.format(3.3) 
SFp['TVnn0'] = ['', '0.2(0.2)', '0.25(0.25)', '0.2(0.2)', '0.2(1.0)']
SFp['TVn'] = '0.0({})'.format(0.04)
SFp['TV0'] = '0.0({})'.format(0.07)
SFp['SsVnn0'] = '1.2({})'.format(1.2)                      #Prior for SV_nn[0][0]
SFp['SsVnn0'] = ['2.0(2.0)', '1.5(1.5)', '1.25(1.25)', '1.0(1.0)', '1.0(1.0)']
SFp['SsVn'] = '0.0({})'.format(0.2)                           #Prior for SV_??[n][n]
SFp['SsV0'] = '0.0({})'.format(0.4)                            #Prior for SV_no[0][0] etc
SFp['VsVnn0'] = '0.8({})'.format(0.8)   
SFp['VsVnn0'] = ['1.25(1.25)', '0.9(0.9)', '0.75(0.75)', '0.5(0.5)', '0.5(1.0)'] 
SFp['VsVn'] = '0.0({})'.format(0.25)  
SFp['VsV0'] = '0.0({})'.format(2.0)  
SFp['XsVnn0'] = '0.20({})'.format(0.30)  
SFp['XsVnn0'] = ['', '0.25(0.25)', '0.3(0.3)', '0.25(0.25)', '0.2(0.4)']
SFp['XsVn'] = '0.0({})'.format(0.07) 
SFp['XsV0'] = '0.0({})'.format(0.14) 
SFp['TsVnn0'] = '0.2({})'.format(0.4)   
SFp['TsVnn0'] = ['', '0.15(0.15)', '0.2(0.2)', '0.2(0.2)', '0.2(0.3)']
SFp['TsVn'] = '0.0({})'.format(0.08) 
SFp['TsV0'] = '0.0({})'.format(0.16) 
#########################
SFp['B5-Tag'] = '2pt_HlG5-G5_m{0}_th0.0'
SFp['B5X-Tag'] = '2pt_HlG5-G5X_m{0}_th0.0'
SFp['B5T-Tag'] = '2pt_HlG5T-G5T_m{0}_th0.0'
SFp['BYZ-Tag'] = '2pt_HlG5T-GYZ_m{0}_th0.0'
SFp['Bs5-Tag'] = '2pt_HsG5-G5_m{0}_th0.0'
SFp['Bs5X-Tag'] = '2pt_HsG5-G5X_m{0}_th0.0'
SFp['Bs5T-Tag'] = '2pt_HsG5T-G5T_m{0}_th0.0'
SFp['BsYZ-Tag'] = '2pt_HsG5T-GYZ_m{0}_th0.0'
SFp['K-Tag'] = '2pt_kaonG5-G5_th{0}'
SFp['pi-Tag'] = '2pt_pionG5-G5_th{0}'
SFp['threePtTagS'] = '3pt_scalar_Hlpi_width{0}_m{2}_th{4}'
SFp['threePtTagV'] = '3pt_tempVec_Hlpi_width{0}_m{2}_th{4}'
SFp['threePtTagX'] = '3pt_spatVec_Hlpi_width{0}_m{2}_th{4}'
SFp['threePtTagT'] = '3pt_tensor_Hlpi_width{0}_m{2}_th{4}'  
SFp['threePtTagSs'] = '3pt_scalar_HsK_width{0}_m{2}_th{4}'
SFp['threePtTagVs'] = '3pt_tempVec_HsK_width{0}_m{2}_th{4}'
SFp['threePtTagXs'] = '3pt_spatVec_HsK_width{0}_m{2}_th{4}'
SFp['threePtTagTs'] = '3pt_tensor_HsK_width{0}_m{2}_th{4}'  
##########  
SFp['svd'] = 0.01
#SFp['svd'] = 0.006
SFp['binsize'] = 1

#################UF #######################################
UF = collections.OrderedDict()
UF['conf'] = 'UF'
UF['filename'] = "binned_uf5_all"
UF['file_location'] = "./gpls/"
UF['masses'] = ['0.194','0.4','0.6','0.8']
UF['twists'] = ['0.0','0.706','1.529','2.235','4.705']
UF['m_l'] = '0.00316'
UF['m_s'] = '0.0165'
UF['Ts'] = [29,34,39,44]
UF['tp'] = 192
UF['L'] = 64
UF['tmaxesB5'] = [96,96,96,96]
UF['tmaxesB5T'] = [96,96,96,96]
UF['tmaxesB5X'] = [96,96,96,96]
UF['tmaxesBYZ'] = [96,96,96,96]
UF['tmaxesBs5'] = [96,96,96,96]
UF['tmaxesBs5T'] = [96,96,96,96]
UF['tmaxesBs5X'] = [96,96,96,96]
UF['tmaxesBsYZ'] = [96,96,96,96]
#UF['tmaxesK'] = [96,96,96,55,22]
#UF['tmaxespi'] = [96,96,70,50,22]
UF['tmaxesK'] = [96,96,70,55,25]
UF['tmaxespi'] = [96,96,60,42,22]
#########################
UF['tmin_2pt'] = 7 + tmin_2pt_adjuster                     #sets tmin for all 2 pts
UF['tmin_3pt'] = 2 + tmin_3pt_adjuster     #sets tmin for all 3 pts
#########################
UF['tminB5'] = UF['tmin_2pt']
UF['tminB5T'] = UF['tmin_2pt']
UF['tminB5X'] = UF['tmin_2pt']
UF['tminBYZ'] = UF['tmin_2pt']
UF['tminBs5'] = UF['tmin_2pt']
UF['tminBs5T'] = UF['tmin_2pt']
UF['tminBs5X'] = UF['tmin_2pt']
UF['tminBsYZ'] = UF['tmin_2pt']
UF['tminK'] = UF['tmin_2pt']
UF['tminpi'] = UF['tmin_2pt']
UF['Stmin'] = UF['tmin_3pt']
UF['Vtmin'] = UF['tmin_3pt']
UF['Ttmin'] = UF['tmin_3pt']
UF['Xtmin'] = UF['tmin_3pt']
UF['Sstmin'] = UF['tmin_3pt']
UF['Vstmin'] = UF['tmin_3pt']
UF['Tstmin'] = UF['tmin_3pt']
UF['Xstmin'] = UF['tmin_3pt']
########################
UF['En_Width_Fraction'] = 0.5
######################### 
UF['Mloosener'] = 0.05                        #Loosener on ground state
UF['oMloosener'] = 0.2                       #Loosener on oscillating ground state
UF['Api_loosener'] = 0.20
UF['AK_loosener'] = 0.15
UF['Aeff_0_loosener'] = 0.20
UF['m3/m0_width_rat'] = 4.0
UF['Aeff_n_loosener'] = 2.0
UF['V10_widener'] = 5.0
UF['Vloosener'] = 1                        #Loosener on Vnn[0][0]
UF['a'] = 0.1715/(3.892*0.1973) #a in units of GeV
UF['aE_pion_loosner'] = 0.25
UF['aE_kaon_loosner'] = 0.10
########################
UF['an'] = '{}({})'.format(0.1,0.1)
UF['aon'] = '{}({})'.format(0.01, 0.01)   
UF['SVnn0'] = '1.6({})'.format(1.6)                         #Prior for SV_nn[0][0]
UF['SVnn0'] = ['2.75(2.75)', '2.25(2.25)', '1.5(1.5)', '1.0(1.0)', '1.0(1.0)'] 
UF['SVn'] = '0.0({})'.format(0.3)                         #Prior for SV_??[n][n]
UF['SV0'] = '0.0({})'.format(0.6)                           #Prior for SV_no[0][0] etc
UF['VVnn0'] = '1.2({})'.format(1.0)
UF['VVnn0'] = ['2.0(2.0)', '1.5(1.5)', '1.0(1.0)', '0.75(0.75)', '1.0(2.0)'] 
UF['VVn'] = '0.0({})'.format(0.32)  
UF['VV0'] = '0.0({})'.format(3.0)  
UF['XVnn0'] = '0.25({})'.format(0.35)
UF['XVnn0'] = ['', '0.25(0.25)', '0.3(0.3)', '0.3(0.3)', '0.3(1.0)']
UF['XVn'] = '0.0({})'.format(0.04)  
UF['XV0'] = '0.0({})'.format(0.08)  
UF['TVnn0'] = '0.20({})'.format(0.4)  
UF['TVnn0'] = ['', '0.2(0.2)', '0.20(0.2)', '0.20(0.2)', '0.2(0.5)']
UF['TVn'] = '0.0({})'.format(0.04)  
UF['TV0'] = '0.0({})'.format(0.07)  
UF['SsVnn0'] = '1.3({})'.format(1.1)                        #Prior for SV_nn[0][0]
UF['SsVnn0'] = ['2.0(2.0)', '1.75(1.75)', '1.5(1.5)', '1.0(1.0)', '1.0(1.0)']
UF['SsVn'] = '0.0({})'.format(0.16)                         #Prior for SV_??[n][n]
UF['SsV0'] = '0.0({})'.format(0.4)                          #Prior for SV_no[0][0] etc
UF['VsVnn0'] = '0.8({})'.format(0.8)  
UF['VsVnn0'] = ['1.25(1.25)', '1.0(1.0)', '0.75(0.75)', '0.6(0.6)', '0.5(1.0)']
UF['VsVn'] = '0.0({})'.format(0.25) 
UF['VsV0'] = '0.0({})'.format(.20) 
UF['XsVnn0'] = '0.20({})'.format(0.20)  
UF['XsVnn0'] = ['', '0.2(0.2)', '0.3(0.3)', '0.3(0.3)', '0.3(1.0)']    
UF['XsVn'] = '0.0({})'.format(0.07)            
UF['XsV0'] = '0.0({})'.format(0.21)              
UF['TsVnn0'] = '0.20({})'.format(0.25)    
UF['TsVnn0'] = ['', '0.15(0.15)', '0.20(0.20)', '0.20(0.20)', '0.20(0.50)']       
UF['TsVn'] = '0.0({})'.format(0.06)
UF['TsV0'] = '0.0({})'.format(0.15)

#########################
UF['B5-Tag'] = '2pt_HlG5-G5_m{0}_th0.0'
UF['B5X-Tag'] = '2pt_HlG5-G5X_m{0}_th0.0'
UF['B5T-Tag'] = '2pt_HlG5T-G5T_m{0}_th0.0'
UF['BYZ-Tag'] = '2pt_HlG5T-GYZ_m{0}_th0.0'
UF['Bs5-Tag'] = '2pt_HsG5-G5_m{0}_th0.0'
UF['Bs5X-Tag'] = '2pt_HsG5-G5X_m{0}_th0.0'
UF['Bs5T-Tag'] = '2pt_HsG5T-G5T_m{0}_th0.0'
UF['BsYZ-Tag'] = '2pt_HsG5T-GYZ_m{0}_th0.0'
UF['K-Tag'] = '2pt_kaonG5-G5_th{0}'
UF['pi-Tag'] = '2pt_pionG5-G5_th{0}'
UF['threePtTagS'] = '3pt_scalar_Hlpi_width{0}_m{2}_th{4}'
UF['threePtTagV'] = '3pt_tempVec_Hlpi_width{0}_m{2}_th{4}'
UF['threePtTagX'] = '3pt_spatVec_Hlpi_width{0}_m{2}_th{4}'
UF['threePtTagT'] = '3pt_tensor_Hlpi_width{0}_m{2}_th{4}'  
UF['threePtTagSs'] = '3pt_scalar_HsK_width{0}_m{2}_th{4}'
UF['threePtTagVs'] = '3pt_tempVec_HsK_width{0}_m{2}_th{4}'
UF['threePtTagXs'] = '3pt_spatVec_HsK_width{0}_m{2}_th{4}'
UF['threePtTagTs'] = '3pt_tensor_HsK_width{0}_m{2}_th{4}'  
##########  
#UF['svd'] = 0.02
UF['svd'] = 0.1
UF['binsize'] = 0
                                                
################ USER INPUTS ################################
#############################################################
# If new main = True, then we use an alternate to the main function developed from 18 July 2025.  
## in new main we fit the re and im half of the data seperately but in the same script.  By doing so, we fit pion/kaon 2pts twice
### To account for this, we do a weighted average of the two copies, which should be correlated. 
### If the chi2/dof of new wavg > 1, the wavg's uncertainty is scaled by sqrt(chi2/dof)
#### Note, new main is set up to always fit all FitCorrs.  Have to manually change it otherwise
##### But, changing twists and masses should still work.
new_main = False
#Ensemble index: [F, Fp, SF, SFp, UF] == 0, 1, 2, 3, 4
## Decay Index : 0 -> H to pi, 1 -> Hs to K
###Pair index = 0 : S + V, = 1 : X + T, only used if fit_by_decay_and_curr == True
Ensemble_Index = 4
Decay_Index = 1
#in altmain = true then there is no need to change pair index here
Pair_Index = 0
#
Use_PVC = False #if true, will use Judd's Partial Variance Cut for SVD diagnosis (at least where I have implemented it)
# Currently PVC is mostly non funcitonal / doesnt improve fits.  Keep False unless PVC is worked on.
#
Fits = [F, Fp, SF, SFp, UF]
Fit = Fits[Ensemble_Index]
Fit['svd'] = 1.0 # = 1 will override smaller manual svd cuts and return to default svd  diagnosis
Fit['special_Fp_pion_n=1_tightener'] = 0.05 #IN case of Hpi, accounts for spurios state for n=1 pion mass
#Fit['special_Fp_VVon_widener'] = 
#
PriorLoosener = 1.0
Nexp = 4  
FitMasses = [0]#,1,2,3]                                # Choose which masses to fit
FitTwists = [0,1]                           # Choose which twists to fit
FitTs = [3]

# Global fit by decay channel options
fit_by_decay_channel_method = False
if fit_by_decay_channel_method == True:
    Bpi_batch = ['B5','B5T','B5X','BYZ', 'pi', 'S', 'V', 'X', 'T']
    BsK_batch = ['Bs5','Bs5T','Bs5X','BsYZ', 'K', 'Xs', 'Ts']
    Decay_Options = [Bpi_batch, BsK_batch]
    FitCorrs = np.array([Decay_Options[Decay_Index]],dtype = object)
    decay_tags = ['Hpi','HsK']
    call_tag = '{}-{}'.format(Fit['conf'], decay_tags[Decay_Index])
else: call_tag = None

fit_by_decay_and_curr = False
if fit_by_decay_and_curr == True:
    Re_Bpi = ['B5T', 'B5', 'pi', 'V', 'S']
    Im_Bpi = ['B5X','BYZ', 'pi', 'X', 'T']
    Re_BsK = ['Bs5','Bs5T', 'K', 'Ss', 'Vs']
    Im_BsK = ['Bs5X','BsYZ', 'K', 'Xs', 'Ts']
    options = [[Re_Bpi, Im_Bpi], [Re_BsK, Im_BsK]]
    FitCorrs = np.array([options[Decay_Index][Pair_Index]],dtype = object)
    decay_tags = ['Hpi','HsK']
    call_tag = '{}-{}'.format(Fit['conf'], decay_tags[Decay_Index])

###GBF testing 3pts
GBF_testing_3pt = False
if GBF_testing_3pt == True:
    mother = [['B5'],['B5T'],['B5X'],['BYZ'], ['Bs5'],['Bs5T'],['Bs5X'],['BsYZ']]
    daughter = [['pi'],['pi'],['pi'],['pi'],['K'],['K'],['K'],['K']]
    current = [['S'],['V'],['X'],['T'],['Ss'],['Vs'],['Xs'],['Ts']]
    #central_value = 0.31
    #Fit['{}Vnn0'.format(current[comp_key][0])] = '{}({})'.format(central_value,Fit['{}width'.format(current[comp_key][0])])
    comp_key = 0
    FitCorrs = np.array([mother[comp_key] + daughter[comp_key] , current[comp_key]] ,dtype=object)

##Solo 2pt testing
Only_2pts = True
if Only_2pts == True:
    B = ['B5','B5T','B5X','BYZ']
    Bs = ['Bs5','Bs5T','Bs5X','BsYZ']
    curr = ['S', 'V', 'X', 'T']
    currs = ['Ss', 'Vs', 'Xs', 'Ts']
    #FitCorrs = np.array([B + Bs + ['pi', 'K']] ,dtype=object)
    if Decay_Index == 0: FitCorrs = np.array([B + ['pi'] + curr] ,dtype=object)  
    #if Decay_Index == 0: FitCorrs = np.array([['B5'] + ['pi'] + ['S']] ,dtype=object)  
    elif Decay_Index == 1: FitCorrs = np.array([Bs + ['K'] + currs],dtype=object) 
    #else: print('Invalid Decay Index, must be 0 or 1, you have chosen {}'.format(Decay_Index))


SaveFit = True
Append_Key_Fit_Stats = False       #Appends csv file 'Key_Fit_Stats.csv', only funcitonal if Savefit = True. mostly used for svd testing
noise = False
SepMass = False                                 #defunct funcitonality, keep false
SvdFactor = Fit['svd']                                     # Multiplies saved SVD
             # Multiplies all prior error by loosener
Tolerance = 1e-6                  # Digits of precision for fit.  Default is 1e-8, only implemented for chained marg fits
                          #Number of exponentials used to model fit, 3 or 4 is typical
Nmarg_non = 4                           #If Marginilisation = true then fit will only consider the bottom Nmarg_non exponentials
Nmarg_osc = 4                           #If Marginilisation = true then fit will only consider the bottom Nmarg_osc exponentials
Chained = False   # If False puts all correlators above in one fit no matter how they are organised
Marginalised = False         #If marginilised = true, will marginilize number of highest exponentials = Nexp - Nmarg
ResultPlots = False         # Tell what to plot against, "Q", "N","Log(GBF)", False
smallsave = False #saves only the ground state non-oscillating and 3pts
log2sqrt = False #'sqrt'  ### For 2pt amp and dE transformation to get only positive
Amp_mother_scaling = True  ### Uses scale to set mother meson amplitude prior uncertainty
####################################################################################################
# This ensures that Amp Scaling isnt being used if not all mass options are in use
if Amp_mother_scaling == True:
    if FitMasses != [0,1,2,3]:
        Amp_mother_scaling = False
# Marginilisation options
Nmarg = (Nmarg_non, Nmarg_osc)
################You shouldn't need to edit below here ##############################################
setup = ['B5-S-pi','Bs5-Ss-K','B5T-V-pi','Bs5T-Vs-K','B5X-X-pi','Bs5X-Xs-K','BYZ-T-pi','BsYZ-Ts-K'] # how the correlators are arranged - DO NOT CHANGE
notwist0 = ['T','Ts','X','Xs'] #list any fits which do not use tw-0
non_oscillating = ['pi'] #any daughters which do no osciallate (only tw 0 is affected)
##############################################################
##############################################################
def main():
    # get which correlators we want
    daughters,currents,parents = read_setup(setup)
    # plots corrs if first time with this data file 
    #plots(Fit,daughters,parents,currents)
    allcorrs,links,parrlinks = elements_in_FitCorrs(FitCorrs)
    # remove masses and twists we don't want to fit
    #print('Making params...')
    make_params(Fit,FitMasses,FitTwists,FitTs,daughters,currents,parents)
    #print('Params made.')
    # make models
    if Chained == True:
        print('Making chained models...')
        models = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,True,allcorrs,links,parrlinks,SepMass)
        print('Chained models made.')
        print('Making data...')
        data = make_pdata('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),models,  Fit['binsize'])
        for key in data:
            print(key, data[key])
        print('Data made.')
    else: 
        models,svdcut = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks,SepMass)
        data = make_pdata('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),models,Fit['binsize'])#process for speed
        #Beginning implementation of Fp Vcurr T=21, 24 fix
        '''if Fit['conf'] == 'Fp':
            new_data = {}
            for key in data:
                if key.startswith('3pt_tempVec_Hlpi_width21') == False:
                    if key.startswith('3pt_tempVec_Hlpi_width24') == False:
                        new_data[key] = data[key]
            data = new_data
            for key in data:
                print(key, data[key])'''
        #End implementation of Fp Vcurr T=21, 24 fix
############################ Do chained fit #########################################################
    if Chained and Marginalised == False:
        prior = make_prior(Fit,Nexp,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating,log2sqrt,Amp_mother_scaling)
        do_chained_fit(data,prior,Nexp,models,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None,Tolerance)
############################ Do chained marginalised fit ##############################################
    elif Chained and Marginalised != False:
        prior = make_prior(Fit,Nexp,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating,log2sqrt,Amp_mother_scaling)
        do_chained_marginalised_fit(data,prior,Nexp,models,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,SaveFit,smallsave,None,Tolerance,Nmarg)
######################### Do unchained fit ############################################################
    else:
        prior = make_prior(Fit,Nexp,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating,log2sqrt,Amp_mother_scaling, FitTwists=FitTwists)
        #if Use_PVC == True:
            #dset = cf.read_dataset('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),binsize=Fit['binsize'])
            #pvc_re_im_fit(dset,prior,Nexp,models,svdcut,Fit,noise,allcorrs)
        #else:
        do_unchained_fit(data,prior,Nexp,models,svdcut,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,SaveFit,smallsave,None,call_tag,Append_Key_Fit_Stats)        
#####################################################################################################
    return()
#####################################################################################################
def alt_main():
    #prereqs
    Re_Bpi = ['B5','B5T','B5X','BYZ', 'pi', 'S', 'V']
    Im_Bpi = ['B5','B5T','B5X','BYZ', 'pi', 'X', 'T']
    Re_BsK = ['Bs5','Bs5T','Bs5X','BsYZ', 'K', 'Ss', 'Vs']
    Im_BsK = ['Bs5','Bs5T','Bs5X','BsYZ', 'K', 'Xs', 'Ts']
    options = [[Re_Bpi, Im_Bpi], [Re_BsK, Im_BsK]]
    #re_half -> pair index = 0
    Pair_Index = 0
    #defiing fitcorrs
    FitCorrs = np.array([options[Decay_Index][Pair_Index]],dtype = object)
    daughters,currents,parents = read_setup(setup)
    allcorrs,links,parrlinks = elements_in_FitCorrs(FitCorrs)
    #print('allcorrs', allcorrs)
    #print('Making params...')
    make_params(Fit,FitMasses,FitTwists,FitTs,daughters,currents,parents) #note, we only need to call this function once in this two part fit
    #print('Params made.')
    # make models
    models,svdcut = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks,SepMass)
    data = make_pdata('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),models,Fit['binsize'])#process for speed
    prior = make_prior(Fit,Nexp,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating,log2sqrt,Amp_mother_scaling, FitTwists=FitTwists)
    re_fit = re_im_fit(data, prior, Nexp, models, svdcut, Fit, noise, allcorrs)

    #im_half -> pair index = 1
    Pair_Index = 1
    #defiing fitcorrs
    FitCorrs = np.array([options[Decay_Index][Pair_Index]],dtype = object)
    daughters,currents,parents = read_setup(setup)
    allcorrs,links,parrlinks = elements_in_FitCorrs(FitCorrs)
    '''print('Making params...')
    make_params(Fit,FitMasses,FitTwists,FitTs,daughters,currents,parents)
    print('Params made.')'''
    # make models
    models,svdcut = make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,SvdFactor,Chained,allcorrs,links,parrlinks,SepMass)
    #print('Models pre check: ', models)
    data = make_pdata('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),models,Fit['binsize'])#process for speed
    prior = make_prior(Fit,Nexp,allcorrs,currents,daughters,parents,PriorLoosener,data,notwist0,non_oscillating,log2sqrt,Amp_mother_scaling, FitTwists=FitTwists)
    im_fit = re_im_fit(data, prior, Nexp, models, svdcut, Fit, noise, allcorrs)

    #Now we combine previous fits, doing a weighted average of the daughter mesons
    combined_fit = gv.BufferDict()
    temp_re, temp_im = gv.BufferDict(), gv.BufferDict()
    temp_dicts = (temp_re, temp_im)
    daughter = ('pion', 'kaon')[Decay_Index]
    tick = 0
    for fitp in (re_fit, im_fit):
        for key in fitp:
            if daughter not in key:
                if key in ('eps_pi2','eps_K2','osc_pi2', 'osc_K2', 'A_pi2', 'A_K2'):
                    temp_dicts[tick][key] = fitp[key]
                else: combined_fit[key] = fitp[key] 
            elif daughter in key:
                temp_dicts[tick][key] = fitp[key] 
            else: print('key error in alt_main() func')
        tick += 1
    #Now we can get the weighted average of each daughter meson fitp and add to combined fit
    ##checking that dict keys and lengths match
    key_tick = 0
    for key in temp_re:
        if key in temp_im: key_tick += 1
        else: print('Key in temp_re not found in temp_im, something has gone wrong')
    #print('{}, {}, {}'.format(key_tick, len(temp_re), len(temp_im)))
    if key_tick == len(temp_re) and key_tick == len(temp_im):
        for key in temp_re:
            if key in ('eps_pi2','eps_K2','osc_pi2', 'osc_K2', 'A_pi2', 'A_K2'):
                wavg = lsqfit.wavg((temp_re[key], temp_im[key]))
                print('Weighted average of {} = {}'.format(key, wavg))
                print('--- wavg.chi2/dof =', round(wavg.chi2/wavg.dof,3))
                if (wavg.chi2/wavg.dof) > 1:
                    S = gv.sqrt(wavg.chi2/wavg.dof)
                    print('--- wavg.chi2/dof > 1.  Scaling by S = sqrt(chi^2/dof) = ', round(S,3))
                    dy = wavg.sdev * gv.sqrt(S**2-1) / wavg.mean
                    scaled_wavg = wavg * gv.gvar(1.0,dy)
                    print('--------- New weighted average of {} = {}'.format(key, scaled_wavg))
                    wavg_n = scaled_wavg
                combined_fit[key] = wavg
            else:
                wavg = []
                log_wavg = []
                new_key = key[4:-1]
                for n in range(Nexp):
                    wavg_n = lsqfit.wavg((gv.exp(temp_re[key][n]), gv.exp(temp_im[key][n])))
                    if n == 0:
                        print('Weighted average of ground state {} = {}'.format(new_key, wavg_n))
                        print('--- wavg.chi2/dof =', round(wavg_n.chi2/wavg_n.dof,3))
                        if (wavg_n.chi2) > 1:
                            S = gv.sqrt(wavg_n.chi2)
                            print('--- wavg.chi2/dof > 1.  Scaling by S = sqrt(chi^2/dof) = ', round(S,3))
                            dy = wavg_n.sdev * gv.sqrt(S**2-1) / wavg_n.mean
                            scaled_wavg = wavg_n * gv.gvar(1.0,dy)
                            print('--------- New weighted average of {} = {}'.format(new_key, scaled_wavg))
                            wavg_n = scaled_wavg
                    wavg.append(wavg_n)
                    log_wavg.append(gv.log(wavg_n))
                wavg = np.array(wavg)
                log_wavg = np.array(log_wavg)
                combined_fit[key] = log_wavg
                combined_fit[new_key] = wavg
            print('-'*70)
            
    #Saving gvdump and txt version of final combined fit
    if SaveFit == True: alt_save_fit(combined_fit, Fit['conf'], Decay_Index)

if new_main == False: main()
else: alt_main()
