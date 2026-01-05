import collections
import gvar as gv
#from z_functions import add_Physical_Masses_in_GeV

#In this file we create the ensemble sepcific dictionaries that are necessary for the modified z-expansion.
#The first chunk of each ensemble's dict items come from the FF_control script
#The second chunk are meson masses
##Would need to update if more ensembles are added in the future
def import_Ensemble_Dictionaries():
    gv_dict = {}
    non_gv_dict = {}

    #F['tag'] = 'F'
    non_gv_dict['F_masses'] = ['0.450','0.55','0.675','0.8']
    #non_gv_dict['F_twists'] = ['0.0','0.4281','1.282','2.1410','2.570']
    non_gv_dict['F_m_l'] = '0.0074'
    non_gv_dict['F_m_s'] = '0.0376'
    non_gv_dict['F_Ts'] = [15,18,21,24]
    non_gv_dict['F_tp'] = 96
    non_gv_dict['F_L'] = 32
    non_gv_dict['F_a'] = 0.1715/(1.9006*0.1973)
    non_gv_dict['F_a/fm'] = 0.09
    non_gv_dict['F_alt'] = 'f5'
    gv_dict['F_w0/a'] = gv.gvar('1.9006(20)')
    non_gv_dict['F_light'] = float(non_gv_dict['F_m_l'])
    non_gv_dict['F_strange_val'] = float(non_gv_dict['F_m_s'])
    non_gv_dict['F_strange_sea'] = 0.037
    non_gv_dict['F_charm_sea'] = 0.440
    non_gv_dict['F_charm_tuned'] = float(non_gv_dict['F_masses'][0]) #set to lowest heavy quark mass option
    non_gv_dict['F_eta_s'] = (gv.gvar('0.314015(89)') / non_gv_dict['F_a']).mean #From Wills Toward accurate form factors for B-to-light meson decay from lattice QCD https://arxiv.org/pdf/2010.07980
    # ^^^ probably needs to be converted to GeV --- now it is!
    gv_dict['F_delta_FV'] = gv.gvar('0.0(1.0)') #TEMP TEMP TEMP


    #Fp['tag'] = 'Fp'
    non_gv_dict['Fp_masses'] = ['0.433','0.555','0.678','0.8']
    #non_gv_dict['Fp_twists'] = ['0.0','0.58','0.87','1.13','3.000','5.311']
    non_gv_dict['Fp_m_l'] = '0.0012'
    non_gv_dict['Fp_m_s'] = '0.036'
    non_gv_dict['Fp_Ts'] = [15,18,21,24]
    non_gv_dict['Fp_tp'] = 96
    non_gv_dict['Fp_L'] = 64
    non_gv_dict['Fp_a'] = 0.1715/(1.9518*0.1973)
    non_gv_dict['Fp_a/fm'] = 0.088
    non_gv_dict['Fp_alt'] = 'fp'
    gv_dict['Fp_w0/a'] = gv.gvar('1.9518(7)')
    non_gv_dict['Fp_light'] = float(non_gv_dict['Fp_m_l'])
    non_gv_dict['Fp_strange_val'] = float(non_gv_dict['Fp_m_s'])
    non_gv_dict['Fp_strange_sea'] = 0.0363
    non_gv_dict['Fp_charm_sea'] = 0.432
    non_gv_dict['Fp_charm_tuned'] = float(non_gv_dict['Fp_masses'][0]) #set to lowest heavy quark mass option
    #IMPORTANT The following is from F5 not Fphys.  Ideally the eta_s mass is not dependent on the light quark mass, but I assume there
    ## is going to be some ofset / corection / uncertainty here.  Chris confirms this is an okay assumption
    non_gv_dict['Fp_eta_s'] = (gv.gvar('0.314015(89)') / non_gv_dict['Fp_a']).mean #From Wills Toward accurate form factors for B-to-light meson decay from lattice QCD https://arxiv.org/pdf/2010.07980
    gv_dict['Fp_delta_FV'] = gv.gvar('0.0(1.0)') #TEMP TEMP TEMP

    #SF['tag'] = 'SF'
    non_gv_dict['SF_masses'] = ['0.274','0.5','0.65','0.8']
    #non_gv_dict['SF_twists'] = ['0.0','1.261','2.108','2.666','5.059']
    non_gv_dict['SF_m_l'] = '0.0048'
    non_gv_dict['SF_m_s'] = '0.0234'
    non_gv_dict['SF_Ts'] = [22,25,28,31]
    non_gv_dict['SF_tp'] = 144
    non_gv_dict['SF_L'] = 48
    non_gv_dict['SF_a'] = 0.1715/(2.896*0.1973)
    non_gv_dict['SF_a/fm'] = 0.059
    non_gv_dict['SF_alt'] = 'sf5'
    gv_dict['SF_w0/a'] = gv.gvar('2.896(6)')
    non_gv_dict['SF_light'] = float(non_gv_dict['SF_m_l'])
    non_gv_dict['SF_strange_val'] = float(non_gv_dict['SF_m_s'])
    non_gv_dict['SF_strange_sea'] = 0.024
    non_gv_dict['SF_charm_sea'] = 0.286
    non_gv_dict['SF_charm_tuned'] = float(non_gv_dict['SF_masses'][0]) #set to lowest heavy quark mass option
    non_gv_dict['SF_eta_s'] = (gv.gvar('0.207020(84)') / non_gv_dict['SF_a']).mean #From Wills Toward accurate form factors for B-to-light meson decay from lattice QCD https://arxiv.org/pdf/2010.07980
    #^^^ Meson mass now in GeV
    gv_dict['SF_delta_FV'] = gv.gvar('0.0(1.0)') #TEMP TEMP TEMP


    #SFp['tag'] = 'SFp'
    non_gv_dict['SFp_masses'] = ['0.2585','0.5','0.65','0.8']
    #non_gv_dict['SFp_twists'] = ['0.0','2.522','4.216','7.94','10.118']
    non_gv_dict['SFp_m_l'] = '0.0008'
    non_gv_dict['SFp_m_s'] = '0.0219'
    non_gv_dict['SFp_Ts'] = [22,25,28,31]
    non_gv_dict['SFp_tp'] = 192
    non_gv_dict['SFp_L'] = 96
    non_gv_dict['SFp_a'] = 0.1715/(3.0170*0.1973) #from 2207.04765
    non_gv_dict['SFp_a/fm'] = 0.06
    non_gv_dict['SFp_alt'] = 'sfp'
    gv_dict['SFp_w0/a'] = gv.gvar('3.0170(23)')
    non_gv_dict['SFp_light'] = float(non_gv_dict['SFp_m_l'])
    non_gv_dict['SFp_strange_val'] = float(non_gv_dict['SFp_m_s'])
    non_gv_dict['SFp_strange_sea'] = 0.022
    non_gv_dict['SFp_charm_sea'] = 0.260
    non_gv_dict['SFp_charm_tuned'] = float(non_gv_dict['SFp_masses'][0]) #set to lowest heavy quark mass option
    #IMPORTANT The following is from SF5 not SFphys.  Ideally the eta_s mass is not dependent on the light quark mass, but I assume there
    ## is going to be some ofset / corection / uncertainty here.  Chris confirmed this is okay asusmption
    non_gv_dict['SFp_eta_s'] = (gv.gvar('0.207020(84)') / non_gv_dict['SFp_a']).mean #From Wills Toward accurate form factors for B-to-light meson decay from lattice QCD https://arxiv.org/pdf/2010.07980
    gv_dict['SFp_delta_FV'] = gv.gvar('0.0(1.0)') #TEMP TEMP TEMP


    #UF['tag'] = 'UF'
    non_gv_dict['UF_masses'] = ['0.194','0.4','0.6','0.8']
    #non_gv_dict['UF_twists'] = ['0.0','0.706','1.529','2.235','4.705']
    non_gv_dict['UF_m_l'] = '0.00316'
    non_gv_dict['UF_m_s'] = '0.0165'
    non_gv_dict['UF_Ts'] = [29,34,39,44]
    non_gv_dict['UF_tp'] = 192
    non_gv_dict['UF_L'] = 64
    non_gv_dict['UF_a'] = 0.1715/(3.892*0.1973)
    non_gv_dict['UF_a/fm'] = 0.044
    non_gv_dict['UF_alt'] = 'uf5'
    gv_dict['UF_w0/a'] = gv.gvar('3.892(12)')
    non_gv_dict['UF_light'] = float(non_gv_dict['UF_m_l'])
    non_gv_dict['UF_strange_val'] = float(non_gv_dict['UF_m_s'])
    non_gv_dict['UF_strange_sea'] = 0.0158 
    non_gv_dict['UF_charm_sea'] = 0.188
    non_gv_dict['UF_charm_tuned'] = float(non_gv_dict['UF_masses'][0]) #set to lowest heavy quark mass option
    non_gv_dict['UF_eta_s'] = (gv.gvar('0.15407(17)') / non_gv_dict['UF_a']).mean  #From Wills Toward accurate form factors for B-to-light meson decay from lattice QCD https://arxiv.org/pdf/2010.07980
    gv_dict['UF_delta_FV'] = gv.gvar('0.0(1.0)') #TEMP TEMP TEMP
    ##############################################################################################

    non_gv_dict['eta_s_phys'] = 0.6885 #GeV

    return gv_dict, non_gv_dict

#import_Ensemble_Dictionaries()