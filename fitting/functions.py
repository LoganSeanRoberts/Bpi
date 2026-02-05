import gvar as gv
import corrfitter as cf
import numpy as np
import csv
import collections
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
matplotlib.use('Agg')
plt.rc("font",**{"size":18})
import datetime
import os
import pickle
import copy
#from plotting import *
import lsqfit
#import pvc_fitter as vc
lsqfit.nonlinear_fit.set(fitter='gsl_multifit',alg='subspace2D',scaler='more',solver='cholesky')#,solver='cholesky')
#lsqfit.nonlinear_fit.set(fitter='gsl_multifit',alg='lm',scaler='more',solver='cholesky') #'lm' here is the Levenberg-Marquardt algo.  Maybe get better stats?
#lsqfit.nonlinear_fit.set(fitter='gsl_multifit',alg='subspace2D',scaler='more',solver='qr')
#################################### 
maxiter=5000
######################################################################################################

def read_setup(setup):
    #Reads in setups, and strips out currents, parents and daughters, as well as which is which
    daughters = []
    currents = []
    parents = []
    for element in setup:
        lab = element.split('-')
        daughters.append(lab[2])
        currents.append(lab[1])
        parents.append(lab[0])
    return(daughters,currents,parents)
######################################################################################################

def strip_list(l): #Strips elemenst from list l
    stripped = ''
    for element in l:
        stripped = '{0}{1}'.format(stripped,element)
    return(stripped)

######################################################################################################


def make_params(Fit,FitMasses,FitTwists,FitTs,daughters,currents,parents):
    #print('make_params Fit[Ts]', Fit['Ts'])
    #Removes things we do not want to fit, specified by FitMasses, FitTwists, FitTs assumes parents have varing mass and daughters varing twist
    j = 0
    for i in range(len(Fit['masses'])):
        if i not in FitMasses:
            del Fit['masses'][i-j]
            for element in set(parents):
                del Fit['tmaxes{0}'.format(element)][i-j]
            j += 1
    j = 0
    for i in range(len(Fit['twists'])):
        if i not in FitTwists:
            del Fit['twists'][i-j]
            for element in set(daughters):
                del Fit['tmaxes{0}'.format(element)][i-j]
            j += 1
    j = 0
    for i in range(len(Fit['Ts'])):
        if i not in FitTs:
            del Fit['Ts'][i-j]
            j += 1
    return()

#######################################################################################################

def make_data(filename,binsize):
    # Reads in filename.gpl, checks all keys have same configuration numbers, returns averaged data
    print('Reading data, binsize = ', binsize) 
    dset = cf.read_dataset(filename,binsize=binsize)
    print('dset read...')
    #for k in dset:
    #    print(k, np.shape(dset[k]), np.min(dset[k]), np.max(dset[k]), dset[k][:1])
    sizes = []
    for key in dset:
        #print(key,np.shape(dset[key]))
        sizes.append(np.shape(dset[key]))
    if len(set(sizes)) != 1:
        print('Not all elements of gpl the same size')
        for key in dset:
            print(key,np.shape(dset[key]))
    else:
        print(filename, 'Size',set(sizes))
        print('Averaging dataset...')
        avg_dset = gv.dataset.avg_data(dset)
        print('Dataset averaged.')
    return(avg_dset)

######################################################################################################

def make_pdata(filename,models,binsize):
    # Reads in filename.gpl, checks all keys have same configuration numbers, returns averaged data 
    print('Reading processed data, binsize = ', binsize)
    dset = cf.read_dataset(filename,binsize=binsize)
    sizes = []
    for key in dset:
        #print(key,np.shape(dset[key]))
        sizes.append(np.shape(dset[key]))
    if len(set(sizes)) != 1:
        print('Not all elements of gpl the same size')
        for key in dset:
            print(key,np.shape(dset[key]))
    else:
        print(filename, 'Size',set(sizes))
    return(cf.process_dataset(dset, models))

#######################################################################################################

def effective_mass_calc(tag,correlator,tp):
    #finds the effective mass and amplitude of a two point correlator
    M_effs = []
    for t in range(2,len(correlator)-2):
        thing  = (correlator[t-2] + correlator[t+2])/(2*correlator[t]) 
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
    return(M_eff)

######################################################################################################

def effective_amplitude_calc(tag,correlator,tp,M_eff,Fit,corr):
    #finds the effective mass and amplitude of a two point correlator
    tmin = Fit['tmin{0}'.format(corr)]
    A_effs = []
    if len(correlator) == tp:
        tmin = 0
    for t in range(tmin,tmin+len(correlator)):
        numerator = correlator[t-tmin]
        if numerator >= 0:
            A_effs.append( gv.sqrt(numerator/(gv.exp(-M_eff*t)+gv.exp(-M_eff*(tp-t)))))
    rav = []
    for i in range(len(A_effs)-4):
        rav.append((A_effs[i] + A_effs[i+1] + A_effs[i+2] + A_effs[i+3])/4)
    A_eff = rav[0]
    diff = abs((rav[1] - rav[0]).mean)
    for i in range(1,len(rav)-1):
        if abs((rav[i+1]-rav[i]).mean) < diff:
            diff = abs((rav[i+1]-rav[i]).mean)
            A_eff = (rav[i] + rav[i+1])/2
    an = gv.gvar(Fit['an'])
    if A_eff.sdev/A_eff.mean > 2:
        print('Replaced A_eff for {0} {1} -> {2}'.format(tag,A_eff,an))
        A_eff = an
    #return(na) #AUTO USE GIVEN PRIORS, 
    return(A_eff)
    #has been recently changed so this only aplies to zero twist
########################################################################################
########################################################################################

def effective_V_calc(corr,daughter,parent,correlators,dcorr,pcorr,Fit,mass,twist,dA_eff,pA_eff,dM_eff,pM_eff):
    #finds the effective V_nn[0][0]
    tp = Fit['tp']
    dtmin = Fit['tmin{0}'.format(daughter)]
    ptmin = Fit['tmin{0}'.format(parent)]
    Vtmin = Fit['{0}tmin'.format(corr)]
    dcorr2 = []
    pcorr2 = []
    Vs = []
    Vs2 = []
    #print(corr,daughter,parent,mass,twist)
    if len(dcorr) == int(tp):
        dcorr2 = dcorr
    else:
        for i in range(dtmin):
            dcorr2.append(0)
        dcorr2.extend(dcorr)
        for i in range(int(tp/2)-len(dcorr2)+1):
            dcorr2.append(0)
    #print(dcorr2)
    if len(pcorr) == int(tp):
        pcorr2 = pcorr
    else:
        for i in range(ptmin):
            pcorr2.append(0)
        pcorr2.extend(pcorr)
        for i in range(int(tp/2)-len(pcorr2)+1):
            pcorr2.append(0)
    #print(pcorr2)
    
    #print(Vcorr2)
    #for ind, T in enumerate([Fit['Ts'][-1]]):#change made here adding [~~~~~[-1]]
    #for ind, T in enumerate([Fit['Ts'][-1]]*4): ### change made here, using ts[-1] 4 times
    for ind, T in enumerate(Fit['Ts']):
        Vcorr2 = []
        V_effs = []
        V_effs2 = []
        correlator = correlators[ind]
        if len(correlator) == int(tp):
            Vcorr2 = correlator
        else:
            for i in range(Vtmin):
                Vcorr2.append(0)
            Vcorr2.extend(correlator)
            for i in range(T-len(Vcorr2)+1):
                Vcorr2.append(0)
        for t in range(T):
            numerator = Vcorr2[t]*pA_eff*dA_eff
            numerator2 = Vcorr2[t]/(pA_eff*dA_eff)
            denominator = dcorr2[T-t]*pcorr2[t] # modified as now other way around
            denominator2 = gv.exp(-pM_eff*t)*gv.exp(-dM_eff*(T-t)) 
            if numerator != 0 and denominator !=0:
                V_effs.append(numerator/denominator)
                V_effs2.append(numerator2/denominator2)
        rav = []
        for i in range(len(V_effs)-4):
            rav.append((V_effs[i] + V_effs[i+1] + V_effs[i+2] + V_effs[i+3])/4)
        V_eff = rav[0]
        diff = abs((rav[1] - rav[0]).mean)
        for i in range(1,len(rav)-1):
            if abs((rav[i+1]-rav[i]).mean) < diff:
                diff = abs((rav[i+1]-rav[i]).mean)
                if (rav[i] + rav[i+1]) > 0:
                    V_eff = (rav[i] + rav[i+1])/2
        Vs.append(V_eff)
        rav2 = []
        for i in range(len(V_effs2)-4):
            rav2.append((V_effs2[i] + V_effs2[i+1] + V_effs2[i+2] + V_effs2[i+3])/4)
        V_eff2 = rav2[0]
        diff2 = abs((rav2[1] - rav2[0]).mean)
        for i in range(1,len(rav2)-1):
            if abs((rav2[i+1]-rav2[i]).mean) < diff2:
                diff2 = abs((rav2[i+1]-rav2[i]).mean)
                if (rav2[i] + rav2[i+1]) > 0:
                    V_eff2 = (rav2[i] + rav2[i+1])/2
        Vs2.append(V_eff2)
    V_eff =  sum(Vs)/len(Vs) ### V_eff and V_eff2 swapped (untrue here)
    V_eff2 =  sum(Vs2)/len(Vs2) ############################
    V = gv.gvar(Fit['{0}Vnn0'.format(corr)])
    if abs((V_eff.mean-V).mean/(V_eff.mean-V).sdev) > 1 or V_eff.mean < 0:
        if abs((V_eff2.mean-V).mean/(V_eff2.mean-V).sdev) > 1 or V_eff2.mean < 0:
            print('Replaced V_eff for {0} m {1} tw {2}: {3} --> {4}'.format(corr,mass,twist,V_eff,V))
            V_eff = V
        else:
            V_eff = V_eff2
    return(V) #AUTO USE GIVEN PRIORS, LOGAN CHANGED IT HERE
    #return(V_eff)  #For calculate priors, return V_eff

#######################################################################################################

def SVD_diagnosis(Fit,models,corrs,svdfac,currents,SepMass):
    binsize = Fit['binsize']
    #Feed models and corrs (list of corrs in this SVD cut)
    if list(set(corrs).intersection(currents)) ==[]:
        filename = 'SVD/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(corrs),binsize,SepMass)
    else:
        filename = 'SVD/{0}{1}{2}{3}{4}{5}{6}{7}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(corrs),strip_list(Fit['Ts']),binsize,SepMass)
    #print(filename)
    for corr in corrs:
       if 'tmin{0}'.format(corr) in Fit:
           filename += '{0}'.format(Fit['tmin{0}'.format(corr)])
           for element in Fit['tmaxes{0}'.format(corr)]:
               filename += '{0}'.format(element)
       if '{0}tmin'.format(corr) in Fit:
           filename += '{0}'.format(Fit['{0}tmin'.format(corr)])
 
    #print(filename)
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        pickle_off = open(filename,"rb")
        svd = pickle.load(pickle_off)
        print('Loaded SVD for {0} : {1:.2g} x {2} = {3:.2g}'.format(corrs,svd,svdfac,svd*svdfac))
        pickle_off.close()
    else:
        print('Calculating SVD for {0}'.format(corrs))
        s = gv.dataset.svd_diagnosis(cf.read_dataset('{0}{1}.gpl'.format(Fit['file_location'],Fit['filename']),binsize=binsize), models=models, nbstrap=20)
        svd = s.svdcut
        ######## save plot ##########################
        '''plt.figure()
        x = s.val / s.val[-1]
        ratio = s.bsval / s.val
        idx = x > s.mincut
        ratio = ratio[idx]
        x = x[idx]
        y = gv.mean(ratio)
        yerr = gv.sdev(ratio)
        plt.errorbar(x=x, y=y, yerr=yerr, fmt='+', color='b')
        sig = (2. / len(s.val)) ** 0.5
        plt.plot([x[0], x[-1]], [1. - sig, 1. - sig], 'k:')
        plt.axhline(1,ls='--',color='k')
        plt.axvline(s.svdcut,ls=':',color='g')
        #plt.axvline(0.013,ls='--',color='g')
        plt.xscale('log')
        #plt.savefig('svd_plots/{0}.pdf'.format(filename.split('/')[1]))'''
        ###############################################
        pickle_on = open(filename,"wb")
        print('Calculated SVD for {0} : {1:.2g} x {2} = {3:.2g}'.format(corrs,svd,svdfac,svd*svdfac))
        pickle.dump(svd,pickle_on)
    return(svd*svdfac)

#######################################################################################################

def make_models(Fit,FitCorrs,notwist0,non_oscillating,daughters,currents,parents,svdfac,Chained,allcorrs,links,parrlinks,SepMass,NoSVD=False):
    #several forms [(A,B,C,D)],[(A,B),(C),(D)],[(A,B),[(C),(D)]]
    #First make all models and then stick them into the correct chain
    #print('Ts:   ', Fit['Ts'])
    models = collections.OrderedDict()
    tp = Fit['tp']
    for corr in set(parents):
        if corr in allcorrs:
            models['{0}'.format(corr)] = []
            for i,mass in enumerate(Fit['masses']):
                tag = Fit['{0}-Tag'.format(corr)].format(mass)
                models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), b=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), dE=('dE:{0}'.format(tag), 'dE:o{0}'.format(tag)),s=(1,-1)))

    for corr in set(daughters):
        if corr in allcorrs:
            models['{0}'.format(corr)] = []
            for i,twist in enumerate(Fit['twists']):
                tag = Fit['{0}-Tag'.format(corr)].format(twist)
                if twist == '0.0' and corr in notwist0:
                    pass
                elif twist == '0.0' and corr in non_oscillating:
                    models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag)), b=('{0}:a'.format(tag)), dE=('dE:{0}'.format(tag))))
                else:
                    models['{0}'.format(corr)].append(cf.Corr2(datatag=tag, tp=tp, tmin=Fit['tmin{0}'.format(corr)], tmax=Fit['tmaxes{0}'.format(corr)][i], a=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), b=('{0}:a'.format(tag), 'o{0}:a'.format(tag)), dE=('dE:{0}'.format(tag), 'dE:o{0}'.format(tag)),s=(1,-1)))

    for i,corr in enumerate(currents):
        if corr in allcorrs:
            models['{0}'.format(corr)] = []
            for  mass in Fit['masses']:
                for twist in Fit['twists']:
                    #Begin implimentation of possible Fp fix for Vcurr widths = 21, 24
                    '''if Fit['conf'] == 'Fp':
                        if corr == 'V':
                            Fit['Ts'] = [15,18]'''
                    #End implimentation
                    #print('Ts', Fit['Ts'])
                    for T in Fit['Ts']:
                        #print('T', T, 'in Ts',  Fit['Ts'])
                        tag = Fit['threePtTag{0}'.format(corr)].format(T,Fit['m_s'],mass,Fit['m_l'],twist)
                        ptag = Fit['{0}-Tag'.format(parents[i])].format(mass)
                        dtag = Fit['{0}-Tag'.format(daughters[i])].format(twist)
                        #if corr == 'X' and mass not in Xdatam:
                        #    pass
                        #elif corr == 'X' and twist not in Xdatatw:
                        #    pass
                        if twist == '0.0' and corr in notwist0:
                            #print('notwist0', corr)
                            pass
                        elif twist == '0.0' and daughters[i] in non_oscillating:
                            #print('non_oscillating', corr)
                            models['{0}'.format(corr)].append(cf.Corr3(datatag=tag, T=T, tmin=Fit['{0}tmin'.format(corr)], a=('{0}:a'.format(ptag), 'o{0}:a'.format(ptag)), dEa=('dE:{0}'.format(ptag), 'dE:o{0}'.format(ptag)), sa=(1,-1), b=('{0}:a'.format(dtag)), dEb=('dE:{0}'.format(dtag)),Vnn='{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist), Von='{0}Von_m{1}_tw{2}'.format(corr,mass,twist)))
                        else:
                            #print('else', corr)
                            models['{0}'.format(corr)].append(cf.Corr3(datatag=tag, T=T, tmin=Fit['{0}tmin'.format(corr)], a=('{0}:a'.format(ptag), 'o{0}:a'.format(ptag)), dEa=('dE:{0}'.format(ptag), 'dE:o{0}'.format(ptag)), sa=(1,-1), b=('{0}:a'.format(dtag), 'o{0}:a'.format(dtag)), dEb=('dE:{0}'.format(dtag), 'dE:o{0}'.format(dtag)), sb=(1,-1), Vnn='{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist), Vno='{0}Vno_m{1}_tw{2}'.format(corr,mass,twist),Von='{0}Von_m{1}_tw{2}'.format(corr,mass,twist),Voo='{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)))
    
    #Now we make these models into our chain calculating an svd cut for each. We make them in two halves so we can sndwich a marginalisation term if we like later        
    if Chained:
        print('if chained...')
        models_2pts = []
        models_3pts_re = []
        models_3pts_im = []
        for key in links:
            #print(links)
            #finalmodelsB = []
            link = [] #link is models in link
            for corr in links[key]:
                link.extend(models['{0}'.format(corr)]) 
            svd = SVD_diagnosis(Fit,link,links[key],svdfac,currents,SepMass)
            models_2pts.append({'svdcut':svd})
            models_2pts.append(tuple(link))
        return (tuple(models_2pts))
        #return(finalmodels) #troubleshooting here, parallel fits arent working :^(
    else:
        finalmodels = []
        for corr in allcorrs:
            finalmodels.extend(models['{0}'.format(corr)])
        if NoSVD == False:
            svd = SVD_diagnosis(Fit,finalmodels,allcorrs,svdfac,currents,SepMass)
            return(tuple(finalmodels),svd)                
        else:
            return(tuple(finalmodels))
    

#######################################################################################################

def elements_in_FitCorrs(a):
    # reads [A,[B,C],[[D,E],F]] and interprets which elements will be chained and how. Returns alphabetical list of all elements, links in chain and links in parallell chain
    allcorrs = []
    links = collections.OrderedDict()
    parrlinks = collections.OrderedDict()
    for i in range(np.shape(a)[0]):
        links[i] =[]
        if len(np.shape(a[i])) == 0: #deals with one corr in chain 
            #print(a[i],i,'fit alone in chain')
            allcorrs.append(a[i])
            links[i].append(a[i])
        elif len(np.shape(a[i][0])) == 0 : #deals with multiple elements in chain 
            for j in range(len(a[i])):
                #print(a[i][j],i,'fit together in chain')
                allcorrs.append(a[i][j])
                links[i].append(a[i][j])
        else:
            del links[i]  #don't need thi key if it is in paralell
            for j in range(np.shape(a[i])[0]):
                parrlinks[j] = []
                if len(np.shape(a[i][j])) == 0: #deals with one corr in parr chain 
                    allcorrs.append(a[i][j])
                    parrlinks[j].append(a[i][j])
                else:                           # deals with multiple elements in parralell chain
                    for k in range(len(a[i][j])):
                        allcorrs.append(a[i][j][k])
                        parrlinks[j].append(a[i][j][k])
    print('allcorrs', sorted(allcorrs))
    #print('links', links)

    ##This is very scuffed, but I think it works to separate into two 3pt halves
   # print('parrlinks ', parrlinks)
    
    '''z=0
    if len(parrlinks) != 0:
        parrlinks_3pt_re = collections.OrderedDict()
        parrlinks_3pt_re[z] = []
        for x in range(len(parrlinks[0])):
            parrlinks_3pt_re[z].append(parrlinks[0][x])

        parrlinks_3pt_im = collections.OrderedDict()
        parrlinks_3pt_im[z] = []
        for x in range(len(parrlinks[1])):
            parrlinks_3pt_im[z].append(parrlinks[1][x])


        parrlinks = [parrlinks_3pt_re, parrlinks_3pt_im]
        
        print('parrlinks 0', parrlinks[0])
        print('parrlinks 1', parrlinks[1])
        print('new parrlinks,', parrlinks)'''
    
    #return(sorted(allcorrs),links,parrlinks_3pt_re, parrlinks_3pt_im)
    return(sorted(allcorrs),links,parrlinks)


######################################################################################################
# The non trunctated version of the dispersion relation
def full_Edispersion_relation(aM_daughter, twist, N_x):
    aE = gv.arccosh(1 + 0.5*(aM_daughter**2) + 3*(1 - gv.cos(float(twist)*np.pi/N_x)))
    return aE
def full_Adispersion_relation(aM_daughter, aE_daughter, aAmp_tw0):
    aAmp = aAmp_tw0 * gv.sqrt(aM_daughter/aE_daughter)
    return aAmp
######################################################################################################

def make_prior(Fit,N,allcorrs,currents,daughters,parents,loosener,data,notwist0,non_oscillating,log2sqrt,Amp_mother_scaling, FitTwists = None):
    No = N  # number of oscillating exponentials
    if log2sqrt == True:
        def gv_func(x):
            return gv.sqrt(x)
        func = 'sqrt'
    else:
        def gv_func(x):
            return gv.log(x)
        func = 'log'
    #special_prior_widener = Fit['Special_Prior_Widener']
    #scalar_fraction = Fit['Scalar_Fraction']
    prior =  gv.BufferDict()
    tw_corr = True
    otw_corr = True
    #d2_pri = gv.gvar('0.0(1.0)')
    #c2_pri = gv.gvar('0.0(1.0)')
    #oc2_pri = gv.gvar('0.0(1.0)')
    #eps_pi2, eps_K2, eps_pi4, eps_K4 = gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)')
    #A_pi2, A_K2, A_pi4, A_K4 = gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)')
    #osc_eps = gv.gvar('0.0(1.0)')

    #The following bit of code detects if there are pions and or kaons in the fit, if so, it adds the appropriate dispersion variables
    pi, K = 0,0
    for corr in allcorrs:
        if pi == 0:
            if 'pi' in corr: 
                eps_pi2, A_pi2, osc_pi2 = gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)')
                if Fit['conf'] == 'Fp': eps_pi2 = gv.gvar('0.0(36.0)')
                prior['eps_pi2'], prior['A_pi2'], prior['osc_pi2'] = eps_pi2, A_pi2, osc_pi2
                pi = 1 
        if K == 0:
            if 'K' in corr: 
                eps_K2, A_K2, osc_K2 = gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)'), gv.gvar('0.0(1.0)')
                if Fit['conf'] == 'Fp': eps_K2 = gv.gvar('0.0(50.0)')
                if Fit['conf'] == 'SF': eps_K2 = gv.gvar('0.0(5.0)') #1.0
                if Fit['conf'] == 'SFp': eps_K2 = gv.gvar('0.0(3.0)')
                prior['eps_K2'], prior['A_K2'], prior['osc_K2'] = eps_K2, A_K2, osc_K2
                K = 1

    '''if len(daughters) != 0 and '0.0' in Fit['twists'] and tw_corr:
        for corr in set(daughters).intersection(allcorrs):
            #prior['d2_{0}'.format(corr)] = gv.gvar('0.0(1.0)')
            #prior['c2_{0}'.format(corr)] = gv.gvar('0.0(1.0)')
            #prior['d2'] = d2_pri
            #prior['c2'] = c2_pri
            #prior['eps_pi2'], prior['eps_K2'], prior['eps_pi4'], prior['eps_K4'] = eps_pi2, eps_K2, eps_pi4, eps_K4
            #prior['A_pi2'], prior['A_K2'], prior['A_pi4'], prior['A_K4'] = A_pi2, A_K2, A_pi4, A_K4
            prior['eps_pi2'], prior['eps_K2'] = eps_pi2, eps_K2
            prior['A_pi2'], prior['A_K2'] = A_pi2, A_K2
        print('Daughter twists correlated')
    if len(daughters) != 0 and '0.0' in Fit['twists'] and otw_corr:
        for corr in set(daughters).intersection(allcorrs):
            #prior['oc2_{0}'.format(corr)] = gv.gvar('0.0(1.0)')
            #prior['oc2'] = oc2_pri
            prior['osc_eps'] = osc_eps
        print('Daughter oscillating twists correlated')'''
    
    tp = Fit['tp']
    #En = '{0}({1})'.format(0.5*Fit['a'],0.25*Fit['a']*loosener) #Lambda with error of half
    Lambda_QCD = 0.5*Fit['a'] 
    En = '{0}({1})'.format(Lambda_QCD, loosener * 0.5 * Lambda_QCD) #Lambda with error of half 
    En_gvar = gv.gvar(En)
    #an = '{0}({1})'.format(gv.gvar(Fit['an']).mean,gv.gvar(Fit['an']).sdev*loosener)
    #aon = '{0}({1})'.format(gv.gvar(Fit['aon']).mean,gv.gvar(Fit['aon']).sdev*loosener)

    if Amp_mother_scaling == True:
        Max_Ratio = Fit['m3/m0_width_rat']
        U_var = Fit['Aeff_0_loosener']
        a_var = U_var * (Max_Ratio - 1) / (float(Fit['masses'][-1])/float(Fit['masses'][0])-1)
        b_var = U_var - a_var
        m0 = float(Fit['masses'][0])

    for corr in allcorrs:
        if corr in parents:
            for mass in Fit['masses']:
                #Mother meson ampl scaling implemntation
                if Amp_mother_scaling == True:
                    a0_scaler = float(mass)/m0 * a_var + b_var
                else: a0_scaler = Fit['Aeff_0_loosener']

                tag = Fit['{0}-Tag'.format(corr)].format(mass)
                M_eff = effective_mass_calc(tag,data[tag],tp)
                a_eff = effective_amplitude_calc(tag,data[tag],tp,M_eff,Fit,corr)
                a0 = gv.gvar(a_eff.mean,loosener*a0_scaler*a_eff.mean)
                an = gv.gvar(a_eff.mean,loosener*Fit['Aeff_n_loosener']*a_eff.mean)
                # Parent
                prior[func+'({0}:a)'.format(tag)] = gv_func(gv.gvar(N * [an]))
                prior[func+'(dE:{0})'.format(tag)] = gv_func(gv.gvar(N * [En]))
                prior[func+'({0}:a)'.format(tag)][0] = gv_func(a0) 
                prior[func+'(dE:{0})'.format(tag)][0] = gv_func(gv.gvar(M_eff.mean,loosener*Fit['Mloosener']*M_eff.mean))
                # Parent -- oscillating part
                prior[func+'(o{0}:a)'.format(tag)] = gv_func(gv.gvar(No * [an]))
                prior[func+'(dE:o{0})'.format(tag)] = gv_func(gv.gvar(No * [En]))
                prior[func+'(dE:o{0})'.format(tag)][0] = gv_func(gv.gvar((M_eff+gv.gvar(En)*(4/5)).mean,loosener*Fit['oMloosener']*((M_eff+gv.gvar(En)*(4/5)).mean)))
                
                
        if corr in daughters:
            #The numbers 2.05 and 1.85 come from pdg mass splitting predicitons between n=0 and n=1 state
            #splittings are scaled in units of lambda QCD = En_gvar.mean = 0.5 in lattice units
            if 'pi' in corr: dE_n_eq_1 = '{0}({1})'.format(2*En_gvar.mean, En_gvar.sdev) #2.05
            if 'K' in corr: dE_n_eq_1 = '{0}({1})'.format(2*En_gvar.mean, En_gvar.sdev) #1.85
            for twist in Fit['twists']:
                if twist =='0.0' and corr in notwist0:
                    pass
                else:
                    ap2 = 3*(np.pi*float(twist)/Fit['L'])**2
                    #print(twist,ap2)
                    tag0 = Fit['{0}-Tag'.format(corr)].format('0.0')
                    M_eff = np.sqrt(effective_mass_calc(tag0,data[tag0],tp)**2 +  ap2)   #from dispersion relation
                    tag = Fit['{0}-Tag'.format(corr)].format(twist)
                    a_eff = effective_amplitude_calc(tag,data[tag],tp,M_eff,Fit,corr)
                    if 'pi' in corr: 
                        Meff0 = gv.gvar(M_eff.mean,loosener*Fit['aE_pion_loosner']*M_eff.mean)
                        a0 = gv.gvar(a_eff.mean,loosener*Fit['Api_loosener']*a_eff.mean)
                    elif 'K' in corr: 
                        Meff0 = gv.gvar(M_eff.mean,loosener*Fit['aE_kaon_loosner']*M_eff.mean)
                        a0 = gv.gvar(a_eff.mean,loosener*Fit['AK_loosener']*a_eff.mean)
                    else: print('missing daughter meson in corr')
                    an = gv.gvar(a_eff.mean,loosener*Fit['Aeff_n_loosener']*a_eff.mean)
                    # Daughter
                    prior[func+'({0}:a)'.format(tag)] = gv_func(gv.gvar(N * [an]))     #Now amplitudes priors are for all ground and excited states
                    prior[func+'(dE:{0})'.format(tag)] = gv_func(gv.gvar(N * [En]))
                    #prior[func+'(dE:{0})'.format(tag)][1] = gv.log(gv.gvar(gv.gvar(En).mean,0.25*gv.gvar(En).mean)) #tighten 1st excited state prior
                    #here 2.3 corresponds to 2.05 * lambda QCD = Excited pdg pion 1300MeV - pdg pion 140MeV = 1150MeV.
                    #stll giving error of +- half lambda QCD.  we use 1.85 for kaon for similar reasons
                    #If splitting flag = Flase then the E1- E0 splittin is just lambda QCD
                    splitting_flag = True
                    #Forcing the n=1 pion mass width to be smaller to acocunt for spurrious state
                    Fp_pion_mass_n_eq_1_tightener = True
                    if Fit['conf'] == 'Fp':
                        pion_mass_n_eq_1_tightener = Fit['special_Fp_pion_n=1_tightener']
                    if twist !='0.0' and '0.0' in Fit['twists'] and func+'(dE:{0})'.format(tag0) in prior and tw_corr:
                        if 'pi' in corr: 
                            Ep_n0 = full_Edispersion_relation(prior['dE:{0}'.format(tag0)][0], twist, Fit['L']) * (1 + prior['eps_pi2']*ap2/(np.pi)**2)
                            prior[func+'(dE:{0})'.format(tag)][0] = gv_func(Ep_n0)
                            prior[func+'({0}:a)'.format(tag)][0] = gv_func(full_Adispersion_relation(prior['dE:{0}'.format(tag0)][0], prior['dE:{0}'.format(tag)][0], prior['{0}:a'.format(tag0)][0])*(1 + prior['A_pi2']*ap2/(np.pi)**2))
                            if splitting_flag == True:
                                Ep_n1 = full_Edispersion_relation(Meff0+gv.gvar(dE_n_eq_1), twist, Fit['L'])* (1 + prior['eps_pi2']*ap2/(np.pi)**2)
                                prior[func+'(dE:{})'.format(tag)][1] = gv_func(Ep_n1 - Ep_n0)
                        elif 'K' in corr: 
                            Ep_n0 = full_Edispersion_relation(prior['dE:{0}'.format(tag0)][0], twist, Fit['L']) * (1 + prior['eps_K2']*ap2/(np.pi)**2)
                            #prior[func+'(dE:{0})'.format(tag)][0] = gv_func(gv.gvar(Ep_n0.mean, Ep_n0.sdev*0.05))
                            prior[func+'(dE:{0})'.format(tag)][0] = gv_func(Ep_n0)
                            A_n0 = full_Adispersion_relation(prior['dE:{0}'.format(tag0)][0], prior['dE:{0}'.format(tag)][0], prior['{0}:a'.format(tag0)][0])*(1 + prior['A_K2']*ap2/(np.pi)**2)
                            #prior[func+'({0}:a)'.format(tag)][0] = gv_func(gv.gvar(A_n0.mean, A_n0.sdev*0.05))
                            prior[func+'({0}:a)'.format(tag)][0] = gv_func(A_n0)
                            #
                            if splitting_flag ==  True: 
                                Ep_n1 = full_Edispersion_relation(Meff0+gv.gvar(dE_n_eq_1), twist, Fit['L']) * (1 + prior['eps_K2']*ap2/(np.pi)**2)
                                prior[func+'(dE:{})'.format(tag)][1] = gv_func(Ep_n1 - Ep_n0)
                        else: print('dispersion variable error')
                    else: 
                        prior[func+'(dE:{0})'.format(tag)][0] = gv_func(Meff0)
                        if splitting_flag ==  True: 
                            prior[func+'(dE:{0})'.format(tag)][1] = gv_func(gv.gvar(dE_n_eq_1))
                            if Fp_pion_mass_n_eq_1_tightener == True:
                                if Fit['conf'] == 'Fp':
                                    if 'pi' in corr:
                                        dE_n_eq_1 = '{0}({1})'.format(2.05*En_gvar.mean, pion_mass_n_eq_1_tightener*En_gvar.sdev)
                                        prior[func+'(dE:{})'.format(tag)][1] = gv_func(gv.gvar(dE_n_eq_1))

                        prior[func+'({0}:a)'.format(tag)][0] = gv_func(a0)

                    # Daughter -- oscillating part
                    if twist =='0.0' and corr in non_oscillating:
                        pass
                    else:
                        prior[func+'(o{0}:a)'.format(tag)] = gv_func(gv.gvar(No * [an]))
                        prior[func+'(dE:o{0})'.format(tag)] = gv_func(gv.gvar(No * [En]))
                        #changes made to reflect new pdg inspired mass splittings to account for E0osc = E0nonosc + 1.69*lambdaqcd roughly for pion
                        #This is informed by pdg values, and this number plus a braod prior of 0.5 lambda qcd accounts for variation
                        delta_pi = 1.69
                        delta_K = 1.0 #0.373
                        #creating lowest lying osc state mass with error = .5 Lambda QCD
                        if 'K' in corr:
                            osc_mass = gv.gvar(effective_mass_calc(tag0,data[tag0],tp).mean + En_gvar.mean*delta_K, En_gvar.sdev)
                            if twist == '0.0':
                                prior[func+'(dE:o{0})'.format(tag)][0] = gv_func(gv.gvar(osc_mass.mean,osc_mass.sdev))
                            else: prior[func+'(dE:o{0})'.format(tag)][0] = gv_func(full_Edispersion_relation(osc_mass, twist, Fit['L']))*(1 + prior['osc_K2']*ap2/(np.pi)**2)
                        if 'pi' in corr:
                            osc_mass = gv.gvar(effective_mass_calc(tag0,data[tag0],tp).mean + En_gvar.mean*delta_pi, En_gvar.sdev)
                            prior[func+'(dE:o{0})'.format(tag)][0] = gv_func(full_Edispersion_relation(osc_mass, twist, Fit['L']))*(1 + prior['osc_pi2']*ap2/(np.pi)**2)
                        

        if corr in currents:
            '''print('corr:', corr)
            print('notwist0', notwist0)
            print(Fit['masses'])'''
            for mass in Fit['masses']:
                tw = -1 #implementing for twist specific 3pt Veff priors
                for twist in Fit['twists']:
                    tw += 1
                    tw_i = FitTwists[tw]
                    if twist =='0.0' and corr in notwist0:
                        #print('notwist corr:', corr, twist)
                        pass
                    else: 
                        #print('notwist corr:', corr, twist)
                        '''daughter = daughters[currents.index(corr)]
                        parent = parents[currents.index(corr)]
                        dcorr = data[Fit['{0}-Tag'.format(daughter)].format(twist)]
                        pcorr = data[Fit['{0}-Tag'.format(parent)].format(mass)]
                        correlators = []
                        for T in Fit['Ts']: 
                            correlators.append(data[Fit['threePtTag{0}'.format(corr)].format(T,Fit['m_s'],mass,Fit['m_l'],twist)])
                        ptag = Fit['{0}-Tag'.format(parent)].format(mass)
                        pM_eff = effective_mass_calc(ptag,data[ptag],tp)
                        pA_eff = effective_amplitude_calc(ptag,data[ptag],tp,pM_eff,Fit,parent)

                        dtag = Fit['{0}-Tag'.format(daughter)].format(twist)
                        dM_eff = effective_mass_calc(dtag,data[dtag],tp)
                        dA_eff = effective_amplitude_calc(dtag,data[dtag],tp,dM_eff,Fit,daughter)'''

                        #Logan making a change here - V loosener is now a prior width loosener on only and all 3pt amps
                        #additionally, Veff is now automatically the prior, no auto calc

                        Vnn0 = '{0}({1})'.format(gv.gvar(Fit['{0}Vnn0'.format(corr)][tw_i]).mean, loosener*Fit['Vloosener']*gv.gvar(Fit['{0}Vnn0'.format(corr)][tw_i]).sdev)
                        Vn   = '{0}({1})'.format(gv.gvar(Fit['{0}Vn'.format(corr)]).mean,  loosener*Fit['Vloosener']*gv.gvar(Fit['{0}Vn'.format(corr)]).sdev)
                        V0   = '{0}({1})'.format(gv.gvar(Fit['{0}V0'.format(corr)]).mean,  loosener*Fit['Vloosener']*gv.gvar(Fit['{0}V0'.format(corr)]).sdev)
                    V10_widener = Fit['V10_widener']
                    #Vosc00 = 1 #attempt to account for highly oscillating terms
                    
                    if twist =='0.0' and corr in notwist0:
                        pass
                    elif twist =='0.0' and daughters[currents.index(corr)] in non_oscillating :
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(Vnn0)
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][1][0] = gv.gvar(Vn) * V10_widener    
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [No* [Vn]])
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0) * V10_widener 
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)][1][0] = gv.gvar(Vn) * V10_widener      
                    else:
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [N * [Vn]])
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(Vnn0)
                        prior['{0}Vnn_m{1}_tw{2}'.format(corr,mass,twist)][1][0] = gv.gvar(Vn) * V10_widener   
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(N * [No * [Vn]])
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0) 
                        prior['{0}Vno_m{1}_tw{2}'.format(corr,mass,twist)][1][0] = gv.gvar(Vn) * V10_widener  
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(No * [No * [Vn]])
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0) 
                        prior['{0}Voo_m{1}_tw{2}'.format(corr,mass,twist)][1][0] = gv.gvar(Vn) * V10_widener  
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)] = gv.gvar(No * [N * [Vn]])
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)][0][0] = gv.gvar(V0) * V10_widener 
                        prior['{0}Von_m{1}_tw{2}'.format(corr,mass,twist)][1][0] = gv.gvar(Vn) * V10_widener 
    return(prior)
            
######################################################################################################

def get_p0(Fit,fittype,Nexp,allcorrs,prior,FitCorrs, Nmarg=None):
    # We want to take in several scenarios in this order, choosing the highest in preference. 
    # 1) This exact fit has been done before, modulo priors, svds t0s etc
    # 1marg) This exact chained marginilized fit has been done before, modulo priors, svds t0s etc, but with a larger nmarg
    # 2) Same but different type of fit, eg marginalised 
    # 3) This fit has been done before with Nexp+1
    # 4) This fit has been done beofore with Nexp-1
    # 5a) Some elemnts have bene fitted to Nexp before,
    # 5b) Some elements of the fit have been fitted in other combinations before

    if Nmarg == None:
        filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp)
    else:
        filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp,Nmarg[0])
    #filename1_marg_up = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp,Nmarg+1)
    #filename1_marg_down = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp,Nmarg-1)
    filename2 = 'p0/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),Nexp)
    filename3 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp+1)
    filename4 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp-1)
    filename5a = 'p0/{0}{1}{2}'.format(Fit['conf'],Fit['filename'],Nexp)
    filename5b = 'p0/{0}{1}'.format(Fit['conf'],Fit['filename'])
    #case 1
    if os.path.isfile(filename1):
        p0 = gv.load(filename1)
        print('Loaded p0 from exact fit')

    
    #case 1marg
    #elif (Nmarg != None and fittype == 'chained-marginalised'):
    #    if os.path.isfile(filename1_marg_up):
    #        p0 = gv.load(filename1_marg_up)
    #        print('Loaded p0 from exact fit with 1 less marginalized exp (greater Nmarg)')
    #    elif os.path.isfile(filename1_marg_down):
    #        p0 = gv.load(filename1_marg_down)
    #        print('Loaded p0 from exact fit with 1 more marginalized exp (smaller Nmarg)')
    #    else: p0 = None
    
            
    #case 2
    
    elif os.path.isfile(filename2):
        p0 = gv.load(filename2)
        print('Loaded p0 from exact fit of different type')
    #case 3    
    elif os.path.isfile(filename3):
        p0 = gv.load(filename3)
        print('Loaded p0 from exact fit Nexp+1')
        
    #case 4    
    elif os.path.isfile(filename4):
        p0 = gv.load(filename4)
        print('Loaded p0 from exact fit Nexp-1')
        
    #case 5    
    elif os.path.isfile(filename5b):
        p0 = gv.load(filename5b)
        print('Loaded global p0')
        if os.path.isfile(filename5a):
            pnexp = gv.load(filename5a)
            for key in pnexp:
                if key in prior:
                    if key not in p0:
                        print('Error: {0} in global Nexp but not in global fit'.format(key))
                        p0[key] = pnexp[key]
                    del p0[key]
                    p0[key] = pnexp[key]
                    print('Loaded {0} p0 from global Nexp'.format(key))
    
    else:
        p0 = None
    return(p0)
######################################################################################################

def update_p0(p,finalp,Fit,fittype,Nexp,allcorrs,FitCorrs,Q,marg=False, Nmarg = None):
    # We want to take in several scenarios in this order 
    # 1) This exact fit has been done before, modulo priors, svds t0s etc
    # 2) Same but different type of fit, eg marginalised 
    # 3) Global Nexp
    # 4) Global
    # 5) if Marg is True, we don't want to save anything but filename 1 as Nexp = nmarg and is not similar to if we do other fits

    if Nmarg == None:
        filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp)
    else:    
        filename1 = 'p0/{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp,Nmarg[0])
    filename2 = 'p0/{0}{1}{2}{3}{4}{5}{6}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),Nexp)
    filename3 = 'p0/{0}{1}{2}'.format(Fit['conf'],Fit['filename'],Nexp)
    filename4 = 'p0/{0}{1}'.format(Fit['conf'],Fit['filename'])

    #case 1
    
    for element in ['eps_pi2','eps_K2','osc_pi2','osc_K2', 'A_pi2', 'A_K2']:
        for corr in allcorrs:
            #if '{0}_{1}'.format(element,corr) in p:
            #    del p['{0}_{1}'.format(element,corr)]
            if element in p:
                del p[element]
    
    for element in ['eps_pi2','eps_K2','osc_pi2','osc_K2', 'A_pi2', 'A_K2']:
        for corr in allcorrs:
            #if '{0}_{1}'.format(element,corr) in finalp:
            #    del finalp['{0}_{1}'.format(element,corr)]
            if element in finalp:
                del finalp[element]
    gv.dump(p,filename1)
    if marg == False:
        #case 2
        gv.dump(finalp,filename2)

        #case 3
        if os.path.isfile(filename3) and Q > 0.05:
            p0 = gv.load(filename3) #load exisiting global Nexp
            for key in finalp:  # key in this output
                p0[key] =  finalp[key]  #Update exisiting and add new
            gv.dump(p0,filename3)
    
        else:
            gv.dump(finalp,filename3)

        if os.path.isfile(filename4) and Q > 0.05:
            p0 = gv.load(filename4) # load existing, could be any length
            for key in finalp:  # key in new 
                if key in p0: # if 
                    #print('shape p0[key]',np.shape(p0[key]),key)#######
                    if len(np.shape(p0[key])) == 1 and len(p0[key]) <= Nexp:
                        #print('shape p0[key]',np.shape(p0[key]),key)
                        del p0[key]
                        p0[key] = finalp[key]
                        print('Updated global p0 {0}'.format(key))
                    elif np.shape(p0[key])[0] <= Nexp:
                        #print('shape p0[key]',np.shape(p0[key]),key)
                        del p0[key]
                        p0[key] = finalp[key]
                        print('Updated global p0 {0}'.format(key))
                else:
                    p0[key] =  finalp[key]
                    print('Added new element to global p0 {0}'.format(key))
            gv.dump(p0,filename4)
        else:
            gv.dump(finalp,filename4)
    return()

######################################################################################################

def save_fit(fit,Fit,allcorrs,fittype,Nexp,SvdFactor,PriorLoosener,currents,smallsave):
    filename = 'Fits/{0}{1}{2}{3}{4}{5}{6}_Nexp{7}_sfac{8}_pfac{9}_Q{10:.2f}_chi{11:.3f}_sm{12}'.format(Fit['conf'],Fit['filename'],strip_list(Fit['masses']),strip_list(Fit['twists']),strip_list(allcorrs),strip_list(Fit['Ts']),fittype,Nexp,SvdFactor,PriorLoosener,fit.Q,fit.chi2/fit.dof,smallsave)
    for corr in allcorrs:
        if corr in currents:
            filename += '_{0}tmin{1}'.format(corr,Fit['{0}tmin'.format(corr)])
    savedict = gv.BufferDict()
    if smallsave:
        for key in fit.p:
            if len(key) <5:
                pass
            elif key[0] == 'l':
                key2 = key.split('(')[1].split(')')[0]
                if key2.split(':')[0] =='dE' and key2.split(':')[1][0] != 'o':
                    savedict[key] = [fit.p[key][0]] #palt
            elif key[2] =='n' and key[3] == 'n':
                savedict[key] = [[fit.p[key][0][0]]] #palt
            elif key[3] =='n' and key[4] == 'n':
                savedict[key] = [[fit.p[key][0][0]]] #palt
    elif smallsave == False:
        savedict = fit.p
    print('Started gv.gdump to {1}, smallsave = {0}'.format(smallsave,'{0}.pickle'.format(filename)),datetime.datetime.now())        
    gv.gdump(savedict,'{0}.pickle'.format(filename))
    print('Finished gv.gdump fit, starting save fit output',datetime.datetime.now())
    f = open('{0}.txt'.format(filename),'w')
    f.write(fit.format(pstyle='v'))
    f.close()
    ###########################

    #fit_compare = open('fit_compare.txt','w')
    #for line in fit.format(maxline=10000):
    #    fit_compare.write('{}\n'.format(line))
    #fit_compare.close()
    #print(fit.format(maxline = 100))

    ############################
    ############################
    ############################

    '''coef_array = open('coef_array.txt','w')
    ctag_array = open('ctag_array.txt','w')
    for line in fit.p:
        ctag_array.write('{}\n'.format(line))
        coef_array.write('{}, {}, {}\n'.format(fit.p[line][0].mean,fit.p[line][1].mean,fit.p[line][2].mean))
    coef_array.close()
    ctag_array.close()'''

    ###########################
    ###########################
    ###########################
    print('Finished save fit output',datetime.datetime.now())
    return(filename)
    

######################################################################################################

def do_chained_fit(pdata,prior,Nexp,models,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,save,smallsave,GBF,Tolerance):#if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with no marginalisation Nexp = NMax
    #fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-6,0.0,0.0))
    
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(Tolerance))
    p0 = get_p0(Fit,'chained',Nexp,allcorrs,prior,FitCorrs) 
    print(30 * '=','Chained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())
    #print(prior)
    fit = fitter.chained_lsqfit(pdata=pdata, prior=prior, p0=p0, noise=noise,debug=True)
    #fit = fitter.chained_lsqfit(pdata=pdata, prior=prior, p0=p0, noise=noise,debug=True)
    update_p0([f.pmean for f in fit.chained_fits.values()],fit.pmean,Fit,'chained',Nexp,allcorrs,FitCorrs,fit.Q) #fittype=chained, for marg,includeN
    if GBF == None:
        print(fit)
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior,Fit)
        #print_Z_V(fit.p,Fit,allcorrs)
        '''if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)'''
            #print_fit_results(fit) do this later
        if save:
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents,smallsave) ### Force to save
    elif fit.logGBF - GBF < 1 and fit.logGBF - GBF > 0:
        print('log(GBF) went up by less than 1: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    elif fit.logGBF - GBF < 0:
        print('log(GBF) went down {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    else:
        print(fit)
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior,Fit)
        #print_Z_V(fit.p,Fit,allcorrs)
        print('log(GBF) went up {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return(fit.logGBF)

######################################################################################################

def do_chained_marginalised_fit(data,prior,Nexp,models,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,FitCorrs,save,smallsave,GBF,Tolerance,Nmarg):#if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with marginalisation nterm = nexp,nexp Nmarg=Marginalisation us in p0 bits
 
    fitter = cf.CorrFitter(models=models, maxit=maxiter, nterm=Nmarg, fast=False, tol = Tolerance)
    p0 = get_p0(Fit,'chained-marginalised',Nexp,allcorrs,prior,FitCorrs, Nmarg=Nmarg)
    print(30 * '=','Chained-marginalised','Nexp =',Nexp,'nterm = ({0},{1})'.format(Nmarg[0], Nmarg[1]),'Date',datetime.datetime.now())
    fit = fitter.chained_lsqfit(pdata=data, prior=prior, p0=p0, noise=noise,debug=True)
    update_p0([f.pmean for f in fit.chained_fits.values()],fit.pmean,Fit,'chained-marginalised',Nexp,allcorrs,FitCorrs,fit.Q,marg=True,Nmarg=Nmarg) #fittype=chained, for marg,includeN
    if GBF == None:
        print(fit)#.format(pstyle='m'))
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior,Fit)
        #print_Z_V(fit.p,Fit,allcorrs)
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained-marginalised_Nmarg{0}'.format(Nmarg),Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return()
    elif fit.logGBF - GBF < 1 and fit.logGBF - GBF > 0:
        print('log(GBF) went up by less than 1: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    elif fit.logGBF - GBF < 0:
        print('log(GBF) went down {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    else:
        print(fit)#.format(pstyle='m'))
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior,Fit)
        #print_Z_V(fit.p,Fit,allcorrs)
        print('log(GBF) went up {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'chained-marginalised_Nmarg{0}'.format(Nmarg),Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return(fit.logGBF)

######################################################################################################
#implementing funcitonality for 18 July 2025 change of alt/new main funciton
#Leaves out some old implementation for simplicity
def re_im_fit(data,prior,Nexp,models,svdcut,Fit,noise,allcorrs):
    print('Models',models)
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-8,0.0,0.0))
    #p0 = get_p0(Fit,'unchained',Nexp,allcorrs,prior,allcorrs) 
    print(30 * '=','Unchained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())
    fit = fitter.lsqfit(pdata=data, prior=prior, svdcut=svdcut, noise=noise,debug=True, udata = None)
    print(fit.format(maxline=10000))
    print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
    print_results(fit.p,prior,Fit)
    return(fit.p)
######################################################################################################
# 10/10/2025
#implementing Judd's partial variance cut
def pvc_re_im_fit(dset,prior,Nexp,models,svdcut,Fit,noise,allcorrs):
    fitter = vc.pvc_CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-8,0.0,0.0))
    p0 = get_p0(Fit,'unchained',Nexp,allcorrs,prior,allcorrs) 
    print(30 * '=','Partical Variance Cut Fit','Nexp =',Nexp,'Date',datetime.datetime.now())
    #partial_variance_cut= 'auto'
    fit_vc=fitter.pvc_lsqfit(dset=dset, prior=prior, p0=p0, svdcut=svdcut, noise=noise,debug=True, partial_variance_cut='auto') 
    print(fit_vc.format(maxline=10000))
    print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit_vc.chi2/fit_vc.dof,fit_vc.Q,fit_vc.logGBF))
    print_results(fit_vc.p,prior,Fit)
    return(fit_vc.p)
######################################################################################################
#implementing funcitonality for 07 Aug 2025 change of alt/new main funciton
#Leaves out some old implementation for simplicity
#Changes fit naming protocal to more simple combo of: ensmeble, decay index (Hpi or HsK), and date-time
def alt_save_fit(combined_fit,ens,Decay_Index):
    if Decay_Index == 0: decay_tag = 'Hpi'
    else: decay_tag = 'HsK'
    filename = 'Fits/{0}_{1}_{2}'.format(ens,decay_tag, datetime.datetime.now())
    print('Started gv.gdump to {0}'.format('{0}.pickle'.format(filename)))        
    gv.gdump(combined_fit,'{0}.pickle'.format(filename))
    print('Finished gv.gdump fit, starting save fit output')
    f = open('{0}.txt'.format(filename),'w')
    f.write(str(combined_fit))
    f.close()
    ###########################
    print('Finished saving combined fit output')
    #return(filename)
######################################################################################################

def do_unchained_fit(data,prior,Nexp,models,svdcut,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,save,smallsave,GBF,call_tag,Append_Key_Fit_Stats):#if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with no marginalisation Nexp = NMax
    print('Models',models)
    # Begin implementation of forced unccorrelation of data
    #    for i in range(len(list(data[key]))):
    #        data[key][i] = gv.gvar(data[key][i].mean, data[key][i].sdev)
    # End implementation of forced unccorrelation of data
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-6,0.0,0.0))
    p0 = get_p0(Fit,'unchained',Nexp,allcorrs,prior,allcorrs) # FitCorrs = allcorrs 
    print(30 * '=','Unchained-Unmarginalised','Nexp =',Nexp,'Date',datetime.datetime.now())
    fit = fitter.lsqfit(pdata=data, prior=prior, svdcut=svdcut, noise=noise,debug=True, udata = None)
    #fit = fitter.lsqfit(pdata=data, prior=prior, p0=p0, svdcut=svdcut, noise=noise,debug=True)
    #update_p0(fit.pmean,fit.pmean,Fit,'unchained',Nexp,allcorrs,allcorrs,fit.Q) #fittype=chained, for marg,includeN
    ### do_tmin_2pt_Testing(fit.p, Fit, Nexp, fit.Q)   #### Comment this out when not tmin testing
    #do_tmin_3pt_Testing(fit.p, Fit, Nexp, fit.Q)   #### Comment this out when not tmin testing
    if GBF == None:
        print(fit.format(maxline=10000))
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior,Fit)
        #print(gv.evalcorr(data['2pt_pionG5-G5_th5.311'][0:5]))
        #print_Z_V(fit.p,Fit,allcorrs)
        #if fit.Q > 0.05 and save: #threshold for a 'good' fit
        if save ==True:
            fit_location = save_fit(fit,Fit,allcorrs,'unchained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            if Append_Key_Fit_Stats == True:
                append_Key_Fit_Stats(call_tag, fit.chi2, fit.dof, fit.Q, fit.logGBF, Nexp, PriorLoosener, fit_location)
                #append_Key_Fit_Stats_SVD(call_tag, fit.chi2, fit.dof, fit.Q, fit.logGBF, svdcut, fit.svdn, fit_location, SvdFactor)
            #print_fit_results(fit) do this later
        return()
    elif fit.logGBF - GBF < 1 and fit.logGBF - GBF > 0:
        print('log(GBF) went up by less than 1: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    elif fit.logGBF - GBF < 0:
        print('log(GBF) went down: {0:.2f}'.format(fit.logGBF - GBF))
        return(fit.logGBF)
    else:
        print(fit.format(maxline=10000))
        print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
        print_results(fit.p,prior,Fit)
        #print_Z_V(fit.p,Fit,allcorrs)
        print('log(GBF) went up more than 1: {0:.2f}'.format(fit.logGBF - GBF))
        if fit.Q > 0.05 and save: #threshold for a 'good' fit
            save_fit(fit,Fit,allcorrs,'unchained',Nexp,SvdFactor,PriorLoosener,currents,smallsave)
            #print_fit_results(fit) do this later
        return(fit.logGBF)

#######################################################################################################
#Tacked on function that appends a csv file to make comparing slightly different fits much easier
def append_Key_Fit_Stats_SVD(call_tag, chi2, dof, Q, logGBF, svd, n, fit_locaiton, SvdFactor):
    dof_str = '{}'.format(dof)
    chi2_per_dof_str = '{0:.3f}'.format(chi2/dof)
    Q_str = '{0:.3f}'.format(Q)
    logGBF_str = '{0:.0f}'.format(logGBF)
    svd_str = '{:.2E}'.format(svd)
    n_str = '{}'.format(n)
    row = [call_tag, chi2_per_dof_str, Q_str, logGBF_str, svd_str, SvdFactor, n_str, dof_str, fit_locaiton]
    f = open('Key_Fit_Stats.csv', 'a')
    writer = csv.writer(f, delimiter =',')
    writer.writerow(row)
    f.close()
    print('Key fit stats appended to Key_Fit_Stats.csv')
    #######################################################################################################
#Tacked on function that appends a csv file to make comparing slightly different fits much easier
def append_Key_Fit_Stats(call_tag, chi2, dof, Q, logGBF, Nexp, prior_widener, fit_locaiton):
    chi2_per_dof_str = '{0:.3f}'.format(chi2/dof)
    Q_str = '{0:.3f}'.format(Q)
    logGBF_str = '{0:.0f}'.format(logGBF)
    row = [call_tag, chi2_per_dof_str, Q_str, logGBF_str, Nexp, prior_widener, fit_locaiton]
    f = open('Key_Fit_Stats.csv', 'a')
    writer = csv.writer(f, delimiter =',')
    writer.writerow(row)
    f.close()
    print('Key fit stats appended to Key_Fit_Stats.csv')

#######################################################################################################

def do_sep_mass_fit(data,prior,Nexp,models,svdcut,Fit,noise,currents,allcorrs,SvdFactor,PriorLoosener,save,smallsave,GBF):
    #if GBF = None doesn't pass GBF, else passed GBF 
    #do chained fit with no marginalisation Nexp = NMax
    print('Models',models)
    fitter = cf.CorrFitter(models=models, maxit=maxiter, fast=False, tol=(1e-6,0.0,0.0))
    p0 = get_p0(Fit,'sepmass',Nexp,allcorrs,prior,allcorrs) # FitCorrs = allcorrs 
    print(30 * '=','Seperate Mass Fit','Nexp =',Nexp,'Date',datetime.datetime.now())
    fit = fitter.lsqfit(pdata=data, prior=prior, p0=p0, svdcut=svdcut, noise=noise,debug=True)
    update_p0(fit.pmean,fit.pmean,Fit,'sepmass',Nexp,allcorrs,allcorrs,fit.Q) #fittype=chained, for marg,includeN
    print(fit)
    print('chi^2/dof = {0:.3f} Q = {1:.3f} logGBF = {2:.0f}'.format(fit.chi2/fit.dof,fit.Q,fit.logGBF))
    print_results(fit.p,prior,Fit)
    return(fit)

######################################################################################################

def print_p_p0(p,p0,prior):
    print('{0:<40}{1:<20}{2:<40}{3:<20}'.format('key','p','p0','prior'))
    for key in prior:
        if len(np.shape(p[key])) ==1 :
            for element in range(len(p[key])):
                if element == 0:
                    print('{0:<40}{1:<20}{2:<40}{3:<20}'.format(key,p[key][element],p0[key][element],prior[key][element]))
                else:
                    print('{0:>40}{1:<20}{2:<40}{3:<20}'.format('',p[key][element],p0[key][element],prior[key][element]))
    return()

#####################################################################################################
    ### 16/02/2024 - Logan makes change here to view correlations between final posteriors
def print_results(p,prior,Fit):
    Corr_elements = [] ### CorrMtrx addition
    Corr_keys = [] ### ### CorrMtrx addition
    print(100*'-')
    print('{0:<40}{1:<15}{2:<15}{3:<15}{4}'.format('key','p','p error','prior','prior error'))
    print(100*'-')
    print('Dispersion Relation Variables')
    print(100*'-')
    for key in prior:
        if key in ('eps_pi2','eps_K2', 'A_pi2', 'A_K2', 'osc_pi2', 'osc_K2'):
            pr = prior[key]
            po = p[key]
            print('{0:<40}{1:<15}{2:<15}{3:<15}{4:<10}'.format(key,po,'-----',pr,'-----'))
    print(100*'-')
    print('Ground state energies')
    print(100*'-')
    for key in prior:
        if key[0] in ('l','s'):
            key = key.split('(')[1].split(')')[0]
        if key.split(':')[0] =='dE' and key.split(':')[1][0] != 'o':
            pr = prior[key][0]
            po = p[key][0]
            d = po-pr
            n = int(abs(d.mean/d.sdev))
            print('{0:<40}{1:<15}{2:<15.3%}{3:<15}{4:<10.2%}{5:<5}'.format(key,po,po.sdev/po.mean,pr,pr.sdev/pr.mean,n*'*'))
            
            #if '{0}'.format(key.split(':')[1]) == Fit['BG-Tag'].format(Fit['masses'][0]):
             #   print('split: ', p['dE:{0}'.format(Fit['BNG-Tag'].format(Fit['masses'][0]))][0]-p[key][0])  
    print('')
    print('Oscillating lowest lying energies')
    print(100*'-')
    for key in prior:
        if key[0] == 'l':
            key = key.split('(')[1].split(')')[0]
        if key.split(':')[0] =='dE' and key.split(':')[1][0] == 'o':
            pr = prior[key][0]
            po = p[key][0]
            d = po-pr
            n = int(abs(d.mean/d.sdev))
            print('{0:<40}{1:<15}{2:<15.3%}{3:<15}{4:<10.2%}{5:<5}'.format(key,po,po.sdev/po.mean,pr,pr.sdev/pr.mean,n*'*'))
    
    print('')
    print('Ground state 2pt-Amplitudes')
    print(100*'-')
    for key in prior:
        if key[0] in ('l','s'):
            key = key.split('(')[1].split(')')[0]
        if key.endswith('a'):
            if not key.startswith('o'):
                pr = prior[key][0]
                po = p[key][0]
                d = po-pr
                n = int(abs(d.mean/d.sdev))
                print('{0:<40}{1:<15}{2:<15.3%}{3:<15}{4:<10.2%}{5:<5}'.format(key,po,po.sdev/po.mean,pr,pr.sdev/pr.mean,n*'*'))
    
    print('')
    print('Oscillating lowest lying 2pt-Amplitudes')
    print(100*'-')
    for key in prior:
        if key[0] in ('l','s'):
            key = key.split('(')[1].split(')')[0]
        if key.endswith('a'):
            if key.startswith('o'):
                pr = prior[key][0]
                po = p[key][0]
                d = po-pr
                n = int(abs(d.mean/d.sdev))
                print('{0:<40}{1:<15}{2:<15.3%}{3:<15}{4:<10.2%}{5:<5}'.format(key,po,po.sdev/po.mean,pr,pr.sdev/pr.mean,n*'*'))
    print('')

    print('V_nn[0][0]')
    print(100*'-')
    for key in prior:
        key_ext = '{0}____'.format(key)
        if key_ext[1] != '2' and key_ext[2] =='n' and key_ext[3] == 'n':
            pr = prior[key][0][0]
            po = p[key][0][0]
            Corr_elements.append(po) ### CorrMtrx addition
            Corr_keys.append(key)
            d = po-pr
            n = int(abs(d.mean/d.sdev))
            print('{0:<40}{1:<15}{2:<15.3%}{3:<15}{4:<10.2%}{5:<5}'.format(key,po,po.sdev/po.mean,pr,pr.sdev/pr.mean,n*'*'))
        if key_ext[1] != '2' and key_ext[3] =='n' and key_ext[4] == 'n':
            pr = prior[key][0][0]
            po = p[key][0][0]
            Corr_elements.append(po) ### CorrMtrx addition
            Corr_keys.append(key)
            d = po-pr
            n = int(abs(d.mean/d.sdev))
            print('{0:<40}{1:<15}{2:<15.3%}{3:<15}{4:<10.2%}{5:<5}'.format(key,po,po.sdev/po.mean,pr,pr.sdev/pr.mean,n*'*'))
    print(100*'-')

    '''
    ### CorrMtrx PLotting
    plt.rcParams.update({'font.size': 28})
    CorrMtrx = np.abs(np.array(gv.evalcorr(Corr_elements)))

    fig, ax = plt.subplots(ncols=1,figsize=(20,20))#gridspec_kw={"height_ratios":[0.05]})
    heatmap = ax.imshow(CorrMtrx, cmap= 'Wistia')
    #fig.colorbar(heatmap, ax=ax)
    cbar = fig.colorbar(heatmap, ax=ax, location='right', shrink=0.7)#anchor=(0, 0.3)
    cbar.minorticks_on()


    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(Corr_keys)), labels = Corr_keys)
    ax.set_yticks(np.arange(len(Corr_keys)), labels = Corr_keys)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    for i in range(len(Corr_keys)):
        for j in range(len(Corr_keys)):
            if i == j: txt_color = "black"
            else: txt_color = "black"
            text = ax.text(j, i, round(CorrMtrx[i, j],2),ha="center", va="center", color=txt_color)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    #ax.set_title('H→π  F-5 ensemble, 3pt-Amplitude Correlation Matrix', y=0, pad = -25, verticalalignment = "top")
    
    #fig.colorbar(heatmap, cax=ax,orientation="vertical", pad=0.1)
    
    plt.tight_layout()
    #plt.title('{}-ensemble correlation matrix subset'.format(Fit['conf']), y=-0.03)
    plt.savefig('./CorrMtrx_plots/{}_{}.svg'.format(Fit['conf'],datetime.datetime.now()), format = 'svg')
    '''

    return()
#####################################################################################################
def do_tmin_2pt_Testing(p, Fit, Nexp, Q):
    print(100*'-')
    taste = 'HlG5-G5X'
    m0, m1, m2, m3 = Fit['masses'][0], Fit['masses'][1], Fit['masses'][2], Fit['masses'][3]
    key_E1, key_E2 = 'dE:2pt_'+ taste +'_m'+ m0 + '_th0.0' , 'dE:2pt_'+ taste +'_m'+ m1 + '_th0.0'
    key_E3, key_E4 = 'dE:2pt_'+ taste +'_m'+ m2 + '_th0.0' , 'dE:2pt_'+ taste +'_m'+ m3 + '_th0.0'

    key_A1, key_A2 = '2pt_'+ taste +'_m' + m0 +'_th0.0:a', '2pt_'+ taste +'_m' + m1 +'_th0.0:a'
    key_A3, key_A4 = '2pt_'+ taste +'_m' + m2+'_th0.0:a', '2pt_'+ taste +'_m' + m3 +'_th0.0:a'
 
    keys_All = [key_E1, key_E2, key_E3, key_E4, key_A1, key_A2, key_A3, key_A4]
    pos = []

    for i in range(8):
        pos.append(p[keys_All[i]][0])


    #row = 'Bpi, {0}, Nexp={1}, tmin={2}, Q={3:.3e}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}'.format(Fit['conf'], Nexp, Fit['tmin_2pt'], Q, pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7])
    row = ['Bpi', '{}'.format(Fit['conf']), Nexp, Fit['tmin_2pt'], '{:.3e}'.format(Q), pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7]]
    f = open('./tmin_testing_2pt/'+ Fit['conf'] + '/' + taste +'.csv', 'a')
    writer = csv.writer(f, delimiter =',')
    writer.writerow(row)
    f.close()
    print('Appending to ./'+ Fit['conf'] +'/'+ taste +'.csv')

#####################################################################################################
def do_tmin_3pt_Testing(p, Fit, Nexp, Q):
    print(100*'-')
    x_tag = 'XVnn'
    #width = Fit['Ts'][0]
    m0, m3, th1, th4 = Fit['masses'][0], Fit['masses'][1], Fit['twists'][1], Fit['twists'][2]
    key_V01, key_V04 = x_tag + '_m' + m0 + '_tw' + th1, x_tag + '_m' + m0 + '_tw' + th4
    key_V31, key_V34 = x_tag + '_m' + m3 + '_tw' + th1, x_tag + '_m' + m3 + '_tw' + th4

 
    keys_All = [key_V01, key_V04,key_V31, key_V34]
    pos = []

    for i in range(4):
        pos.append(p[keys_All[i]][0][0])


    row = ['Bpi', '{}'.format(Fit['conf']), Nexp, Fit['tmin_3pt'], '{:.3e}'.format(Q), pos[0], pos[1], pos[2], pos[3]]
    f = open('./tmin_testing_3pt/'+ Fit['conf'] + '/' + x_tag +'.csv', 'a')
    writer = csv.writer(f, delimiter =',')
    writer.writerow(row)
    f.close()
    print('Appending to ./'+ Fit['conf'] +'/'+ x_tag +'.csv')

#####################################################################################################
#####################################################################################################
