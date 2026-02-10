from matplotlib import pyplot as plt
import gvar as gv
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

#fit_file = 'Fbinned_f5_all0.4500.00.4281B5B5TB5XBYZSTVXpi24unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.573_smFalse_Stmin2_Ttmin2_Vtmin2_Xtmin2.pickle'
#fit_file = 'Fbinned_f5_all0.4500.00.4281Bs5Bs5TBs5XBsYZKSsTsVsXs24unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.559_smFalse_Sstmin2_Tstmin2_Vstmin2_Xstmin2.pickle'
#fit_file = 'UFbinned_uf5_all0.1940.00.706B5B5TB5XBYZSTVXpi44unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.219_smFalse_Stmin2_Ttmin2_Vtmin2_Xtmin2.pickle'
#fit_file = 'UFbinned_uf5_all0.1940.00.706Bs5Bs5TBs5XBsYZKSsTsVsXs44unchained_Nexp4_sfac1.0_pfac1.0_Q1.00_chi0.357_smFalse_Sstmin2_Tstmin2_Vstmin2_Xstmin2.pickle'

#Mass Check
fit_file = 'Fbinned_f5_all0.4500.550.6750.0B5B5TB5XBYZSTVXpi24unchained_Nexp3_sfac1.0_pfac1.0_Q1.00_chi0.386_smFalse_Stmin2_Ttmin2_Vtmin2_Xtmin2.pickle'
#Twist Check
fit_file = 'Fbinned_f5_all0.4500.00.42811.2822.1410B5B5TB5XBYZSTVXpi24unchained_Nexp3_sfac1.0_pfac1.0_Q1.00_chi0.437_smFalse_Stmin2_Ttmin2_Vtmin2_Xtmin2.pickle'


tag = 'TWISTTEST'

fit_file = './fitting/Fits/'+fit_file
print('Loading in ', fit_file, '...')
gv_data = gv.load(fit_file)

#for key in gv_data: print(key[7:-7])

#keystrings = ['dE:2pt_HlG5-G5_', 'dE:2pt_HlG5T-G5T_', 'dE:2pt_HlG5-G5X_', 'dE:2pt_HlG5T-GYZ_', 'SVnn', 'VVnn', 'XVnn', 'TVnn'] #'dE:2pt_pionG5-G5',
#keystrings = ['dE:2pt_HlG5-G5_', 'dE:2pt_HlG5T-G5T_',  'SVnn', 'VVnn'] #'dE:2pt_pionG5-G5',
keystrings = ['dE:2pt_pionG5-G5','SVnn', 'TVnn'] #'dE:2pt_pionG5-G5',

alt_strings = [r'$E_\pi$', r'$S$', r'$T$',]

tws = ['0.4281', '1.282', '2.1410', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR']

#alt_strings = [r'$E_\pi$', r'$E_H$ $(S)$', r'$E_{{H^\prime}}$ $(V^0)$', r'$E_{{H^{{\prime\prime}}}}$ $(V^1)$', r'$E_{{H^{{\prime\prime\prime}}}}$ $(T)$', r'$S$', r'$V^0$', r'$V^1$', r'$T$']
#alt_strings = [r'$A_\pi$', r'$A_H$ $(S)$', r'$A_{{H^\prime}}$ $(V^0)$', r'$A_{{H^{{\prime\prime}}}}$ $(V^1)$', r'$E_{{H^{{\prime\prime\prime}}}}$ $(T)$', r'$S$', r'$V^0$', r'$V^1$', r'$T$']

#keystrings = ['dE:2pt_kaonG5-G5_th0.4281', 'dE:2pt_HsG5-G5_', 'dE:2pt_HsG5T-G5T_', 'dE:2pt_HsG5-G5X_', 'dE:2pt_HsG5T-GYZ_', 'SsVnn', 'VsVnn', 'XsVnn', 'TsVnn'] #'dE:2pt_pionG5-G5',
#alt_strings = [r'$E_K$', r'$E_{{H_s}}$ $(S_s)$', r'$E_{{H_s^\prime}}$ $(V_s^0)$', r'$E_{{H_s^{{\prime\prime}}}}$ $(V_s^1)$', r'$E_{{H_s^{{\prime\prime\prime}}}}$ $(T_s)$', r'$S_s$', r'$V_s^0$', r'$V_s^1$', r'$T_s$']

keys, params_0 = [], []

x = 0

for keystring in keystrings:
    tw = 0
    for key in gv_data:
        if '2pt' in key:
            #if ':a' in key:
            #    if 'o2pt' not in key:
            if keystring in key:
                if '0.0' not in key:
                    keys.append(r'{}, $\theta = {}$'.format(alt_strings[x], tws[tw]))
                    params_0.append(gv_data[key][0])
                    tw += 1
        elif 'Vnn' in key:
            if keystring in key:
                if '0.0' not in key:
                    keys.append(r'{}, $\theta = {}$'.format(alt_strings[x], tws[tw]))
                    params_0.append(gv_data[key][0][0])
                    tw += 1        
    x += 1

# ### CorrMtrx PLotting
plt.rcParams.update({'font.size': 28})
CorrMtrx = np.abs(np.array(gv.evalcorr(params_0)))

fig, ax = plt.subplots(ncols=1,figsize=(13,13))#gridspec_kw={"height_ratios":[0.05]})
heatmap = ax.imshow(CorrMtrx, cmap= 'Wistia')
#fig.colorbar(heatmap, ax=ax)
#cbar = fig.colorbar(heatmap, ax=ax, location='right', shrink=0.7)#anchor=(0, 0.3)
#cbar.minorticks_on()


# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(len(keys)), labels = keys)
ax.set_yticks(np.arange(len(keys)), labels = keys)
ax.invert_yaxis()
ax.xaxis.tick_top()
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

for i in range(len(keys)):
    for j in range(len(keys)):
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
plt.savefig('./cov_matrix/{}_custom_mtrx.pdf'.format(tag), format = 'pdf')
print('Plotting complete.')


