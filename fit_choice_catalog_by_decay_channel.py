### Logan Roberts #
### 12/04/2024 ####


def fit_Pick_by_Decay_Channel(ensemble, decay_channel):
    ### Assuming the pickle file are stored in ../fitting/Fits/
    d = './fitting/Fits/'
    ### F
    F_fits = {}
    ### Fp
    Fp_fits = {}
    ### SF
    SF_fits = {}
    ### SFp
    SFp_fits = {}
    ### UF - Temp Temp Temp
    UF_fits = {}
   
    Sep2025 = True
    if Sep2025 == True:
        F_fits['Hpi'] = d + 'F_Hpi_2025-08-30 18:05:40.638677.pickle'
        F_fits['HsK'] = d + 'F_HsK_2025-08-30 22:55:11.520107.pickle'

        Fp_fits['Hpi'] = d + 'Fp_Hpi_2025-08-30 19:12:19.031305.pickle'
        Fp_fits['HsK'] = d + 'Fp_HsK_2025-08-31 00:40:11.096266.pickle'

        SF_fits['Hpi'] = d + 'SF_Hpi_2025-08-30 20:05:39.995200.pickle'
        SF_fits['HsK'] = d + 'SF_HsK_2025-08-31 01:41:05.973244.pickle'

        SFp_fits['Hpi'] = d + 'SFp_Hpi_2025-08-30 21:34:27.596241.pickle'
        SFp_fits['HsK'] = d + 'SFp_HsK_2025-08-31 01:52:23.701160.pickle'

        UF_fits['Hpi'] = d + 'UF_Hpi_2025-08-31 01:21:06.505856.pickle'
        UF_fits['HsK'] = d + 'UF_HsK_2025-08-31 05:04:59.545173.pickle'

    '''Oct2025_Forced_Disp = False
    if Oct2025_Forced_Disp == True:
        Fp_fits['HsK'] = d + 'Fp_HsK_2025-10-01 18:16:03.050048.pickle'
        SF_fits['HsK'] = d + 'SF_HsK_2025-10-01 19:16:50.133132.pickle'
        SFp_fits['HsK'] = d + 'SFp_HsK_2025-10-01 19:57:02.063159.pickle'
    '''
    
    Oct2025_EpsFix = False
    if Oct2025_EpsFix == True: 
        Fp_fits['Hpi'] = d + 'Fp_Hpi_2025-10-21 11:19:40.826680.pickle'
        Fp_fits['HsK'] = d + 'Fp_HsK_2025-10-22 11:59:00.096229.pickle'
        SF_fits['HsK'] = d + 'SF_HsK_2025-10-23 07:27:57.901590.pickle'
        SFp_fits['HsK'] = d + 'SFp_HsK_2025-10-23 20:17:46.862282.pickle'
        

    Nov2025_3ptTwistPriors = True
    if Nov2025_3ptTwistPriors == True:
        F_fits['Hpi'] = d + 'F_Hpi_2025-11-21 18:35:50.574444.pickle'
        F_fits['HsK'] = d + 'F_HsK_2025-11-23 08:43:12.156205.pickle'

        Fp_fits['Hpi'] = d + 'Fp_Hpi_2025-11-22 00:09:34.521157.pickle'
        Fp_fits['HsK'] = d + 'Fp_HsK_2025-11-23 10:40:41.063590.pickle'

        SF_fits['Hpi'] = d + 'SF_Hpi_2025-11-22 00:57:29.145620.pickle'
        SF_fits['HsK'] = d + 'SF_HsK_2025-11-25 19:53:56.262608.pickle'

        SFp_fits['Hpi'] = d + 'SFp_Hpi_2025-11-22 21:34:57.106737.pickle'
        SFp_fits['HsK'] = d + 'SFp_HsK_2025-11-25 21:02:03.477248.pickle'

        UF_fits['Hpi'] = d + 'UF_Hpi_2025-11-23 02:38:54.064933.pickle'
        UF_fits['HsK'] = d + 'UF_HsK_2025-11-26 17:01:39.677199.pickle'

    chi2_scaling = False
    if chi2_scaling == True:
        F_fits['Hpi'] = d + 'F_Hpi_2025-12-03 17:46:58.004450.pickle'
        F_fits['HsK'] = d + 'F_HsK_2025-12-04 10:02:16.226320.pickle'

        Fp_fits['Hpi'] = d + 'Fp_Hpi_2025-12-04 02:55:51.676758.pickle'
        Fp_fits['HsK'] = d + 'Fp_HsK_2025-12-04 21:16:26.243763.pickle'

        SF_fits['Hpi'] = d + 'SF_Hpi_2025-12-04 04:12:55.716937.pickle'
        SF_fits['HsK'] = d + 'SF_HsK_2025-12-05 10:46:35.805842.pickle'

        SFp_fits['Hpi'] = d + 'SFp_Hpi_2025-12-04 05:34:23.774362.pickle'
        SFp_fits['HsK'] = d + 'SFp_HsK_2025-12-06 11:16:38.959497.pickle'

        UF_fits['Hpi'] = d + 'UF_Hpi_2025-12-04 08:31:36.634354.pickle'
        UF_fits['HsK'] = d + 'UF_HsK_2025-12-06 16:00:01.814763.pickle'

    ### Grouping together
    All_fits = {}
    All_fits['F'] = F_fits
    All_fits['Fp'] = Fp_fits
    All_fits['SF'] = SF_fits
    All_fits['SFp'] = SFp_fits
    All_fits['UF'] = UF_fits

    return All_fits[ensemble][decay_channel]


