# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:38:22 2020

@author: capli
"""

def get_epos_path(run_number='R20_07094'):
    '''
    Returns the full path to the epos.  This allows us to have different data
    directories while not having to change the code everytime.
    
    Valid run_numbers are
    07049 ~ 'template'
    07148 ~ Mg doped
    07199 ~ InGaNQW
    07209 ~ AlGaN
    07247 ~ CSR ~ 2
    07248 ~ CSR ~ 2
    07249 ~ CSR ~ 0.5
    07250 ~ CSR ~ 0.1
    '''    
    
    import os 
    
    login_name = os.getlogin().lower().strip()

    ben_wd = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript'
    ben_data_dict = {}
    
    ben_data_dict['R20_07094'] = os.path.join(ben_wd,'R20_07094-v03.epos')
    ben_data_dict['R20_07148'] = os.path.join(ben_wd,'R20_07148-v01.epos')
    ben_data_dict['R20_07199'] = os.path.join(ben_wd,'R20_07199-v03.epos')
    ben_data_dict['R20_07209'] = os.path.join(ben_wd,'R20_07209-v01.epos')
    ben_data_dict['R20_07247'] = os.path.join(ben_wd,'R20_07247.epos')
    ben_data_dict['R20_07248'] = os.path.join(ben_wd,'R20_07248-v01.epos')
    ben_data_dict['R20_07249'] = os.path.join(ben_wd,'R20_07249-v01.epos')
    ben_data_dict['R20_07250'] = os.path.join(ben_wd,'R20_07250-v01.epos')
    ben_data_dict['R20_07104'] = os.path.join(ben_wd,'R20_07104-v01.epos')
    ben_data_dict['R20_07199_redo'] = os.path.join(ben_wd,'R20_07199-v07.epos')
    
    
    
    
    luis_wd = r'C:\Users\lnm\Documents\Div 686\Data'
    luis_data_dict = {}
    
    luis_data_dict['R20_07094'] = os.path.join(luis_wd, '180821_GaN_A71', 'R20_07094-v03.epos')
    luis_data_dict['R20_07148'] = os.path.join(luis_wd, '181210_D315_A74', 'R20_07148-v01.epos')
    luis_data_dict['R20_07199'] = os.path.join(luis_wd, '190406_InGaNQW_A82', 'R20_07199-v03.epos')
    luis_data_dict['R20_07209'] = os.path.join(luis_wd, '190421_AlGaN50p7_A83', 'R20_07209-v01.epos')
    luis_data_dict['R20_07247'] = os.path.join(luis_wd, '190508_GaN_A84', 'R20_07247.epos')
    luis_data_dict['R20_07248'] = os.path.join(luis_wd, '190508_GaN_A84', 'R20_07248-v01.epos')
    luis_data_dict['R20_07249'] = os.path.join(luis_wd, '190508_GaN_A84', 'R20_07249-v01.epos')
    luis_data_dict['R20_07250'] = os.path.join(luis_wd, '190508_GaN_A84', 'R20_07250-v01.epos')
    luis_data_dict['R20_07199_redo'] = os.path.join(luis_wd, '190406_InGaNQW_A82', 'R20_07199-v07.epos')
    
    
    if login_name == 'capli':
        data_dict = ben_data_dict
    elif login_name == 'lnm':
        data_dict = luis_data_dict
    else:
        raise Exception('The following login name was not recognized: {}'.format(login_name)) 
    
    if run_number not in data_dict.keys():
        raise Exception('The following run_number was not recognized: {}'.format(run_number)) 
        
    return data_dict[run_number]
    
