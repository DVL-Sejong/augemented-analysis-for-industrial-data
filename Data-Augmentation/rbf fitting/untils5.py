def generation_lstm_data(path, name, task_list, device):
    count = 0
    fft_col = ['POW.Cz.Theta','POW.Cz.Alpha','POW.Cz.BetaL','POW.Cz.BetaH','POW.Cz.Gamma',
           'POW.Fz.Theta','POW.Fz.Alpha','POW.Fz.BetaL','POW.Fz.BetaH','POW.Fz.Gamma',
           'POW.Fp1.Theta','POW.Fp1.Alpha','POW.Fp1.BetaL','POW.Fp1.BetaH','POW.Fp1.Gamma',
           'POW.F7.Theta','POW.F7.Alpha','POW.F7.BetaL','POW.F7.BetaH','POW.F7.Gamma','POW.F3.Theta',
           'POW.F3.Alpha','POW.F3.BetaL','POW.F3.BetaH','POW.F3.Gamma','POW.FC1.Theta','POW.FC1.Alpha',
           'POW.FC1.BetaL','POW.FC1.BetaH','POW.FC1.Gamma','POW.C3.Theta','POW.C3.Alpha','POW.C3.BetaL',
           'POW.C3.BetaH','POW.C3.Gamma','POW.FC5.Theta','POW.FC5.Alpha','POW.FC5.BetaL','POW.FC5.BetaH',
           'POW.FC5.Gamma','POW.FT9.Theta','POW.FT9.Alpha','POW.FT9.BetaL','POW.FT9.BetaH','POW.FT9.Gamma',
           'POW.T7.Theta','POW.T7.Alpha','POW.T7.BetaL','POW.T7.BetaH','POW.T7.Gamma','POW.CP5.Theta',
           'POW.CP5.Alpha','POW.CP5.BetaL','POW.CP5.BetaH','POW.CP5.Gamma','POW.CP1.Theta','POW.CP1.Alpha',
           'POW.CP1.BetaL','POW.CP1.BetaH','POW.CP1.Gamma','POW.P3.Theta','POW.P3.Alpha','POW.P3.BetaL',
           'POW.P3.BetaH','POW.P3.Gamma','POW.P7.Theta','POW.P7.Alpha','POW.P7.BetaL','POW.P7.BetaH',
           'POW.P7.Gamma','POW.PO9.Theta','POW.PO9.Alpha','POW.PO9.BetaL','POW.PO9.BetaH','POW.PO9.Gamma',
           'POW.O1.Theta','POW.O1.Alpha','POW.O1.BetaL','POW.O1.BetaH','POW.O1.Gamma','POW.Pz.Theta',
           'POW.Pz.Alpha','POW.Pz.BetaL','POW.Pz.BetaH','POW.Pz.Gamma','POW.Oz.Theta','POW.Oz.Alpha',
           'POW.Oz.BetaL','POW.Oz.BetaH','POW.Oz.Gamma','POW.O2.Theta','POW.O2.Alpha','POW.O2.BetaL',
           'POW.O2.BetaH','POW.O2.Gamma','POW.PO10.Theta','POW.PO10.Alpha','POW.PO10.BetaL','POW.PO10.BetaH',
           'POW.PO10.Gamma','POW.P8.Theta','POW.P8.Alpha','POW.P8.BetaL','POW.P8.BetaH','POW.P8.Gamma',
           'POW.P4.Theta','POW.P4.Alpha','POW.P4.BetaL','POW.P4.BetaH','POW.P4.Gamma','POW.CP2.Theta',
           'POW.CP2.Alpha','POW.CP2.BetaL','POW.CP2.BetaH','POW.CP2.Gamma','POW.CP6.Theta','POW.CP6.Alpha',
           'POW.CP6.BetaL','POW.CP6.BetaH','POW.CP6.Gamma','POW.T8.Theta','POW.T8.Alpha','POW.T8.BetaL',
           'POW.T8.BetaH','POW.T8.Gamma','POW.FT10.Theta','POW.FT10.Alpha','POW.FT10.BetaL','POW.FT10.BetaH',
           'POW.FT10.Gamma','POW.FC6.Theta','POW.FC6.Alpha','POW.FC6.BetaL','POW.FC6.BetaH','POW.FC6.Gamma',
           'POW.C4.Theta','POW.C4.Alpha','POW.C4.BetaL','POW.C4.BetaH','POW.C4.Gamma','POW.FC2.Theta',
           'POW.FC2.Alpha','POW.FC2.BetaL','POW.FC2.BetaH','POW.FC2.Gamma','POW.F4.Theta','POW.F4.Alpha',
           'POW.F4.BetaL','POW.F4.BetaH','POW.F4.Gamma','POW.F8.Theta','POW.F8.Alpha','POW.F8.BetaL',
           'POW.F8.BetaH','POW.F8.Gamma','POW.Fp2.Theta','POW.Fp2.Alpha','POW.Fp2.BetaL','POW.Fp2.BetaH',
           'POW.Fp2.Gamma']
    
    eeg_col = ['EEG.Cz', 'EEG.Fz', 'EEG.Fp1',  'EEG.F7', 'EEG.F3', 'EEG.FC1', 'EEG.C3','EEG.FC5',
           'EEG.FT9', 'EEG.T7','EEG.CP5',  'EEG.CP1', 'EEG.P3', 'EEG.P7', 'EEG.PO9', 'EEG.O1',
           'EEG.Pz', 'EEG.Oz', 'EEG.O2', 'EEG.PO10', 'EEG.P8', 'EEG.P4', 'EEG.CP2', 'EEG.CP6',
           'EEG.T8', 'EEG.FT10', 'EEG.FC6', 'EEG.C4', 'EEG.FC2', 'EEG.F4', 'EEG.F8', 'EEG.Fp2']
        
    for t in task_list:
        print(t)
        if count == 0:
            df = pd.read_csv(path.format(name, t +'_cq'),  header=1)
            df_EEG = df[3560:65000].reset_index(drop = True).values
            eeg_input = torch.tensor(df_EEG.reshape(512, 120, 32),
                                     dtype = float, device = device)
            #df_fft = df[fft_col][500:37940].dropna().reset_index(drop = True).values
            #fft_input = torch.tensor(df_fft.reshape(180, 160, 13),
            #                         dtype = float, device = device)
            
            if t == '10' or t =='20': 
                output = torch.zeros((512,), device = device, dtype=int)
            elif t == '20': 
                output = torch.cat([output, torch.zeros((512,), device = device, dtype = int) + 1])
            else: 
                output = torch.cat([output, torch.zeros((512,), device = device, dtype = int) + 2])
        else:
            df = pd.read_csv(path.format(name, t + '_cq'))
            df_EEG = df[3560:65000].reset_index(drop = True).values
            eeg_input = torch.cat([eeg_input, torch.tensor(df_EEG.reshape(512, 120, 32),
                                                           dtype = float, device = device)])
            
            #df_fft = df[fft_col][500:37940].dropna().reset_index(drop = True).values
            #fft_input = torch.cat([fft_input, torch.tensor(df_fft.reshape(180, 160, 13),
            #                                               dtype = float, device = device)])
            
            if t == '10' or t == '1back':
                output = torch.cat([output, torch.zeros((512,), device = device, dtype = int)])
            elif t == '20': 
                output = torch.cat([output, torch.zeros((512,), device = device, dtype = int) + 1])
            else: 
                output = torch.cat([output, torch.zeros((512,), device = device, dtype = int) + 2])
                
        count += 1
    
    return eeg_input, output 
