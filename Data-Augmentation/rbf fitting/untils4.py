def generation_lstm_con_data(path_list, name, task_list, device):
    count = 0
    for path in path_list:
        if path == path_list[0]:
            cq = '_cq'
        else:
            cq = '_cq_noise'
        for t in task_list:
            print(t)
            if count == 0:
                df = pd.read_csv(path.format(name, t +cq),  header=1)
                df_EEG = df[3560:34280].reset_index(drop = True).values
                eeg_input = torch.tensor(df_EEG.reshape(256, 120, 32),
                                        dtype = float, device = device)
                #df_fft = df[fft_col][500:37940].dropna().reset_index(drop = True).values
                #fft_input = torch.tensor(df_fft.reshape(180, 160, 13),
                #                         dtype = float, device = device)
                
                if t == '10' or t =='20': 
                    output = torch.zeros((256,), device = device, dtype=int)
                elif t == '20': 
                    output = torch.cat([output, torch.zeros((256,), device = device, dtype = int) + 1])
                else: 
                    output = torch.cat([output, torch.zeros((256,), device = device, dtype = int) + 2])
            else:
                df = pd.read_csv(path.format(name, t + cq))
                df_EEG = df[3560:34280].reset_index(drop = True).values
                eeg_input = torch.cat([eeg_input, torch.tensor(df_EEG.reshape(256, 120, 32),
                                                            dtype = float, device = device)])
                
                #df_fft = df[fft_col][500:37940].dropna().reset_index(drop = True).values
                #fft_input = torch.cat([fft_input, torch.tensor(df_fft.reshape(180, 160, 13),
                #                                               dtype = float, device = device)])
                
                if t == '10' or t == '1back':
                    output = torch.cat([output, torch.zeros((256,), device = device, dtype = int)])
                elif t == '20': 
                    output = torch.cat([output, torch.zeros((256,), device = device, dtype = int) + 1])
                else: 
                    output = torch.cat([output, torch.zeros((256,), device = device, dtype = int) + 2])
                    
            count += 1
    
    return eeg_input, output