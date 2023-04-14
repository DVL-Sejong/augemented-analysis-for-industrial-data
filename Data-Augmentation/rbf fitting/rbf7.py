def add_rbf_parameter(self, input_data, error):
    find_index_input = input_data.clone().detach()
    find_index_error = error.clone().detach()
    
    find_weight = error.clone().detach()
    find_sigma = error.clone().detach()
    
    center_index_list = []

    for i in range(self.add_rbf_num * (self.change_time + 1)):
        index_ = torch.argmax(torch.sum(torch.abs(find_index_error), dim = 0)).cpu().detach().tolist()

        find_index_error[:,:,index_] = 0
        center_index_list.append(index_)

    center_index_list = torch.tensor(center_index_list, device=device)
    initcenter = torch.index_select(find_index_input, 0, center_index_list)[-self.add_rbf_num:].reshape(self.add_rbf_num,1)
    initweight = torch.index_select(find_weight, 2, center_index_list)[:,:,-self.add_rbf_num:].reshape(self.in_feature, self.add_rbf_num)


    sigma_list = []
    dft = torch.log(torch.abs(torch.fft.fft(find_sigma).real))
    
    dft = (torch.abs(dft / torch.max(dft))**-1)
    for k in center_index_list:
        sigma_list.append(torch.mean(dft[:,:,k]).reshape(1))
    initsigma = torch.cat(sigma_list)[-self.add_rbf_num:].reshape(self.add_rbf_num,1)

    return initcenter, initweight, initsigma