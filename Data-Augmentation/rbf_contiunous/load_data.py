def load_data(data_len, add_rbf_num, path):
    data = []
    rbfnumlist = []
    for i in range(data_len):
        data.append(torch.load(path.format(i)))
        rbfnumlist.append(int(data[i].size(1)/ add_rbf_num))
    return data, rbfnumlist