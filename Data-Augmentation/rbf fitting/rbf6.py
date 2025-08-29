def rbf_gaussian_derivative_centers(self, input_data): # done
    output = (2 * (input_data - self.centers) / (torch.pow(self.sigma, 2))) * self.rbf_gaussian(input_data)

    return output  # size = (add_rbf_num, data_num)

def rbf_gaussian_derivative_sigma(self, input_data): # done
    output = (2 * torch.pow((input_data - self.centers), 2) / (torch.pow(self.sigma, 3))) * self.rbf_gaussian(input_data)

    return output  # size = (add_rbf_num, data_num)