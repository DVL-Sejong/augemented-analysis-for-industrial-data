def L2_F(self, input_data): #done
    return -2 * (input_data - self.centers) / torch.pow(self.sigma, 2) # (add_rbf_num, data_num)

# partial derivative define

def L2_2_derivative_weight(self, input_data, radial_output):
    return self.L2_F(input_data) * radial_output               # (add_rbf_num, data_num)