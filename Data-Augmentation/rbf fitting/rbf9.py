def backward_propagation(self, input_data, R, pred, target, target_grad, pred_grad):
        
    L2_1_error = -2 * (target - pred)
    L2_2_error = -2 * (target_grad - pred_grad)

    # updata partial derivative
    deltaSigma1 = self.rbf_gaussian_derivative_sigma(input_data) * L2_1_error                       # (in_feature, add_rbf_num, data_num)
    deltaSigma1 *= self.weights.reshape(self.in_feature, self.add_rbf_num, 1)                   # (in_feature, add_rbf_num, data_num)

    deltaSigma2 = self.rbf_gaussian_derivative_sigma(input_data) * L2_2_error                       # (in_feature, add_rbf_num, data_num)
    deltaSigma2 *= self.L2_F(input_data) * self.weights.reshape(self.in_feature, self.add_rbf_num, 1)    # (in_feature, add_rbf_num, data_num)

    deltaSigma =  torch.sum(torch.sum(deltaSigma1, dim=2), dim = 0) + torch.sum(torch.sum(deltaSigma2, dim=2), dim = 0) # (add_rbf_num) 

    # center partial derivative
    deltaCenter1 = self.rbf_gaussian_derivative_centers(input_data) * L2_1_error
    deltaCenter1 *= self.weights.reshape(self.in_feature, self.add_rbf_num, 1)
    
    deltaCenter2 = self.rbf_gaussian_derivative_centers(input_data) * L2_2_error
    deltaCenter2 *= self.L2_F(input_data) * self.weights.reshape(self.in_feature, self.add_rbf_num, 1)

    deltaCenter =  torch.sum(torch.sum(deltaCenter1, dim=2),dim =0) + torch.sum(torch.sum(deltaCenter2, dim=2), dim = 0) # (add_rbf_num)


    # weight partial derivative
    delta_weight1 = torch.sum((R * L2_1_error), dim=2)        # (in_feature, add_rbf_num)
    delta_weight2 = torch.sum((self.L2_2_derivative_weight(input_data, R) * L2_2_error), dim = 2) # (in_feature, add_rbf_num)
    delta_weight = delta_weight1 + delta_weight2 # (in_feature, add_rbf_num)

    # BP update
    self.weights -= self.lr * delta_weight
    self.centers -= self.lr * deltaCenter.reshape(self.add_rbf_num, 1)
    self.sigma -= self.lr * deltaSigma.reshape(self.add_rbf_num, 1)