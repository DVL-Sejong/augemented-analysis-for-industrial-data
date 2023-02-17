def backward_propagation(self, input_data, radial_output, pred, target, target_grad, pred_grad):
    L2_1_error = -2 * (target - pred)
    L2_2_error = -2 * (target_grad - pred_grad)

    len_ = len(L2_2_error)

    # sigma update
    deltaSigma1 = self.rbf_gaussian_derivative_sigma(input_data) * L2_1_error
    deltaSigma1 *= self.linear_layer_weights.reshape(self.add_rbf_number, 1)

    deltaSigma2 = self.rbf_gaussian_derivative_sigma(input_data) * L2_2_error
    deltaSigma2 *= self.L2_F(input_data) * self.linear_layer_weights.reshape(self.add_rbf_number, 1)
    deltaSigma = torch.sum(deltaSigma1, dim=1) + torch.sum(deltaSigma2, dim=1)

    # center update
    deltaCenter1 = self.rbf_gaussian_derivative_centers(input_data) * L2_1_error
    deltaCenter1 *= self.linear_layer_weights.reshape(self.add_rbf_number, 1)

    deltaCenter2 = self.rbf_gaussian_derivative_centers(input_data) * L2_2_error
    deltaCenter2 *= self.L2_F(input_data) * self.linear_layer_weights.reshape(self.add_rbf_number, 1)
    deltaCenter = torch.sum(deltaCenter1, dim=1) + torch.sum(deltaCenter2, dim=1)

    # weight update
    delta_weight1 = torch.sum((radial_output * L2_1_error), dim=1)
    delta_weight1 = delta_weight1.reshape(1, self.add_rbf_number)

    delta_weight2 = torch.sum((self.L2_2_derivateive_weight(input_data, radial_output) * L2_2_error), dim=1)
    delta_weight2 = delta_weight2.reshape(1, self.add_rbf_number)
    delta_weight = delta_weight1 + delta_weight2

    # BP update
    self.linear_layer_weights -= self.lr * delta_weight
    self.radial_layer_centers -= self.lr * deltaCenter.reshape(self.add_rbf_number, 1)
    self.radial_sigma -= self.lr * deltaSigma.reshape(self.add_rbf_number, 1)