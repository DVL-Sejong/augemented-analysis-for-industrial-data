def predict(self, input_data): # ? 
    rbf_output = torch.exp(-1 * (torch.pow((input_data - self.done_centers), 2) / \
                                    (torch.pow(self.done_sigma, 2))))
    pred = torch.mm(self.done_weights.reshape(self.in_feature, self.add_rbf_num),
                        rbf_output).reshape(self.in_feature, 1, input_data.size(-1))

    return rbf_output, pred

def Loss(self, pred, target, pred_grad, true_grad):
    # value loss + gradient loss 

    return torch.mean(torch.pow(target - pred,2) + torch.pow(true_grad - pred_grad, 2))