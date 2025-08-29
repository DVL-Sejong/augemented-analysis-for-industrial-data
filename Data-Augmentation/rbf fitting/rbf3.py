def rbf_gradient(self, input_data, C, S, W): 
    rbf_output = (-2 * (input_data - C) / torch.pow(S, 2)) * \
                    (torch.exp(-1 * (torch.pow((input_data - C), 2) / (torch.pow(S, 2)))))
    rbf_grad = torch.mm(W, rbf_output)
    
    return rbf_grad.reshape(self.in_feature, 1, input_data.size(-1)) # (in_feature, 1, data_num)

def first_grad(self, input_data, target): 
    space = (input_data,)
    ori_grad = torch.gradient(target, spacing = space, dim = 2, edge_order  = 1)
    return ori_grad[0] # (in_feature, 1, data_num)
    
def target_grad(self, input_data, centers, sigmas, weights, ori_grad): 
    true_grad = ori_grad - self.rbf_gradient(input_data, centers, sigmas, weights)
        
    return true_grad # (in_feature, 1, data_num)