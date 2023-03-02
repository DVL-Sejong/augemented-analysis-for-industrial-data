def rbf_gradient(x, center_list, sigma_list, weight_list):
    rbf_output = (-2 * (x - center_list) / torch.pow(sigma_list, 2)) * (
        torch.exp(-1 * (torch.pow((x - center_list), 2) / (torch.pow(sigma_list, 2)))))
    rbf_grad = torch.mm(weight_list, rbf_output)

    return rbf_grad

def sin_grad(data):
    grad = 5*torch.sin(data*torch.pi/4) +5/4*torch.pi*data*torch.cos(data*torch.pi/4)
    return grad