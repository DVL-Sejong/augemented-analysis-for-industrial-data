def data_gen(input_):
    output = torch.exp(-1 * (torch.pow(input_ - C, 2) / \
                             (2 * torch.pow(S, 2))))
    return output  # size = (num_rbf, 1)


def data_gradient(x, center_list, sigma_list, weight_list):
    rbf_output = (-1 * (x - center_list) / torch.pow(sigma_list, 2)) * (
        torch.exp(-1 * (torch.pow((x - center_list), 2) / (2 * torch.pow(sigma_list, 2)))))
    rbf_grad = torch.mm(weight_list, rbf_output)

    return rbf_grad


def rbf_gradient(x, center_list, sigma_list, weight_list):
    rbf_output = (-2 * (x - center_list) / torch.pow(sigma_list, 2)) * (
        torch.exp(-1 * (torch.pow((x - center_list), 2) / (torch.pow(sigma_list, 2)))))
    rbf_grad = torch.mm(weight_list, rbf_output)

    return rbf_grad