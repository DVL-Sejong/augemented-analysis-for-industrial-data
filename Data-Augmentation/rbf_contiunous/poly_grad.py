def rbf_gradient(x, center_list, sigma_list, weight_list):
    rbf_output = (-2 * (x - center_list) / torch.pow(sigma_list, 2)) * (
        torch.exp(-1 * (torch.pow((x - center_list), 2) / (torch.pow(sigma_list, 2)))))
    rbf_grad = torch.mm(weight_list, rbf_output)

    return rbf_grad


a = np.arange(0, 10, 0.05)

y = 2 * a ** 3 - 16 * a ** 2 + 2 * a - 10

target3 = torch.tensor(y, device=device).reshape((1, 200))
input_2 = torch.tensor(a, device=device)


def poly_grad(data):
    grad = 6 * torch.pow(data, 2) - 32 * data + 2
    return grad


poly_gr = poly_grad(input_2)