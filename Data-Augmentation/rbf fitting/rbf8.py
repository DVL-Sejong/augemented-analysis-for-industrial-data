def change_init(self, na):
    if na == 1:
        loss_list = self.train_loss_list[-self.change_th:]
        if self.number > self.change_th and max(loss_list) == min(loss_list):
            self.change_time += 1
        elif self.number > self.change_th and loss_list[0] < loss_list[1] and loss_list[1] < loss_list[2]:
            self.change_time += 1
        else:
            self.change_time = 0
    else:
        self.change_time += 1

def best_forward(self, input_data, best_center, best_sigma, best_weight): # ?
    rbf_output = torch.exp(-1 * (torch.pow((input_data - best_center), 2) / \
                                    (torch.pow(best_sigma, 2))))
    pred = torch.mm(best_weight.reshape(self.in_feature, self.add_rbf_num), 
                    rbf_output).reshape(self.in_feature, 1, input_data.size(-1))

    return rbf_output, pred
