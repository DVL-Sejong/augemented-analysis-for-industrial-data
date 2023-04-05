def plot_train(self, input_data, best_pred): #done
    fig, ax = plt.subplots(1, 3, figsize = (30, 5))
    for i in range(self.in_feature):
        ax[i].plot(input_data.cpu().detach().numpy(), self.target[i][0].cpu().detach().numpy())
        ax[i].plot(input_data.cpu().detach().numpy(), best_pred[i][0].cpu().detach().numpy())
    plt.show()