class PSO_update(nn.Module):
    def __init__(self, input_data, linear_weights, centers, sigmas, device):
        super(PSO_update, self).__init__()

        self.input_data = input_data
        self.linear_weights = linear_weights
        self.centers = centers
        self.sigmas = sigmas
        self.device = device

    def rbfnn(self, weights):
        R = torch.exp(-1 * (torch.pow((self.input_data - self.centers), 2) / \
                            (torch.pow(self.sigmas, 2))))
        output = torch.mm(weights, R)

        return output  # size = (1, num_rbf)

    def initial_velocity(self):
        rbf_output = (-2 * (self.input_data - self.centers) / torch.pow(self.sigmas, 2)) * (
            torch.exp(-1 * (torch.pow((self.input_data - self.centers), 2) / (torch.pow(self.sigmas, 2)))))
        rbf_grad = torch.mm(self.linear_weights, rbf_output)

        return rbf_grad

    # fitness_function
    def cost_function(self, weights, target):
        pred = self.rbfnn(weights)
        error = torch.mean(torch.pow(pred - target, 2))

        return error

    def update_position(self, particle, velocity):
        new_particle = particle + velocity

        return new_particle

    def update_velocity(self, particle, velocity, pbest, gbest, lr):
        # initial velocity
        r1 = lr
        r2 = lr

        new_velocity = velocity + r1 * (pbest - particle) + r2 * (gbest - particle)

        return new_velocity

    def train(self, target, epochs, loss_th, lr):
        particles = torch.randn((100, self.linear_weights.size(0), self.linear_weights.size(1)), dtype=float,
                                device=self.device)
        velocity = torch.zeros_like(particles, device=self.device)
        pbest_position = particles  # (100, 1, rbf_num)
        pbest_fitness = torch.tensor([self.cost_function(particles[i], target) for i in range(100)], device=self.device)

        gbest_index = torch.argmin(pbest_fitness)
        gbest = pbest_position[gbest_index].expand(100, self.linear_weights.size(0),
                                                   self.linear_weights.size(1))  # (1, rbf_num)
        print(gbest.size())
        for epoch in range(epochs):
            print(epoch, end=" ")
            loss = self.cost_function(gbest[0], target)
            if loss <= loss_th:
                break
            else:
                velocity = self.update_velocity(particles, velocity, pbest_position, gbest, lr)
                particles = self.update_position(particles, velocity)

                particle_fitness = torch.tensor([self.cost_function(particles[n], target) for n in range(100)],
                                                device=self.device)
                for n in range(100):
                    if particle_fitness[n] < pbest_fitness[n]:
                        pbest_position[n] = particles[n]
                        pbest_fitness[n] = particle_fitness[n]
                gbest_index = torch.argmin(pbest_fitness)
                gbest = pbest_position[gbest_index].expand(100, self.linear_weights.size(0),
                                                           self.linear_weights.size(1))

        # Print the results
        print('Global Best Position: ', gbest)
        print('Best Fitness Value: ', min(pbest_fitness))
        print('Average Particle Best Fitness Value: ', torch.mean(pbest_fitness))
        print('Number of Generation: ', epoch)

        gbest[0]

        plt.figure(figsize=(20, 10))
        plt.plot(self.input_data.cpu().detach().numpy(), target[0].cpu().detach().numpy())
        plt.plot(self.input_data.cpu().detach().numpy(), self.rbfnn(gbest[0])[0].cpu().detach().numpy())
        plt.show()

        return gbest[0]