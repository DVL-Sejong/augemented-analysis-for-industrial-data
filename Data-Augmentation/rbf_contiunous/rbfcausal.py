class RBFnnCausal(nn.Module):
    def __init__(self, rbfnum, context):
        super(RBFnnCausal, self).__init__()
        self.rbfnum = rbfnum
        self.context = context
        self.causalconvlist = nn.ModuleList([self.CausalConv1d(kernel_size=i) for i in self.rbfnum])

    def CausalConv1d(self, kernel_size, in_channels=3, out_channels=1, dilation=1, **kwargs):
        return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=kernel_size,
                         kernel_size=kernel_size, padding=0,
                         dilation=dilation, device=device)

    def arrange_input(self, data, context):
        '''
        Arrange a single time series into overlapping short sequences.
        Args:
          data: time series of shape (T, dim).
          context: length of short sequences.
        '''
        assert context >= 1 and isinstance(context, int)
        input = torch.zeros(len(data) - context, context, data.shape[1],
                            dtype=torch.float32, device=data.device)
        for i in range(context):
            start = i
            end = len(data) - context + i
            input[:, i, :] = data[start:end]

        return input.detach()

    def forward(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(self.causalconvlist[i](X[i].float()))

        for i in range(len(pred)):
            if i == 0:
                data = pred[i].T
            else:
                data = torch.cat([data, pred[i].T], dim=1)
        data = data.expand((1, data.size(0), data.size(1)))
        data = self.arrange_input(data[0], self.context)

        return data