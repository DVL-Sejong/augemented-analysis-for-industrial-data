class RBFnnCausal(nn.Module):
    def __init__(self, rbfnum):
        super(RBFnnCausal, self).__init__()
        self.rbfnum = rbfnum
        self.causalconvlist = nn.ModuleList([self.CausalConv1d(kernel_size=i) for i in self.rbfnum])

    def CausalConv1d(self, kernel_size, in_channels=3, out_channels=1, dilation=1, **kwargs):
        return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=kernel_size,
                         kernel_size=kernel_size, padding=0, dilation=dilation, device=device)

    def forward(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(self.causalconvlist[i](X[i].float()))

        for i in range(len(pred)):
            if i == 0:
                A = pred[i].T
            else:
                A = torch.cat([A, pred[i].T], dim=1)
        A = A.expand(1, A.size(0), A.size(1))

        return A