def arrange_target(data, context):
    '''
    Arrange a single time series into overlapping short sequences.
    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        target[:, i, :] = data[start+1:end+1]
    return target.detach()