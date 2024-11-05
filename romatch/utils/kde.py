import torch


# def kde(x, std = 0.1, half = True, down = None):
#     # use a gaussian kernel to estimate density
#     if half:
#         x = x.half() # Do it in half precision TODO: remove hardcoding
#     if down is not None:
#         scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
#     else:
#         scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
#     density = scores.sum(dim=-1)
#     return density


def kde(x, std=0.1, half=True, down=None, batch_size=1024):
    if half:
        x_half = x.half()
    else:
        x_half = x.float()

    num_points = x_half.size(0)
    density = torch.zeros(num_points, device=x_half.device)

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        x_batch = x_half[start:end]

        if down is not None:
            indices = torch.arange(0, num_points, down, dtype=torch.long)
            x_down = x_half[indices]
            distances = torch.cdist(x_batch.float(), x_down, p=2)
        else:
            distances = torch.cdist(x_batch.float(), x_half.float(), p=2)

        scores = (-distances ** 2 / (2 * std ** 2)).exp()
        density[start:end] = scores.sum(dim=-1)

    return density