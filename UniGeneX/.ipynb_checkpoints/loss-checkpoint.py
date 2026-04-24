import math
import torch
import torch.nn.functional as F




def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()

def criterion_neg_log_poisson(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Poisson distribution
    """
    if mask is not None:
        input = input[mask]
        target = target[mask]
    else:
        pass
    lamb = torch.exp(input)
    Pois_var = torch.exp(target)
    return (lamb - Pois_var * input).mean()
    

def criterion_neg_log_poisson_additive(
    input: torch.Tensor, input_NN: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Poisson distribution
    """
    if mask is not None:
        input = input[mask]
        target = target[mask]
        input_NN = input_NN[mask]
    else:
        pass
    lamb = torch.exp(input.data) + torch.exp(input_NN)
    log_lamb = torch.log(lamb)
    Pois_var = torch.exp(target)
    return (lamb - Pois_var * log_lamb).mean()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    loss = torch.abs(input - target) / (target + 1e-6)
    return loss.mean()


def smooth_l1_loss(prediction, target, beta1=1.0, beta2=0.1):
    """
    Smooth L1 Loss function.
    
    Args:
        prediction (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.
        beta (float): Smoothing parameter. Default is 1.0.
    
    Returns:
        torch.Tensor: Smooth L1 Loss.
    """
#     diff0 = torch.abs(prediction - target)
    diff = torch.where((diff0 > beta2), 2 * torch.sqrt(diff0) * math.sqrt(beta1), diff0)
#     diff = torch.where((diff0 < beta1) & (diff0 > beta2), 2 * torch.sqrt(diff0) * math.sqrt(beta1), diff0)
#     diff = torch.where(diff0 > beta1, diff0 + beta1, diff)
    diff = torch.where(diff0 <= beta2, math.sqrt(beta1) * diff0 / math.sqrt(beta2) + math.sqrt(beta2 * beta1), diff)
    loss = diff
#     loss = 0.5 * torch.log(diff0 + 1.1)
    
#     diff0 = torch.abs(prediction - target)
#     diff = torch.where((diff0 < beta1), 0.5 * diff0 ** 2 / beta1, diff0)
#     loss = torch.where(diff0 >= beta1, torch.log(diff0) + beta1 / 2 - math.log(beta1), diff)
    return loss.mean()


# import math
# def smooth(diff, beta1 = 1., beta2 = .1):
#     diff0 = torch.abs(diff)
#     diff = torch.where((diff0 < beta1) & (diff0 > beta2), 2 * torch.sqrt(diff0) * math.sqrt(beta1), diff0)
#     diff = torch.where(diff0 > beta1, diff0 + beta1, diff)
#     diff = torch.where(diff0 <= beta2, math.sqrt(beta1) * diff0 / math.sqrt(beta2) + math.sqrt(beta2 * beta1), diff)
#     return diff


# import matplotlib.pyplot as plt
# aa = torch.from_numpy(np.arange(-2,2,0.01))
# plt.plot(aa, smooth(aa))


# def criterion_neg_log_poisson(
#     input: torch.Tensor, target: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Compute the negative log-likelihood of Poisson distribution
#     """
#     lamb = torch.exp(input)
#     Pois_var = torch.exp(target)
#     return (lamb - Pois_var * input)
    
    
# import matplotlib.pyplot as plt
# aa = torch.from_numpy(np.arange(-5,5,0.01))
# fig, ax = plt.subplots(1,1, figsize = (8,3))
# ax.plot(aa, criterion_neg_log_poisson(aa, torch.zeros_like(aa)))
# plt.show()
# plt.close()