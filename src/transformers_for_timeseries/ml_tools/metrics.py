import torch

def trans_scinet_loss(
        possible_answer: torch.Tensor, 
        a_corr: torch.Tensor, 
        mean: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 0.003,
        truncate_input: bool = False,
        ) -> torch.Tensor:
    if truncate_input:
        a_corr = a_corr[:, :possible_answer.shape[1]]

    recon_loss = torch.nn.MSELoss()(possible_answer.squeeze(), a_corr.squeeze())
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + beta * kld_loss, kld_loss, recon_loss


def trans_scinet_reconstruction_error(
        possible_answer: torch.Tensor, 
        a_corr: torch.Tensor, 
        mean: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 0.003,
        truncate_input: bool = False,
        ) -> torch.Tensor:
    if truncate_input:
        a_corr = a_corr[:, :possible_answer.shape[1]]

    ### Build the MAPE manually
    epsilon = 1e-8  # small constant to avoid division by zero
    abs_diff = torch.abs(possible_answer - a_corr)
    denominator = a_corr.clamp(min=epsilon)
    recon_cost = torch.mean(abs_diff / denominator, dim=-1)
    return recon_cost