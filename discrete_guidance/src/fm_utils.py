import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable, Literal, Dict


def flow_matching_sampling(
    num_samples: int,
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    S: int,
    D: int,
    device: torch.device,
    dt: float = 0.001,
    mask_idx: Optional[int] = None,
    pad_idx: Optional[int] = None,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    cond_denoising_model: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    guide_temp: float = 1.0,
    stochasticity: float = 0,
    use_tag: bool = False,
    batch_size: int = 500,
    argmax_final: bool = True,
    max_t: float = 1.0,
    x1_temp: float = 1.0,
    do_purity_sampling: bool = False,
    purity_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[Dict[int, float]] = None,
    eps: float = 1e-9,
):
    """
    Generates samples using flow matching with optional predictor or predictor-free guidance.
    This is a wrapper function that generates samples in batches for memory efficiency.

    Args:
        num_samples (int): Total number of samples to generate
        denoising_model (nn.Module): The unconditional denoising model
        S (int): Size of the categorical state space
        D (int): Dimensionality of each sample
        device (torch.device): Device to run generation on
        dt (float, optional): Time step size for Euler integration. Defaults to 0.001.
        mask_idx (int, optional): Index used for mask token. If None, uses S-1.
        pad_idx (int, optional): Index used for padding token. If None, no padding is used.
        predictor_log_prob (callable, optional): Function that takes (x, t) and returns log p(y|x,t) for predictor guidance
        cond_denoising_model (callable, optional): Function that takes (x, t) and returns logits for predictor-free guidance
        guide_temp (float, optional): Guidance temperature (1/ \gamma). Lower temperature = stronger guidance. Defaults to 1.0.
        stochasticity (float, optional): Amount of stochastic noise in sampling. Defaults to 0.
        use_tag (bool, optional): Whether to use Taylor-approximated guidance. Defaults to False.
        batch_size (int, optional): Batch size for generation. Defaults to 500.
        argmax_final (bool, optional): Whether to argmax final outputs. Defaults to True.
        max_t (float, optional): Maximum time value for sampling. Defaults to 1.0.
        x1_temp (float, optional): Temperature for x1 prediction logits. Defaults to 1.0.
        do_purity_sampling (bool, optional): Whether to use purity-based sampling. Defaults to False.
        purity_temp (float, optional): Temperature for purity sampling. Defaults to 1.0.
        num_unpadded_freq_dict (dict, optional): Dictionary of frequencies for number of unpadded tokens
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-9.

    Returns:
        np.ndarray: Generated samples of shape (num_samples, D)
    """
    # Check if using predictor guidance
    use_predictor_guidance = predictor_log_prob is not None
    # Check if using predictor-free guidance
    use_predictor_free_guidance = cond_denoising_model is not None
    print(
        f"Generating {num_samples} samples: dt={dt}, "
        f"stochasticity={stochasticity}, "
        f"guide_temp={guide_temp}, "
        f"predictor_guidance={use_predictor_guidance}, "
        f"predictor_free_guidance={use_predictor_free_guidance}, "
        f"use_tag={use_tag}"
    )
    # Adjust batch size if needed
    if batch_size > num_samples:
        batch_size = num_samples

    # Generate samples in batches
    counter = 0
    samples = []
    while True:
        x1 = flow_matching_sampling_masking_euler(
            denoising_model=denoising_model,
            batch_size=batch_size,
            S=S,
            D=D,
            device=device,
            dt=dt,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            predictor_log_prob=predictor_log_prob,
            cond_denoising_model=cond_denoising_model,
            guide_temp=guide_temp,
            stochasticity=stochasticity,
            use_tag=use_tag,
            argmax_final=argmax_final,
            max_t=max_t,
            x1_temp=x1_temp,
            do_purity_sampling=do_purity_sampling,
            purity_temp=purity_temp,
            num_unpadded_freq_dict=num_unpadded_freq_dict,
            eps=eps,
        )
        samples.append(x1)
        counter += batch_size
        print(f"{counter} out of {num_samples} generated")
        if counter >= num_samples:
            break
    # Concatenate and trim to exact number of samples requested
    samples = np.concatenate(samples, axis=0)[:num_samples]
    return samples


@torch.no_grad()
def flow_matching_sampling_masking_euler(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
    S: int,
    D: int,
    device: torch.device,
    dt: float = 0.001,
    mask_idx: Optional[int] = None,
    pad_idx: Optional[int] = None,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    cond_denoising_model: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    guide_temp: float = 1.0,
    stochasticity: float = 0,
    use_tag: bool = False,
    argmax_final: bool = True,
    max_t: float = 1.0,
    x1_temp: float = 1.0,
    do_purity_sampling: bool = False,
    purity_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[Dict[int, float]] = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Generates samples using Euler integration of the discrete flow matching model with optional guidance.

    This function implements the core sampling algorithm for discrete flow matching with masking noise.
    It supports both predictor guidance and predictor-free guidance, and includes options for
    purity-based sampling and padding token handling.

    Args:
        denoising_model: Model that takes (x_t: [B,D], t: [B]) and returns logits [B,D,S]
        batch_size: Number of samples to generate in parallel
        S: Size of categorical state space (vocabulary size)
        D: Dimension of each sample (sequence length)
        device: Device to run generation on
        dt: Time step size for Euler integration
        mask_idx: Token index used for masking. Defaults to S-1
        pad_idx: Optional token index used for padding
        predictor_log_prob: Optional predictor function for guided sampling that takes (x,t)
            and returns log p(y|x,t) of shape [B]
        cond_denoising_model: Optional conditional model for predictor-free guidance that
            takes (x,t) and returns logits [B,D,S]
        guide_temp: Temperature for guidance (1 / \gamma). Lower = stronger guidance
        stochasticity: Amount of stochastic noise in sampling
        use_tag: Whether to use Taylor approximation for predictor guidance
        argmax_final: Whether to use argmax for any remaining masked tokens at end
        max_t: Maximum time value to run sampling
        x1_temp: Temperature for softmax of model logits
        do_purity_sampling: Whether to weight sampling by prediction confidence
        purity_temp: Temperature for purity-based sampling weights
        num_unpadded_freq_dict: Optional dict mapping num unpadded tokens to frequencies
        eps: Small constant for numerical stability

    Returns:
        numpy.ndarray: Generated samples of shape [batch_size, D]
    """
    if mask_idx is None:
        mask_idx = S - 1

    B = batch_size

    # Sample initial xt
    xt = mask_idx * torch.ones((B, D), dtype=torch.long, device=device)

    t = 0.0
    num_steps = int(1 / dt)  # TODO: Use ceil or int?
    mask_one_hot = torch.zeros((S,), device=device)
    mask_one_hot[mask_idx] = 1.0

    # Treat the case where fixed pads should be used
    if pad_idx is not None:
        pad_one_hot = torch.zeros((S,), device=device)
        pad_one_hot[pad_idx] = 1.0

        # If 'num_unpadded_freq_dict' is not None,
        # sample pads for x0 (=xt at time t=0) and pad xt
        # overwriting the current xt
        if num_unpadded_freq_dict is not None:
            xt = sample_pads_for_x0(xt, pad_idx, num_unpadded_freq_dict)

    # 여기 tqdm이 찍히는구나
    # dt = 0.001, num_steps = 1000
    for _ in tqdm(range(num_steps)):
        # Get p(x1 | xt), scaled by temperature
        # This is the unconditional prediction
        # If denoising model trained unconditionally, it doesn't use the cls input
        # If it is trained conditionally, this is the index of the unconditional class
        logits = denoising_model(xt, t * torch.ones((B,), device=device))  # (B, D, S)
        pt_x1_probs = F.softmax(logits / x1_temp, dim=-1)  # (B, D, S)

        # If cls free guidance, also get the conditional prediction for the
        # class we want to guide towards
        # p(x1 | xt, y)
        if cond_denoising_model is not None:
            logits_cond = cond_denoising_model(xt, t * torch.ones((B,), device=device))
            pt_x1_probs_cond = F.softmax(logits_cond / x1_temp, dim=-1)

        # Compute the rates and the probabilities
        # See section F.1.1 in DFM

        # When the current state is masked, compute rates for unmasking
        xt_is_mask = (xt == mask_idx).view(B, D, 1).float()
        R_t = (
            xt_is_mask * pt_x1_probs * ((1 + stochasticity * t) / (1 - t))
        )  # (B, D, S)

        # When fixing the pads, do not allow unmasking to padded states by setting
        # the unmasking rates for transitions going to padded states to zero
        if pad_idx is not None:
            R_t *= 1 - pad_one_hot.view(1, 1, -1)

        if cond_denoising_model is not None:
            # Compute conditional rate
            R_t_cond = (
                xt_is_mask * pt_x1_probs_cond * ((1 + stochasticity * t) / (1 - t))
            )  # (B, D, S)

            # Same as in unconditional case above:
            # When fixing the pads, do not allow unmasking to padded states by setting
            # the unmasking rates for transitions going to padded states to zero
            if pad_idx is not None:
                R_t_cond *= 1 - pad_one_hot.view(1, 1, -1)

        if do_purity_sampling:
            # Get purity weight for each dimension for each batch point
            # Only consider dimensions that are currently masked to be eligible for unmasking
            masked_logits = (
                logits * (xt == mask_idx).view(B, D, 1).float()
                + -1e9 * (xt != mask_idx).view(B, D, 1).float()
            )
            # purity_weights: the value of the highest predicted prob at that dimension
            max_masked_logits = torch.max(masked_logits, dim=-1)[0]  # (B, D)
            purity_weights = torch.softmax(
                max_masked_logits / purity_temp, dim=-1
            )  # (B, D)
            # Dimensions with more masks will be upweighted
            R_t *= purity_weights.view(B, D, 1) * torch.sum(
                xt_is_mask, dim=(1, 2)
            ).view(B, 1, 1)
            if cond_denoising_model is not None:
                # Modify conditional rates with purity weights
                masked_logits_cond = (
                    logits_cond * (xt == mask_idx).view(B, D, 1).float()
                    + -1e9 * (xt != mask_idx).view(B, D, 1).float()
                )
                # purity_weights: the value of the highest predicted prob at that dimension
                max_masked_logits_cond = torch.max(masked_logits_cond, dim=-1)[
                    0
                ]  # (B, D)
                purity_weights_cond = torch.softmax(
                    max_masked_logits_cond / purity_temp, dim=-1
                )  # (B, D)
                R_t_cond *= purity_weights_cond.view(B, D, 1) * torch.sum(
                    xt_is_mask, dim=(1, 2)
                ).view(B, 1, 1)

        # When the current state is not a mask, compute rates for remasking
        # Only makes a difference when stochasticity > 0
        remask_rates = (1 - xt_is_mask) * mask_one_hot.view(1, 1, -1) * stochasticity

        # When fixing the pads, do not allow masking of padded states by setting
        # the masking rate for transitions going out of padded states to zero
        if pad_idx is not None:
            xt_is_pad = (xt == pad_idx).view(B, D, 1).float()
            remask_rates *= 1 - xt_is_pad

        R_t += remask_rates

        # Perform predictor guidance by adjusting the unconditional rates
        if predictor_log_prob is not None:
            R_t = get_guided_rates(
                predictor_log_prob,
                xt,
                t,
                R_t,
                S,
                use_tag=use_tag,
                guide_temp=guide_temp,
            )

        # Perform predictor-free guidance by using both the unconditional and conditional rates
        if cond_denoising_model is not None:
            # First add the remask rates to the conditional rates
            # Note the remask rate is the same for unconditional and conditional
            R_t_cond += remask_rates
            # Perform rate adjustment, note that we scale by the inverse temperature
            # If inverse_guide_temp = 0 (guide_temp = inf), equivalent to unconditional
            # If inverse_guide_temp = 1, (guide_temp = 1), equivalent to conditional
            inverse_guide_temp = 1 / guide_temp
            R_t = torch.exp(
                inverse_guide_temp * torch.log(R_t_cond + eps)
                + (1 - inverse_guide_temp) * torch.log(R_t + eps)
            )

        # Set the diagonal of the rates to negative row sum
        R_t.scatter_(-1, xt[:, :, None], 0.0)
        R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

        # Obtain probabilities from the rates
        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        step_probs.scatter_(
            -1,
            xt[:, :, None],
            (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # Sample the next xt
        try:
            xt = torch.distributions.Categorical(step_probs).sample()  # (B, D)
        except ValueError:
            raise ValueError(
                "ValueError in 'torch.distributions.Categorical(step_probs).sample()', step_probs might not valid."
            )

        t += dt
        if t > max_t:
            break

    # For any state that is still masked, take the argmax of the logits
    # of the final xt
    if argmax_final:
        xt_is_mask = (xt == mask_idx).view(B, D).float()
        logits = denoising_model(xt, t * torch.ones((B,), device=device))  # (B, D, S)
        if cond_denoising_model is not None:
            logits_cond = cond_denoising_model(xt, t * torch.ones((B,), device=device))
            logits = (
                inverse_guide_temp * logits_cond + (1 - inverse_guide_temp) * logits
            )
        xt = torch.argmax(logits, dim=-1) * xt_is_mask + xt * (1 - xt_is_mask)

    return xt.detach().cpu().numpy()


def get_guided_rates(
    predictor_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xt: torch.Tensor,  # Shape: (B, D)
    t: float,
    R_t: torch.Tensor,  # Shape: (B, D, S)
    S: int,
    use_tag: bool = False,
    guide_temp: float = 1.0,
    log_prob_ratio_cutoff: float = 80.0,
) -> torch.Tensor:
    """
    Computes guide-adjusted rates for predictor guidance.

    Implements both exact guidance by computing likelihood ratios for all possible transitions,
    and Taylor-approximated guidance (TAG) using gradients of the predictor.

    Args:
        predictor_log_prob (callable): Function that takes (x,t) and returns log p(y|x,t)
        xt (torch.Tensor): Current states of shape (B, D)
        t (float): Current time
        R_t (torch.Tensor): Unconditional rates of shape (B, D, S)
        S (int): Size of categorical state space
        use_tag (bool, optional): Whether to use Taylor approximation. Defaults to False.
        guide_temp (float, optional): Guidance temperature. Defaults to 1.
        log_prob_ratio_cutoff (float, optional): Maximum value for log ratios. Defaults to 80.

    Returns:
        torch.Tensor: Guide-adjusted rates of shape (B, D, S)
    """
    B, D = xt.shape
    device = xt.device
    t = t * torch.ones((B,), device=device)
    if not use_tag:
        # Exact guidance case
        # log p(y|x=z_t), shape (B,)
        log_prob_xt = predictor_log_prob(xt, t)

        # Get all jump transitions, shape (B*D*S, D)
        xt_jumps = get_all_jump_transitions(xt, S)

        # Get log probs for all transitions
        # Shape: (B*D*S,) -> (B, D, S)
        log_prob_xt_jumps = predictor_log_prob(
            xt_jumps, t.repeat(1, D * S).flatten()
        ).view(B, D, S)

        # Compute log ratios
        # Shape (B, D, S)
        log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)

    else:
        # Taylor-approximated guidance (TAG) case
        # One-hot encode categorical data, shape (B, D, S)
        xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)

        # \grad_{x}{log p(y|x)}(z_t), shape (B, D, S)
        with torch.enable_grad():
            xt_ohe.requires_grad_(True)
            # log p(y|x=z_t), shape (B,)
            log_prob_xt_ohe = predictor_log_prob(xt_ohe, t)
            log_prob_xt_ohe.sum().backward()
            # Shape (B, D, S)
            grad_log_prob_xt_ohe = xt_ohe.grad
        # 1st order Taylor approximation of the log difference
        # Shape (B, D, S)
        log_prob_ratio = grad_log_prob_xt_ohe - (xt_ohe * grad_log_prob_xt_ohe).sum(
            dim=-1, keepdim=True
        )

    # Scale log prob ratio by temperature
    log_prob_ratio /= guide_temp

    # Clamp the log prob ratio to avoid overflow in exp
    log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)
    # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
    prob_ratio = torch.exp(log_prob_ratio)
    # Multiply the reverse rate elementwise with the density ratio
    # Note this doesn't deal with the diagonals
    R_t = R_t * prob_ratio

    if R_t.isnan().any():
        raise ValueError(f"The rate matrix 'R_t' contains NaNs.")

    return R_t


def get_all_jump_transitions(
    xt: torch.Tensor,  # Shape: (B, D)
    S: int,
) -> torch.Tensor:  # Shape: (B*D*S, D)
    """
    Gets all possible single-dimension transitions from current states.

    Creates a tensor containing all possible states that differ from input states
    in exactly one position, for each sequence in the batch.

    Args:
        xt: Current state tensor of shape (batch_size, sequence_length)
        S: Size of categorical state space (number of possible values per position)

    Returns:
        Tensor of shape (batch_size * sequence_length * state_space, sequence_length)
        containing all possible single-token transitions
    """
    B, D = xt.shape
    device = xt.device

    # Create B*D*S copies of input sequences
    # Shape: (B, 1, D) -> (B, D*S, D)
    xt_expand = xt.unsqueeze(1).repeat(1, D * S, 1)
    # Flatten batch and transition dimensions
    # Shape: (B, D*S, D) -> (B*D*S, D)
    xt_expand = xt_expand.view(-1, D)

    # Create indices for all possible transitions
    # Shape: (D*S,) -> (B, D*S) -> (B*D*S,)
    jump_idx = torch.arange(D * S).to(device)
    jump_idx = jump_idx.repeat(B, 1).flatten()

    # Create tensor for states after one transition
    xt_jumps = xt_expand.clone()

    # Calculate which dimension changes for each transition
    # Shape: (B*D*S,)
    jump_dims = jump_idx // S

    # Calculate new value for changed dimension
    # Shape: (B*D*S,)
    jump_states = jump_idx % S

    # Apply transitions by assigning new values at transition dimensions
    # Shape: (B*D*S, D)
    xt_jumps[
        torch.arange(jump_idx.size(0), device=device),
        jump_dims,  # Index the transitioned dimension
    ] = jump_states  # Assign the new state

    return xt_jumps


def flow_matching_loss_masking(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    mask_idx: int,
    reduction: str = "mean",
    pad_idx: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Computes flow matching cross entropy loss for masked denoising.

    Args:
        denoising_model: Model that takes (x,t) and returns logits for each position
        x1: Target sequence tensor of shape (B, D)
        mask_idx: Index representing the mask token
        reduction: Reduction for cross entropy loss - 'none', 'mean', or 'sum'
        pad_idx: Optional index representing padding token, which will be preserved during noising
        loss_mask: Optional boolean mask of shape (B, D) indicating which positions to include in loss
        label_smoothing: Label smoothing parameter for cross entropy loss

    Returns:
        Loss tensor (scalar if reduction is 'mean'/'sum', shape (B,) if 'none')
    """
    B, D = x1.shape
    # Sample random timestep t \in [0,1] for each sequence in batch
    t = torch.rand((B,)).to(x1.device)

    # Sample xt by masking x1 according to t
    xt = sample_xt(x1, t, mask_idx, pad_idx)

    # Get model predictions
    logits = denoising_model(xt, t)  # (B, D, S)

    # Create mask for positions to exclude from loss:
    # - Positions that are not masked in xt (already revealed)
    # - Positions marked False in loss_mask (if provided)
    exclude = xt != mask_idx
    if loss_mask is not None:
        exclude = torch.logical_or(exclude, ~loss_mask)

    # Copy x1 to avoid modifying the input
    x1 = x1.clone()
    # Set excluded positions to -1 so they're ignored by cross entropy loss
    x1[exclude] = -1

    loss = F.cross_entropy(
        logits.transpose(1, 2),
        x1,
        reduction=reduction,
        ignore_index=-1,
        label_smoothing=label_smoothing,
    )
    return loss


def sample_xt(
    x1: torch.Tensor, t: torch.Tensor, mask_idx: int, pad_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Samples a noised state xt by masking x1 according to time t.

    Args:
        x1: Input sequence tensor of shape (B, D)
        t: Time values tensor of shape (B,)
        mask_idx: Index representing the mask token
        pad_idx: Optional index representing padding token that should be preserved

    Returns:
        Noised sequence tensor of shape (B, D)
    """
    B, D = x1.shape

    # Copy input to avoid modifying
    xt = x1.clone()  # (B, D)
    # Sample x_{t} from p_{t|1}(x_{t}|x_{1}) by masking
    mask_dims = torch.rand((B, D)).to(x1.device) < (1 - t[:, None])  # (B, D)
    xt[mask_dims] = mask_idx  # (B, D)

    # In case that pads should stay fixed during noising,
    # set all padded dims in x1 to pads in xt
    if pad_idx is not None:
        padded_dims = x1 == pad_idx  # (B, D)
        xt[padded_dims] = pad_idx  # (B, D)

    return xt


def predictor_loss_masking(
    predictor_log_prob_y: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    y: torch.Tensor,  # Shape: (B, ...)
    x1: torch.Tensor,  # Shape: (B, D)
    mask_idx: int,
    reduction: Literal["mean", "sum"] = "mean",
    pad_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes loss for training a noisy predictor model with masked inputs.

    This function trains a predictor to estimate p(y|x,t) when x is partially masked
    according to time t.

    Args:
        predictor_log_prob_y: Function that takes (y: [B,...], x: [B,D], t: [B])
            and returns log p(y|x,t) of shape [B]
        y: Target values tensor of shape [batch_size, ...]
        x1: Input sequence tensor of shape [batch_size, sequence_length]
        mask_idx: Token index used for masking
        reduction: How to reduce the loss:
            - "mean": Average over batch (default)
            - "sum": Sum over batch
        pad_idx: Optional token index for padding that should be preserved during noising

    Returns:
        Negative log likelihood loss:
            - Scalar if reduction is "mean" or "sum"
            - Shape [batch_size] if reduction is "none"

    Raises:
        ValueError: If reduction is not "mean" or "sum"
    """
    B, D = x1.shape
    # Sample continuous time point
    t = torch.rand((B,)).to(x1.device)

    # Sample xt by masking x1 according to time t
    xt = sample_xt(x1, t, mask_idx, pad_idx)

    # The model outputs logits over number of classes
    if reduction == "mean":
        return -torch.mean(predictor_log_prob_y(y, xt, t))
    elif reduction == "sum":
        return -torch.sum(predictor_log_prob_y(y, xt, t))
    else:
        raise ValueError(
            "Input 'reduction' must be either 'sum' or 'mean', got '{reduction}' instead."
        )


def sample_pads_for_x0(x0, pad_idx, num_unpadded_freq_dict):
    """
    Sample pads for x0 (fully masked) and pad x0 with it.

    Args:
        x0 (torch.tensor): Torch tensor of shape (B, D) holding
            the noisy sampled discrete space values.
        pad_idx (int): Pad index.
        num_unpadded_freq_dict (dict): Discrete distribution of the number
            of unpadded tokens as dictionary mapping the number of unpadded
            tokens to their frequency in the form: {<#unpadded>: <frequency>}
            Example: {1: 10, 2: 5, 3: 3}

    Return:
        (torch.tensor): x0 with entries that have been padded.

    """
    # Extract variables from xt
    B = x0.shape[0]
    D = x0.shape[1]
    device = x0.device

    # Ensure that the dict-keys (i.e. "number of unpadded tokens") are integers
    num_unpadded_freq_dict = {
        int(num_unpadded): freq for num_unpadded, freq in num_unpadded_freq_dict.items()
    }

    # Generate a list with the 'number of unpadded token' values and an array with the associated probabilities
    # only including number of unpadded tokens in [0, D] because there are D tokens so that not more than D
    # tokens can be unpadded.
    num_unpadded_vals = [
        num_unpadded_val
        for num_unpadded_val in num_unpadded_freq_dict.keys()
        if num_unpadded_val <= D
    ]
    freq_num_unpadded = np.array(
        [num_unpadded_freq_dict[num_token] for num_token in num_unpadded_vals]
    )
    sum_freq_num_unpadded = np.sum(freq_num_unpadded)
    if 0 == sum_freq_num_unpadded:
        err_msg = f"Cannot compute 'number of unpadded tokens' probabilities because the 'number of unpadded tokens' frequencies sum to zero."
        raise ValueError(err_msg)
    elif sum_freq_num_unpadded < 0:
        err_msg = f"Cannot compute 'number of unpadded tokens' probabilities because the 'number of unpadded tokens' frequencies sum to something less than zero!"
        raise ValueError(err_msg)
    else:
        prob_num_unpadded = freq_num_unpadded / np.sum(
            sum_freq_num_unpadded
        )  # Normalize

    ## Sample the number of tokens
    # Step 1: Draw single sample (n=1) from Multinomial with the 'number of token' probabilities
    #         (prob_num_unpadded) for each batch point (size=B).
    #         np.random.multinomial returns a one-hot vector of len(pvals) per batch point.
    #         Argmax of each one hot vector returns a categorical corresponding to the index a
    #         certain 'number of token' in num_token_vals.
    num_unpadded_index_samples = np.argmax(
        np.random.multinomial(
            n=1, pvals=prob_num_unpadded, size=B
        ),  # (B, len(prob_num_unpadded))
        axis=-1,
    )  # (B, )

    # Step 2: Extract the 'number of unpadded' tokens corresponding to the sampled indices
    num_unpadded_samples = [
        num_unpadded_vals[num_unpadded_index]
        for num_unpadded_index in num_unpadded_index_samples
    ]  # '(B,)'

    ## Generate a 2D torch tensor of shape (B, D) that has True for the dimensions (second tensor axis)
    ## that should be padded for each point in the batch (first tensor axis)
    padded_dims = torch.ones((B, D), dtype=torch.bool, device=device)  # (B, D)
    for batch_pnt_index, num_unpadded_sample in enumerate(num_unpadded_samples):
        # 'padded_dims' is initialized with True entries, thus for the current batch point
        # (first tensor-axis) set the entries to False that should not be padded.
        # The non-added entries are the first 'num_unpadded_sample' ones, while the
        # tokens afterwards should be padded.
        padded_dims[batch_pnt_index, :num_unpadded_sample] = False

    # Pad the dimensions in xt that should be padded (for each batch point)
    # Remark: If we do not use xt.clone(), we get the following two warnings:
    #         (a) UserWarning: Use of index_put_ on expanded tensors is deprecated.
    #           Please clone() the tensor before performing this operation.
    #           This also applies to advanced indexing e.g. tensor[indices] = tensor
    #           (Triggered internally at ../aten/src/ATen/native/TensorAdvancedIndexing.cpp:716.)
    #         (b) UserWarning: Use of masked_fill_ on expanded tensors is deprecated.
    #           Please clone() the tensor before performing this operation.
    #           This also applies to advanced indexing e.g. tensor[mask] = scalar
    #           (Triggered internally at ../aten/src/ATen/native/TensorAdvancedIndexing.cpp:1914.)
    x0_padded = x0.clone()
    x0_padded[padded_dims] = pad_idx

    return x0_padded
