from torch.optim.lr_scheduler import LambdaLR

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a scheduler with linear warmup and optional decay.
        - Linearly increase learning rate for warmup_steps
        - Then linearly decrease learning rate to 0 over remaining steps.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): How many steps to linearly warm up.
        total_steps (int): Total number of training steps.
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)
