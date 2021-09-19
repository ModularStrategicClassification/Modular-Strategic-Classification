import torch
import cvxpy as cp


"""Cost Functions"""


# Quadratic cost
def quad_cost_cvxpy_batched(X, R, scale=1):
    return scale * cp.square(cp.norm(X - R, 2, axis=1))


def quad_cost_cvxpy_not_batched(x, r, scale=1):
    return scale * cp.sum_squares(x - r)


def quad_cost_torch(X, R, scale=1):
    return scale * torch.sum((X - R) ** 2, dim=1)


# Linear cost
def linear_cost_cvxpy_batched(X, R, v, epsilon=0.05, scale=1):
    return scale * (epsilon * cp.square(cp.norm(X - R, 2, axis=1)) + (1 - epsilon) * cp.pos((X - R) @ v))


def linear_cost_cvxpy_not_batched(x, r, v, epsilon=0.05, scale=1):
    return scale * (epsilon * cp.sum_squares(x - r) + (1 - epsilon) * cp.pos((x - r) @ v))


def linear_cost_torch(X, R, v, epsilon=0.05, scale=1):
    return scale * (epsilon * torch.sum((X - R) ** 2, dim=1) + (1 - epsilon) * torch.clamp((X - R) @ v, min=0))


SCMP_cost_functions = {
    "quad": {"cvxpy_batched": quad_cost_cvxpy_batched, "cvxpy_not_batched": quad_cost_cvxpy_not_batched, "torch": quad_cost_torch},
    "linear": {"cvxpy_batched": linear_cost_cvxpy_batched, "cvxpy_not_batched": linear_cost_cvxpy_not_batched, "torch": linear_cost_torch}
}


"""Loss Functions"""


def hinge_loss(Y, Y_pred):
    return torch.mean(torch.clamp(1 - Y_pred * Y, min=0))
