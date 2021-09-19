from abc import ABC, abstractmethod
import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class ResponseEvaluator:
    def __init__(self, x_dim, batch_size,
                 score_fn, score_fn_der,
                 cost_fn_batched, cost_fn_not_batched, cost_const_kwargs,
                 train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap):
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.score = score_fn
        self.score_der = score_fn_der
        self.cost_batched = cost_fn_batched
        self.cost_not_batched = cost_fn_not_batched
        self.cost_const_kwargs = cost_const_kwargs
        self.train_slope = train_slope
        self.eval_slope = eval_slope
        self.x_lower_bound = x_lower_bound
        self.x_upper_bound = x_upper_bound
        self.diff_threshold = diff_threshold
        self.iteration_cap = iteration_cap

        self.create_optimization_problem()
        self.create_differentiable_optimization_problem()

    def create_optimization_problem(self):
        self.x = cp.Variable((self.batch_size, self.x_dim))  # The optimized x, to be optimized to the best response of the contestant, x = argmax{sign(score(x')) - cost(r,x')}.
        self.xt = cp.Parameter((self.batch_size, self.x_dim))  # The x value we take the approximation of the convex part f, the linear approximation of f.
        self.r = cp.Parameter((self.batch_size, self.x_dim))  # The original x value to begin with.
        self.w = cp.Parameter(self.x_dim)  # The weight parameter for the score function.
        self.b = cp.Parameter(1)  # The bias parameter for the score function.
        self.slope = cp.Parameter(1)  # The slope of the sign approximation.

        CCP_target = cp.diag(self.x @ self.f_der_batch(self.xt, self.w, self.b, self.slope).T) - self.g_batch(self.x, self.w, self.b, self.slope) - self.cost_batched(self.x, self.r, **self.cost_const_kwargs)
        CCP_objective = cp.Maximize(cp.sum(CCP_target))
        CCP_constraints = [self.x >= self.x_lower_bound, self.x <= self.x_upper_bound]
        self.CCP_problem = cp.Problem(CCP_objective, CCP_constraints)

    def create_differentiable_optimization_problem(self):
        x_grad = cp.Variable(self.x_dim)  # The optimized x.
        r_grad = cp.Parameter(self.x_dim, value=np.random.randn(self.x_dim))  # The original x value to begin with.
        w_grad = cp.Parameter(self.x_dim, value=np.random.randn(self.x_dim))  # The weight parameter for the score function.
        b_grad = cp.Parameter(1, value=np.random.randn(1))  # The bias parameter for the score function.
        f_der_grad = cp.Parameter(self.x_dim, value=np.random.randn(self.x_dim))  # The approximation for the derivative of f(score(x)) with respect to x.

        CCP_target_grad = x_grad @ f_der_grad - self.g(x_grad, w_grad, b_grad, self.train_slope) - self.cost_not_batched(x_grad, r_grad, **self.cost_const_kwargs)
        CCP_objective_grad = cp.Maximize(CCP_target_grad)
        CCP_constraints_grad = [x_grad >= self.x_lower_bound, x_grad <= self.x_upper_bound]
        CCP_problem_grad = cp.Problem(CCP_objective_grad, CCP_constraints_grad)
        self.CCP_layer = CvxpyLayer(CCP_problem_grad, parameters=[r_grad, w_grad, b_grad, f_der_grad], variables=[x_grad])

    def solve_optimization_problem(self, X, w, b, **kwargs):
        # X, w, b, slope are in numpy form. Returns X_opt in numpy form.
        slope = kwargs.pop("slope")

        self.w.value = w
        self.b.value = b
        X_padded = self.pad_X(X, self.batch_size)  # If the passed X is not of size batch_size (which can happen for example at the final batch), we pad it with zeros (and remove the padding before the return).
        self.r.value = X_padded
        self.slope.value = slope
        self.x.value = X_padded  # Initialize the initial x value to X, for the initial approximation location.

        count = 0
        diff = np.inf
        while diff > self.diff_threshold and count < self.iteration_cap:
            count += 1
            self.xt.value = self.x.value  # Update the approximation x to be the last optimal x value.
            self.CCP_problem.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value) / self.batch_size

        X_opt = self.x.value[:len(X)]
        return X_opt

    def solve_differentiable_optimization_problem(self, X, w, b, X_opt, **kwargs):
        # X, w, b, slope, X_opt are in torch form. Returns the optimal X in torch form (while tracking gradients w.r.t w, b).
        f_der = self.f_der_torch(X_opt, w, b, self.train_slope)
        return self.CCP_layer(X, w, b, f_der)[0]

    def optimize_X(self, X, w, b, requires_grad=True, **kwargs):
        X_np = X.numpy()
        w_np = w.detach().numpy()
        b_np = b.detach().numpy()
        kwargs_np = {key: value.detach().numpy() for (key, value) in kwargs.items()}

        slope = self.train_slope if requires_grad else self.eval_slope
        kwargs_np["slope"] = np.full(1, slope)

        X_opt_np = self.solve_optimization_problem(X_np, w_np, b_np, **kwargs_np)
        X_opt = torch.from_numpy(X_opt_np)

        if requires_grad:
            X_opt = self.solve_differentiable_optimization_problem(X, w, b, X_opt, **kwargs)  # Should be the same as X_opt, only the return value of solve_differentiable_optimization_problem should track gradients w.r.t w, b.

        return X_opt

    # Cvxpy versions, non-batched
    def f(self, x, w, b, slope):  # Unused.
        return 0.5 * cp.norm(cp.hstack([1, (slope * self.score(x, w, b) + 1)]), 2)

    def g(self, x, w, b, slope):
        return 0.5 * cp.norm(cp.hstack([1, (slope * self.score(x, w, b) - 1)]), 2)

    def f_der(self, x, w, b, slope):  # Unused.
        return 0.5 * cp.multiply(slope * ((slope * self.score(x, w, b) + 1) / cp.sqrt((slope * self.score(x, w, b) + 1) ** 2 + 1)), self.score_der(x, w, b))

    # Cvxpy versions, batched
    def f_batch(self, x, w, b, slope):  # Unused.
        return 0.5 * cp.norm(cp.vstack([np.ones(x.shape[0]), (slope * self.score(x, w, b) + 1)]), 2, axis=0)

    def g_batch(self, x, w, b, slope):
        return 0.5 * cp.norm(cp.vstack([np.ones((1, x.shape[0])), cp.reshape((slope * self.score(x, w, b) - 1), (1, x.shape[0]))]), 2, axis=0)

    def f_der_batch(self, x, w, b, slope):
        der = 0.5 * slope * ((slope * self.score(x, w, b) + 1) / cp.sqrt((slope * self.score(x, w, b) + 1) ** 2 + 1))
        return cp.reshape(der, (der.shape[0], 1)) @ cp.reshape(self.score_der(x, w, b), (1, x.shape[1]))

    # Torch version of f_der
    def f_der_torch(self, x, w, b, slope):
        der = 0.5 * slope * ((slope * self.score(x, w, b) + 1) / torch.sqrt((slope * self.score(x, w, b) + 1) ** 2 + 1))
        return der.view(-1, 1) @ self.score_der(x, w, b).view(1, -1)

    @staticmethod
    def pad_X(X, batch_size):
        if len(X) == batch_size:
            return X
        elif len(X) > batch_size:
            raise ValueError(f"The passed X is of size {X.shape} while expected batch size is {batch_size}.")

        pad_amount = batch_size - len(X)
        return np.pad(X, ((0, pad_amount), (0, 0)), 'constant', constant_values=(0, 0))


class FlexibleResponseEvaluator(ResponseEvaluator):
    def __init__(self, x_dim, batch_size, score_fn, score_fn_der, cost_fn_batched, cost_fn_not_batched, cost_const_kwargs, train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap):
        super().__init__(x_dim, batch_size, score_fn, score_fn_der, cost_fn_batched, cost_fn_not_batched, cost_const_kwargs, train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap)

    def create_optimization_problem(self, **kwargs):
        self.x = cp.Variable((self.batch_size, self.x_dim))  # The optimized x, to be optimized to the best response of the contestant, x = argmax{sign(score(x')) - cost(r,x')}.
        self.xt = cp.Parameter((self.batch_size, self.x_dim))  # The x value we take the approximation of the convex part f, the linear approximation of f.
        self.r = cp.Parameter((self.batch_size, self.x_dim))  # The original x value to begin with.
        self.w = cp.Parameter(self.x_dim)  # The weight parameter for the score function.
        self.b = cp.Parameter(1)  # The bias parameter for the score function.
        self.slope = cp.Parameter(1)  # The slope of the sign approximation.

        self.v = cp.Parameter(self.x_dim)
        self.rv = cp.Parameter(self.batch_size)
        flexible_cost_kwargs = {"v": self.v, "Rv": self.rv, **self.cost_const_kwargs}

        CCP_target = cp.diag(self.x @ self.f_der_batch(self.xt, self.w, self.b, self.slope).T) - self.g_batch(self.x, self.w, self.b, self.slope) - self.cost_batched(self.x, self.r, **flexible_cost_kwargs)
        CCP_objective = cp.Maximize(cp.sum(CCP_target))
        CCP_constraints = [self.x >= self.x_lower_bound, self.x <= self.x_upper_bound]
        self.CCP_problem = cp.Problem(CCP_objective, CCP_constraints)

    def create_differentiable_optimization_problem(self, **kwargs):
        x_grad = cp.Variable(self.x_dim)  # The optimized x.
        r_grad = cp.Parameter(self.x_dim, value=np.random.randn(self.x_dim))  # The original x value to begin with.
        w_grad = cp.Parameter(self.x_dim, value=np.random.randn(self.x_dim))  # The weight parameter for the score function.
        b_grad = cp.Parameter(1, value=np.random.randn(1))  # The bias parameter for the score function.
        f_der_grad = cp.Parameter(self.x_dim, value=np.random.randn(self.x_dim))  # The approximation for the derivative of f(score(x)) with respect to x.

        v_grad = cp.Parameter(self.x_dim)
        rv_grad = cp.Parameter(1)
        flexible_cost_kwargs = {"v": v_grad, "rv": rv_grad, **self.cost_const_kwargs}

        CCP_target_grad = x_grad @ f_der_grad - self.g(x_grad, w_grad, b_grad, self.train_slope) - self.cost_not_batched(x_grad, r_grad, **flexible_cost_kwargs)
        CCP_objective_grad = cp.Maximize(CCP_target_grad)
        CCP_constraints_grad = [x_grad >= self.x_lower_bound, x_grad <= self.x_upper_bound]
        CCP_problem_grad = cp.Problem(CCP_objective_grad, CCP_constraints_grad)
        self.CCP_layer = CvxpyLayer(CCP_problem_grad, parameters=[r_grad, w_grad, b_grad, f_der_grad, v_grad, rv_grad], variables=[x_grad])

    def solve_optimization_problem(self, X, w, b, **kwargs):
        slope = kwargs.pop("slope")
        v = kwargs.pop("v")

        self.w.value = w
        self.b.value = b
        X_padded = self.pad_X(X, self.batch_size)
        self.r.value = X_padded
        self.v.value = v
        self.rv.value = self.r.value @ self.v.value
        self.slope.value = slope
        self.x.value = X_padded  # Initialize the initial x value to X, for the initial approximation location.

        count = 0
        diff = np.inf
        while diff > self.diff_threshold and count < self.iteration_cap:
            count += 1
            self.xt.value = self.x.value  # Update the approximation x to be the last optimal x value.
            self.CCP_problem.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value) / self.batch_size

        X_opt = self.x.value[:len(X)]
        return X_opt

    def solve_differentiable_optimization_problem(self, X, w, b, X_opt, **kwargs):
        v = kwargs.pop("v")

        f_der = self.f_der_torch(X_opt, w, b, self.train_slope)
        rv = (X @ v).unsqueeze(1)
        return self.CCP_layer(X, w, b, f_der, v, rv)[0]
