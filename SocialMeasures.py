from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


class AbstractSocialMeasure(ABC):
    @abstractmethod
    def get_loss_term(self, X, Y, X_opt, Y_pred, w, b):
        pass

    def init_fit_train(self):
        pass

    def init_fit_validation(self):
        pass

    def begin_epoch(self, epoch, validation):
        pass

    def begin_batch(self, epoch, batch, Xbatch, Ybatch):
        pass

    def end_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        pass

    def get_batch_info(self, epoch, batch):
        return ""

    def begin_validation(self, epoch):
        pass

    def begin_validation_batch(self, epoch, batch, Xbatch, Ybatch):
        pass

    def end_validation_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        pass

    def get_validation_info(self, epoch):
        return ""

    def end_epoch(self, epoch):
        pass

    def save(self, path, model_name):
        pass

    def load(self, path, model_name):
        pass


class Utility(AbstractSocialMeasure):
    def __init__(self, reg, cost_fn_torch, cost_const_kwargs, train_slope=1, eval_slope=5):
        self.reg = reg

        self.train_slope = train_slope
        self.eval_slope = eval_slope
        if cost_fn_torch is None:
            raise ValueError("User must supply a PyTorch version of the cost function to use the utility social measure.")
        self.cost_fn = cost_fn_torch
        self.cost_const_kwargs = cost_const_kwargs

    def calc_utility(self, X, X_opt, Y_pred, requires_grad=False):
        slope = self.train_slope if requires_grad else self.eval_slope
        with torch.set_grad_enabled(requires_grad):
            # TODO: Why are we using the approximation of sign instead of torch.sign()?
            gain = 0.5 * (torch.sqrt((slope * Y_pred + 1) ** 2 + 1) - torch.sqrt((slope * Y_pred - 1) ** 2 + 1))
            cost = self.cost_fn(X_opt, X, **self.cost_const_kwargs)
            utility = torch.mean(gain - cost)
        return utility

    def get_loss_term(self, X, Y, X_opt, Y_pred, w, b):
        return -self.reg * self.calc_utility(X, X_opt, Y_pred, requires_grad=True)

    def init_fit_train(self):
        self.train_utilities = []

    def init_fit_validation(self):
        self.validation_utilities = []

    def begin_epoch(self, epoch, validation):
        self.train_utilities.append([])

    def end_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        utility = self.calc_utility(Xbatch, Xbatch_opt, Ybatch_pred, requires_grad=False)
        self.train_utilities[epoch].append(utility)

    def get_batch_info(self, epoch, batch):
        return f"utility: {self.train_utilities[epoch][batch]:3.5f}"

    def begin_validation(self, epoch):
        self.validation_utilities.append([])

    def end_validation_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        utility = self.calc_utility(Xbatch, Xbatch_opt, Ybatch_pred, requires_grad=False)
        self.validation_utilities[epoch].append(utility)

    def get_validation_info(self, epoch):
        return f"utility: {np.mean(self.validation_utilities[epoch]):3.5f}"


class Burden(AbstractSocialMeasure):
    def __init__(self, reg, x_dim, score_fn, cost_fn_not_batched, cost_fn_torch, cost_const_kwargs, x_lower_bound=-10, x_upper_bound=10):
        self.reg = reg

        if cost_fn_torch is None:
            raise ValueError("User must supply a PyTorch version of the cost function to use the burden social measure.")
        self.cost_fn_torch = cost_fn_torch
        self.cost_const_kwargs = cost_const_kwargs

        x = cp.Variable(x_dim)
        r = cp.Parameter(x_dim, value=np.random.randn(x_dim))
        w = cp.Parameter(x_dim, value=np.random.randn(x_dim))
        b = cp.Parameter(1, value=np.random.randn(1))

        target = cost_fn_not_batched(x, r, **cost_const_kwargs)
        constraints = [score_fn(x, w, b) >= 0,
                       x >= x_lower_bound,
                       x <= x_upper_bound]

        objective = cp.Minimize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[r, w, b], variables=[x])

    def calc_burden(self, X, Y, w, b, requires_grad=False):
        with torch.set_grad_enabled(requires_grad):
            X_pos = X[Y == 1]
            if len(X_pos) == 0:
                return 0
            # X_min = argmin_{X' : score(X') >= 0} {cost(X', X_pos)}
            X_min = self.layer(X_pos, w, b)[0]
            # cost(X_min, X_pos) = min_{X' : score(X') >= 0} {cost(X, X_pos)}
            burden = torch.mean(self.cost_fn_torch(X_min, X_pos, **self.cost_const_kwargs))
        return burden

    def get_loss_term(self, X, Y, X_opt, Y_pred, w, b):
        return self.reg * self.calc_burden(X, Y, w, b, requires_grad=True)

    def init_fit_train(self):
        self.train_burdens = []

    def init_fit_validation(self):
        self.validation_burdens = []

    def begin_epoch(self, epoch, validation):
        self.train_burdens.append([])

    def end_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        burden = self.calc_burden(Xbatch, Ybatch, w, b, requires_grad=False)
        self.train_burdens[epoch].append(burden)

    def get_batch_info(self, epoch, batch):
        return f"burden: {self.train_burdens[epoch][batch]:3.5f}"

    def begin_validation(self, epoch):
        self.validation_burdens.append([])

    def end_validation_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        burden = self.calc_burden(Xbatch, Ybatch, w, b, requires_grad=False)
        self.validation_burdens[epoch].append(burden)

    def get_validation_info(self, epoch):
        return f"burden: {np.mean(self.validation_burdens[epoch]):3.5f}"


class Recourse(AbstractSocialMeasure):
    def __init__(self, reg, score_fn):
        self.reg = reg
        self.score_fn = score_fn

    def calc_recourse(self, X, X_opt, w, b, requires_grad=False):
        with torch.set_grad_enabled(requires_grad):
            sig = torch.nn.Sigmoid()
            original_score = self.score_fn(X, w, b)
            is_neg = sig(-original_score)

            opt_score = self.score_fn(X_opt, w, b)
            is_not_able_to_be_pos = sig(-opt_score)

            recourse = 1 - torch.mean(is_neg * is_not_able_to_be_pos)
        return recourse

    def get_loss_term(self, X, Y, X_opt, Y_pred, w, b):
        return self.reg * (1 - self.calc_recourse(X, X_opt, w, b, requires_grad=True))

    def init_fit_train(self):
        self.train_recourses = []

    def init_fit_validation(self):
        self.validation_recourses = []

    def begin_epoch(self, epoch, validation):
        self.train_recourses.append([])

    def end_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        recourse = self.calc_recourse(Xbatch, Xbatch_opt, w, b, requires_grad=False)
        self.train_recourses[epoch].append(recourse)

    def get_batch_info(self, epoch, batch):
        return f"recourse: {self.train_recourses[epoch][batch]:3.5f}"

    def begin_validation(self, epoch):
        self.validation_recourses.append([])

    def end_validation_batch(self, epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, w, b):
        recourse = self.calc_recourse(Xbatch, Xbatch_opt, w, b, requires_grad=False)
        self.validation_recourses[epoch].append(recourse)

    def get_validation_info(self, epoch):
        return f"recourse: {np.mean(self.validation_recourses[epoch]):3.5f}"
