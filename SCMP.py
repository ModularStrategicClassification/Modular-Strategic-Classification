import math
import os
import time
import warnings

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from CommonFunctions import *
from ResponseEvaluators import *
from SocialMeasures import *


class SCMP(torch.nn.Module):
    def __init__(self, x_dim, batch_size, loss_fn=hinge_loss,                                                                   # Standard machine learning parameters.
                 cost_fn="quad", cost_fn_not_batched=None, cost_fn_torch=None, cost_const_kwargs=None,                          # Cost function variations & const kwargs.
                 score_fn=None, score_fn_der=None,                                                                              # Score fn - linear and differential parts.
                 utility_reg=0, burden_reg=0, recourse_reg=0, social_measure_dict=None,                                         # Social measures (three defaults and dict).
                 train_slope=1, eval_slope=5, x_lower_bound=-10, x_upper_bound=10, diff_threshold=0.001, iteration_cap=100,     # CCP and cvxpy parameters.
                 strategic=True):
        torch.manual_seed(0)
        np.random.seed(0)
        super().__init__()

        self.x_dim = x_dim
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.train_slope = train_slope
        self.eval_slope = eval_slope
        self.strategic = strategic

        # The model's parameters, assuming a linear score function.
        self.w = torch.nn.parameter.Parameter(math.sqrt(1 / x_dim) * (1 - 2 * torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(1 - 2 * torch.rand(1, dtype=torch.float64, requires_grad=True))

        # If the user does not supply the model with const kwargs, we set it to be an empty dict (so that the default value of cost_const_kwargs is immutable).
        if cost_const_kwargs is None:
            cost_const_kwargs = {}
        self.cost_const_kwargs = cost_const_kwargs

        # If the user passes a string for the cost function, we check the cost function dictionary for it.
        if isinstance(cost_fn, str):
            if cost_fn not in SCMP_cost_functions:
                raise ValueError(f"The passed cost function is not recognized by SCMP, recognized functions are {SCMP_cost_functions.keys()}.")

            if cost_fn_not_batched is not None or cost_fn_torch is not None:
                cost_fn_string_and_function_warning = "User supplied SCMP with both a string cost and a cost function (either not batched or torch version), overriding the supplied cost function with the cost function referred to by the string."
                warnings.warn(cost_fn_string_and_function_warning)

            cost_fn_dictionary = SCMP_cost_functions[cost_fn]
            cost_fn = cost_fn_dictionary["cvxpy_batched"]
            cost_fn_not_batched = cost_fn_dictionary["cvxpy_not_batched"]
            cost_fn_torch = cost_fn_dictionary["torch"]

            if cost_fn == "linear" and "v" not in cost_const_kwargs:
                raise ValueError("A linear cost function was selected, but no weight vector was provided. Supply SCMP with the weight vector via `cost_const_kwargs`, using the key 'v'.")

        # The CCP component which stores gradients requires a non-batched cost function.
        # If the user does not supply SCMP with such a function, we wrap the batched cost function.
        if cost_fn_not_batched is None:
            def cost_no_batch(x, r, **kwargs):
                return cost_fn(cp.reshape(x, (1, self.x_dim)), cp.reshape(r, (1, self.x_dim)), **kwargs)

            cost_fn_not_batched = cost_no_batch

        # If the user does not pass a score function, we use the default linear score (and derivative).
        if score_fn is None:
            score_fn = SCMP.linear_score
        self.score = score_fn

        if score_fn_der is None:
            score_fn_der = SCMP.linear_score_der
        self.score_der = score_fn_der

        # Creates the response evaluator which will be used to compute the optimal x to which the user should transfer.
        self.response_evaluator = ResponseEvaluator(x_dim, batch_size, score_fn, score_fn_der, cost_fn, cost_fn_not_batched, cost_const_kwargs, train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap)

        # Social Measures (e.g. Utility, Burden, Recourse).
        if social_measure_dict is None:
            social_measure_dict = {}
        self.social_measure_dict = social_measure_dict
        if utility_reg > 0:
            self.social_measure_dict["utility"] = Utility(utility_reg, cost_fn_torch, cost_const_kwargs, train_slope, eval_slope)
        if burden_reg > 0:
            self.social_measure_dict["burden"] = Burden(burden_reg, x_dim, score_fn, cost_fn_not_batched, cost_fn_torch, cost_const_kwargs, x_lower_bound, x_upper_bound)
        if recourse_reg > 0:
            self.social_measure_dict["recourse"] = Recourse(recourse_reg, score_fn)

        # Time measurement variables.
        self.ccp_time = 0
        self.fit_time = 0

    def forward(self, X, requires_grad=True):
        if self.strategic:
            init_ccp_time = time.time()
            X_opt = self.optimize_X(X, requires_grad)
            self.ccp_time += time.time() - init_ccp_time
        else:
            X_opt = X
        Y_pred = self.score(X_opt, self.w, self.b)
        return X_opt, Y_pred

    def fit(self, X, Y, Xval=None, Yval=None, opt_class=torch.optim.Adam, opt_kwargs=None, shuffle=True, epochs=100, epochs_without_improvement_cap=4, verbose=None, save_progression=False, path=None, model_name=None):
        train_dataset = TensorDataset(X, Y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        train_batches = len(train_dataloader)
        train_losses = []
        train_errors = []

        for _, social_measure in self.social_measure_dict.items():
            social_measure.init_fit_train()

        validation = Xval is not None
        if validation:
            validation_dataset = TensorDataset(Xval, Yval)
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=shuffle)
            validation_losses = []
            validation_errors = []
            for _, social_measure in self.social_measure_dict.items():
                social_measure.init_fit_validation()

        # This initialization makes opt_kwargs' default value immutable.
        if opt_kwargs is None:
            opt_kwargs = {"lr": 5e-1}
        optimizer = opt_class(self.parameters(), **opt_kwargs)

        if save_progression:
            self.save_initial_progression(path, model_name)

        best_validation_error = 1
        consecutive_no_improvement = 0

        init_fit_time = time.time()
        for epoch in range(epochs):
            if verbose in ["batches", "epochs"]:
                print("starting epoch %03d / %03d." % (epoch + 1, epochs))

            init_epoch_time = time.time()
            batch = 0
            train_losses.append([])
            train_errors.append([])
            for _, social_measure in self.social_measure_dict.items():
                social_measure.begin_epoch(epoch, validation)

            for Xbatch, Ybatch in train_dataloader:
                for _, social_measure in self.social_measure_dict.items():
                    social_measure.begin_batch(epoch, batch, Xbatch, Ybatch)

                optimizer.zero_grad()
                Xbatch_opt, Ybatch_pred = self.forward(Xbatch, requires_grad=True)
                loss = self.loss(Ybatch, Ybatch_pred, Xbatch, Xbatch_opt)
                loss.backward()
                optimizer.step()

                train_losses[-1].append(loss.item())
                accuracy = self.calc_accuracy(Ybatch, Ybatch_pred)
                train_errors[-1].append(1 - accuracy)

                for _, social_measure in self.social_measure_dict.items():
                    social_measure.end_batch(epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, self.w, self.b)

                if verbose in ["batches"]:
                    self.print_batch_info(epoch + 1, batch + 1, train_batches, train_losses, train_errors)

                if save_progression:
                    self.save_batch_progression(epoch + 1, batch + 1, Xbatch, Xbatch_opt, Ybatch, Ybatch_pred, path, model_name)

                batch += 1

            if validation:
                if verbose in ["batches"]:
                    print("  finished training step, calculating validation loss and accuracy.")

                batch = 0

                validation_losses.append([])
                validation_errors.append([])

                for _, social_measure in self.social_measure_dict.items():
                    social_measure.begin_validation(epoch)

                with torch.no_grad():
                    for Xbatch, Ybatch in validation_dataloader:
                        for _, social_measure in self.social_measure_dict.items():
                            social_measure.begin_validation_batch(epoch, batch, Xbatch, Ybatch)

                        Xbatch_opt, Ybatch_pred = self.forward(Xbatch, requires_grad=False)
                        loss = self.loss(Ybatch, Ybatch_pred, Xbatch, Xbatch_opt)
                        validation_losses.append(loss)
                        accuracy = self.calc_accuracy(Ybatch, Ybatch_pred)
                        validation_errors.append(1 - accuracy)

                        for _, social_measure in self.social_measure_dict.items():
                            social_measure.end_validation_batch(epoch, batch, Xbatch, Ybatch, Xbatch_opt, Ybatch_pred, self.w, self.b)

                    batch += 1

            final_epoch_time = time.time()
            epoch_time = final_epoch_time - init_epoch_time
            if verbose in ["batches", "epochs"]:
                if validation:
                    self.print_epoch_info(epoch + 1, epochs, epoch_time, validation_losses, validation_errors)
                else:
                    self.print_epoch_info(epoch + 1, epochs, epoch_time)

            for _, social_measure in self.social_measure_dict.items():
                social_measure.end_epoch(epoch)

            if validation:
                avg_validation_error = np.mean(validation_errors[-1])
                if avg_validation_error < best_validation_error:
                    best_validation_error = avg_validation_error
                    consecutive_no_improvement = 0
                    self.save_model(path, model_name)
                elif epoch != epochs - 1:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= epochs_without_improvement_cap:
                        print(f"ending training due to {consecutive_no_improvement} consecutive epochs without improvement.")
                        break
            else:
                self.save_model(path, model_name)

        final_fit_time = time.time()
        self.fit_time = final_fit_time - init_fit_time
        print(f"Total training time: {self.fit_time} seconds.")

    def evaluate(self, X, Y, strategic_data=True):
        if strategic_data:
            X_opt = self.optimize_X(X, requires_grad=False)
            Y_pred = self.score(X_opt, self.w, self.b)
        else:
            Y_pred = self.score(X, self.w, self.b)
        return self.calc_accuracy(Y, Y_pred)

    def optimize_X(self, X, requires_grad=False):
        X_batches = X.split(self.batch_size)
        X_opt_batches = [self.response_evaluator.optimize_X(X_batch, self.w, self.b, requires_grad=requires_grad) for X_batch in X_batches]
        X_opt = torch.vstack(X_opt_batches)
        return X_opt

    def loss(self, Y, Y_pred, X, X_opt):
        loss_val = self.loss_fn(Y, Y_pred)
        for _, social_measure in self.social_measure_dict.items():
            loss_val += social_measure.get_loss_term(X, Y, X_opt, Y_pred, self.w, self.b)
        return loss_val

    def save_model(self, path, model_name):
        if path is None or model_name is None:
            return
        filename = f"{path}/{model_name}_model.pt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)
        for _, social_measure in self.social_measure_dict.items():
            social_measure.save(path, model_name)
        print(f"saving model to {filename}.")

    def load_model(self, path, model_name):
        filename = f"{path}/{model_name}_model.pt"
        if os.path.isfile(filename):
            print(f"loading model from {filename}.")
            self.load_state_dict(torch.load(filename))
            for _, social_measure in self.social_measure_dict.items():
                social_measure.load(path, model_name)
            self.eval()
            return True

        return False

    def print_batch_info(self, epoch, batch, train_batches, train_losses, train_errors):
        loss = np.mean(train_losses[-1])
        error = np.mean(train_errors[-1])
        # print(f"  batch: {batch:03d} / {train_batches:03d} | loss: {loss:3.5f} | error: {error:3.5f}", end="")

        print("  batch %03d / %03d | loss: %3.5f | error: %3.5f" % (batch, train_batches, loss, error), end="")
        for _, social_measure in self.social_measure_dict.items():
            sm_string = social_measure.get_batch_info(epoch, batch)
            if len(sm_string) > 0:
                print(f" | {sm_string}", end="")
        print("")

    def print_epoch_info(self, epoch, epochs, epoch_time, validation_losses=None, validation_errors=None):
        # print(f"epoch: {epoch:03d} / {epochs:03d}  | time: {epoch_time:03d} | loss: {loss:3.5f} | error: {error:3.5f}", end="")
        print("epoch %03d / %03d | time: %03d sec" % (epoch, epochs, epoch_time), end="")
        if validation_losses is not None:
            loss = validation_losses[-1]
            error = validation_errors[-1]
            print(" | loss: %3.5f | error: %3.5f" % (loss, error), end="")
            for _, social_measure in self.social_measure_dict.items():
                sm_string = social_measure.get_validation_info(epoch)
                if len(sm_string) > 0:
                    print(f" | {sm_string}", end="")
        print("")

    def normalize_weights(self):
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(self.w ** 2) + self.b ** 2)
            self.w /= norm
            self.b /= norm

    def save_initial_progression(self, path, model_name):
        if path is None or model_name is None:
            raise ValueError("In order to save the model's progression, one must supply SCMP with a path and model name.")
        filename = f"{path}/{model_name}_progress.csv"
        if os.path.isfile(filename):
            os.remove(filename)
        init_data = self.flatten_batch_for_saving(-1, 0, X=torch.zeros((1, self.x_dim)), X_opt=torch.zeros((1, self.x_dim)), Y=torch.zeros(1), Y_pred=torch.zeros(1))
        pd.DataFrame(init_data).to_csv(filename, index=False)

    def save_batch_progression(self, epoch, batch, X, X_opt, Y, Y_pred, path, model_name):
        filename = f"{path}/{model_name}_progress.csv"
        epoch_data = self.flatten_batch_for_saving(epoch, batch, X, X_opt, Y, Y_pred)
        pd.DataFrame(epoch_data).to_csv(filename, mode="a", header=False, index=False)

    def flatten_batch_for_saving(self, epoch, batch, X, X_opt, Y, Y_pred):
        return [{"epoch": epoch, "batch": batch,
                 **{f"w{j}": self.w[j].detach().numpy() for j in range(self.x_dim)}, "b": self.b[0].detach().numpy(),
                 **{f"x{j}": X[i][j].detach().numpy() for j in range(self.x_dim)}, "y": Y[i].detach().numpy(),
                 **{f"x_opt{j}": X_opt[i][j].detach().numpy() for j in range(self.x_dim)}, "y_pred": Y_pred[i].detach().numpy()} for i in range(len(X))]

    @staticmethod
    def calc_accuracy(Y, Y_pred):
        with torch.no_grad():
            Y_pred = torch.sign(Y_pred)
            temp = Y - Y_pred
            accuracy = len(temp[temp == 0]) / len(Y)
        return accuracy

    # Default linear score function and derivative.
    @staticmethod
    def linear_score(x, w, b):
        return x @ w + b

    @staticmethod
    def linear_score_der(x, w, b):
        return w


class FlexibleSCMP(SCMP):
    def __init__(self, x_dim, batch_size, v_init, price_fn=None, price_const_kwargs=None, reg_factor=1, mixing_parameter=0.05, cost_scale=1, loss_fn=hinge_loss, score_fn=None, score_fn_der=None, utility_reg=0, burden_reg=0, recourse_reg=0, social_measure_dict=None,
                 train_slope=1, eval_slope=5, x_lower_bound=-10, x_upper_bound=10, diff_threshold=0.001, iteration_cap=100, strategic=True):
        # We pass __init__ the linear cost functions just so cvxpy won't get upset, we override the response evaluator with a flexible response evaluator afterwards so it has no effect.
        super().__init__(x_dim, batch_size, loss_fn, linear_cost_cvxpy_batched, linear_cost_cvxpy_not_batched, {"v": v_init, "scale": cost_scale, "epsilon": mixing_parameter}, score_fn, score_fn_der, utility_reg, burden_reg, recourse_reg, social_measure_dict, linear_cost_torch, train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap, strategic)

        cost_fn_batched = FlexibleSCMP.linear_cost_fn_dpp
        cost_fn_not_batched = FlexibleSCMP.linear_cost_fn_dpp_not_batched
        if price_fn is None:
            price_fn = FlexibleSCMP.default_price_fn
        self.price = price_fn
        if price_const_kwargs is None:
            price_const_kwargs = {}
        self.price_const_kwargs = price_const_kwargs
        self.reg = reg_factor
        self.v_init = v_init
        self.v = torch.nn.parameter.Parameter(torch.clone(v_init))

        self.response_evaluator = FlexibleResponseEvaluator(x_dim, batch_size, self.score, self.score_der, cost_fn_batched, cost_fn_not_batched, {"scale": cost_scale, "epsilon": mixing_parameter}, train_slope, eval_slope, x_lower_bound, x_upper_bound, diff_threshold, iteration_cap)

    def optimize_X(self, X, requires_grad=False):
        X_batches = X.split(self.batch_size)
        X_opt_batches = [self.response_evaluator.optimize_X(X_batch, self.w, self.b, requires_grad=requires_grad, v=self.v) for X_batch in X_batches]
        X_opt = torch.vstack(X_opt_batches)
        return X_opt

    def loss(self, Y, Y_pred, X, X_opt):
        loss_val = super().loss(Y, Y_pred, X, X_opt)
        loss_val += self.reg * self.price(self.v, self.v_init, **self.price_const_kwargs)
        return loss_val

    def flatten_batch_for_saving(self, epoch, batch, X, X_opt, Y, Y_pred):
        return [{"epoch": epoch, "batch": batch,
                 **{f"w{j}": self.w[j].detach().numpy() for j in range(self.x_dim)}, "b": self.b[0].detach().numpy(),
                 **{f"v{j}": self.v[j].detach().numpy() for j in range(self.x_dim)},
                 **{f"x{j}": X[i][j].detach().numpy() for j in range(self.x_dim)}, "y": Y[i].detach().numpy(),
                 **{f"x_opt{j}": X_opt[i][j].detach().numpy() for j in range(self.x_dim)}, "y_pred": Y_pred[i].detach().numpy()} for i in range(len(X))]

    @staticmethod
    def linear_cost_fn_dpp(X, R, v, Rv, scale, epsilon):
        return scale * (epsilon * cp.square(cp.norm(X - R, 2, axis=1)) + (1 - epsilon) * cp.pos(X @ v - Rv))

    @staticmethod
    def linear_cost_fn_dpp_not_batched(x, r, v, rv, scale, epsilon):
        return scale * (epsilon * cp.sum_squares(x - r) + (1 - epsilon) * cp.pos(x @ v - rv))

    @staticmethod
    def default_price_fn(v, v_init):
        v_norm = torch.norm(v)
        v_init_norm = torch.norm(v_init)
        cos = (v @ v_init) / (v_norm * v_init_norm)
        return torch.abs(v_norm - v_init_norm) + torch.norm(cos - 1)
