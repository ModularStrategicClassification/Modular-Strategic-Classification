import torch
import numpy as np
import pandas as pd
import cvxpy as cp

import matplotlib
import matplotlib.pyplot as plt

import Visualization


def initialize():
    matplotlib.rc('font', size=14)


def show_runtime_results():
    results_without_wrap = pd.read_csv("Results/runtime_varying_batchsizes/timing_results_without_wrap.csv")
    batch_sizes_without_wrap = results_without_wrap["batch_sizes"].to_list()
    fit_times_without_wrap = results_without_wrap["fit_times"].to_list()
    ccp_times_without_wrap = results_without_wrap["ccp_times"].to_list()

    results_with_wrap = pd.read_csv("Results/runtime_varying_batchsizes/timing_results_with_wrap.csv")
    batch_sizes_with_wrap = results_with_wrap["batch_sizes"].to_list()
    fit_times_with_wrap = results_with_wrap["fit_times"].to_list()
    ccp_times_with_wrap = results_with_wrap["ccp_times"].to_list()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    _ = fig.suptitle("Runtime for synthtic data")

    _ = ax1.set_title("Runtime without wrapping")
    _ = ax1.set_xlabel("Batch size")
    _ = ax1.set_xticks(batch_sizes_without_wrap)
    _ = ax1.set_ylabel("Runtime (seconds)")
    _ = ax1.plot(batch_sizes_without_wrap, fit_times_without_wrap, label="Fit time")
    _ = ax1.plot(batch_sizes_without_wrap, ccp_times_without_wrap, label="CCP time")
    _ = ax1.legend()
    
    _ = ax2.set_title("Runtime with wrapping")
    _ = ax2.set_xlabel("Batch size")
    _ = ax2.set_xticks(batch_sizes_with_wrap)
    _ = ax2.set_ylabel("Runtime (seconds)")
    _ = ax2.plot(batch_sizes_with_wrap, fit_times_with_wrap, label="Fit time")
    _ = ax2.plot(batch_sizes_with_wrap, ccp_times_with_wrap, label="CCP time")
    _ = ax2.legend()
    
    fig, ax = plt.subplots(figsize=(15, 6))
    _ = fig.suptitle("Difference between runtime with and without unbatched cost function")

    _ = ax.set_title("runtime without unbatched cost - runtime with unbatched cost")
    _ = ax.set_xlabel("Batch size")
    _ = ax.set_xticks(batch_sizes_without_wrap)
    _ = ax.set_ylabel("Runtime difference (seconds)")
    assert(batch_sizes_without_wrap == batch_sizes_with_wrap)
    fit_difference = [fit_times_without_wrap[i] - fit_times_with_wrap[i] for i in range(len(batch_sizes_without_wrap))]
    ccp_difference = [ccp_times_without_wrap[i] - ccp_times_with_wrap[i] for i in range(len(batch_sizes_without_wrap))]
    _ = ax.plot(batch_sizes_without_wrap, fit_difference, label="Fit time difference")
    _ = ax.plot(batch_sizes_without_wrap, ccp_difference, label="CCP time difference")
    _ = ax.legend()


def show_vanilla_results(dataset_amount=4):
    credit_results = pd.read_csv("Results/vanilla/credit_results.csv")
    distress_results = pd.read_csv("Results/vanilla/distress_results.csv")
    fraud_results = pd.read_csv("Results/vanilla/fraud_results.csv")
    spam_results = pd.read_csv("Results/vanilla/spam_results.csv")
    scales = credit_results["scales"].tolist()
    
    results = [credit_results, distress_results, fraud_results, spam_results]
    dataset_names = ["credit", "fin. distress", "fraud", "spam"]
    
    fig, axes = plt.subplots(3, dataset_amount, figsize=(4 * dataset_amount, 9))
    _ = fig.suptitle("Accuracy for various datasets and cost scales")

    for j in range(dataset_amount):
        _ = axes[0, j].set_title(dataset_names[j])
        result = results[j]
        for i in range(3):
            benchmark = result["benchmark"][i]
            SERM = result["SERM"][i]
            blind = result["blind"][i]
            _ = axes[i, j].axhline(y=benchmark, linewidth=3, linestyle="--", color="tab:gray", label="Benchmark")
            _ = axes[i, j].bar(1, SERM, 0.8, label="SERM")
            _ = axes[i, j].bar(2, blind, 0.8, label="Blind")
            _ = axes[i, j].get_xaxis().set_visible(False)
            _ = axes[i, j].set_ylim([0.5, 1])
            _ = axes[i, j].set_yticks([0.5, 1])
            _ = axes[i, j].label_outer()

    lines, labels = fig.axes[-1].get_legend_handles_labels()    
    _ = fig.legend(lines, labels, loc="lower center", ncol=3)


def show_vanilla_vs_hardt_results():
    results = pd.read_csv("Results/vanilla_vs_hardt/results.csv")
    epsilons = results["epsilons"].to_list()
    benchmark = results["benchmark"].to_list()
    SERM = results["SERM"].to_list()
    blind = results["blind"].to_list()
    Hardt = results["Hardt"].to_list()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    _ = fig.suptitle("Linear model on spam data")

    _ = ax.set_xlabel("Mixing parameter")
    _ = ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    _ = ax.set_ylabel("accuracy")
    _ = ax.set_ylim([0.46, 0.84])

    _ = ax.plot(epsilons, benchmark, label="Benchmark", linestyle="--", color="tab:gray")
    _ = ax.plot(epsilons, SERM, "-x", label="SERM")
    _ = ax.plot(epsilons, blind, "-x", label="Blind")
    _ = ax.plot(epsilons, Hardt, "-x", label="Hardt et al. (2016)")

    lines, labels = ax.get_legend_handles_labels()    
    _ = fig.legend(lines, labels, loc="right")


def show_flexibility_results():
    benchmark = pd.read_csv("Results/flexibility/benchmark_results.csv")['0'].to_list()
    oracle = pd.read_csv("Results/flexibility/oracle_results.csv")['0'].to_list()
    naive = pd.read_csv("Results/flexibility/naive_results.csv")['0'].to_list()
    SERM = pd.read_csv("Results/flexibility/SERM_results.csv")['0'].to_list()

    positive = pd.read_csv("Results/flexibility/positive.csv")
    positive_points_x = positive['0'].to_list()
    positive_points_y = positive['1'].to_list()
    negative = pd.read_csv("Results/flexibility/negative.csv")
    negative_points_x = negative['0'].to_list()
    negative_points_y = negative['1'].to_list()
    
    fig, ax = plt.subplots(figsize=(9, 9))
    _ = fig.suptitle("Flexibility")

    _ = ax.plot(0, 0, 'ok')
    _ = ax.plot(positive_points_x, positive_points_y, 'og')
    _ = ax.plot(negative_points_x, negative_points_y, 'or')

    _ = ax.arrow(0, 0, benchmark[3], benchmark[4], color="black")
    _ = ax.arrow(0, 0, oracle[3], oracle[4], color="black")
    _ = ax.arrow(0, 0, SERM[3], SERM[4], color='cyan')

    _ = ax.set_ylim(-1.1, 1.1)
    
    x = np.linspace(-1.1, 1.1, 100)
    def plot_line(w0, w1, b, linestyle, color, label):
        if w1 == 0:
            ax.axvline(-b / w0, linestyle=linestyle, color=color, label=label)
        else:
            ax.plot(x, -(w0 / w1) * x - b / w1, linestyle=linestyle, color=color, label=label)

    plot_line(benchmark[0], benchmark[1], benchmark[2], linestyle="--", color="tab:gray", label="benchmark")
    plot_line(naive[0], naive[1], naive[2], linestyle="-", color="orange", label="naive")
    plot_line(oracle[0], oracle[1], oracle[2], linestyle="-", color="gold", label="oracle")
    plot_line(SERM[0], SERM[1], SERM[2], linestyle="-", color="blue", label="SERM")
    _ = ax.legend()


def show_utility_recourse_burden_results():
    utility_results = pd.read_csv("Results/utility/results.csv")
    utility_accuracies = utility_results["accuracies"].to_list()
    utilities = utility_results["utilities"].to_list()
    utility_benchmark = utility_accuracies[0]
    utility_accuracies, utilities = utility_accuracies[1:], utilities[1:]
    
    burden_results = pd.read_csv("Results/burden/results.csv")
    burden_accuracies = burden_results["accuracies"].to_list()
    burdens = burden_results["burdens"].to_list()
    burden_benchmark = burden_accuracies[0]
    burden_accuracies, burdens = burden_accuracies[1:], burdens[1:]
    
    recourse_results = pd.read_csv("Results/recourse/results.csv")
    recourse_accuracies = recourse_results["accuracies"].to_list()
    recourses = recourse_results["recourses"].to_list()
    recourse_benchmark = recourse_accuracies[0]
    recourse_accuracies, recourses = recourse_accuracies[1:], recourses[1:]
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    _ = fig.suptitle("Social Measures")
    
    _ = ax[0].set_xlabel("utility")
    #_ = ax[0].set_xlim([0, 1])
    _ = ax[0].set_ylabel("accuracy")
    _ = ax[0].set_title("Expected utility tradeoff")
    _ = ax[0].axhline(y=utility_benchmark, linewidth=3, linestyle="--", color="gray", label="Benchmark")
    _ = ax[0].plot(utilities, utility_accuracies, 'o',  label="SERM")
    _ = ax[0].legend()
    
    _ = ax[1].set_xlabel("social burden (decreasing)")
    _ = ax[1].set_xlim([0.14, 0.04])
    _ = ax[1].set_ylabel("accuracy")
    _ = ax[1].set_title("Social burden tradeoff")
    _ = ax[1].axhline(y=burden_benchmark, linewidth=3, linestyle="--", color="gray", label="Benchmark")
    _ = ax[1].plot(burdens, burden_accuracies, 'o',  label="SERM")
    _ = ax[1].legend()
    
    _ = ax[2].set_xlabel("recourse")
    #_ = ax[2].set_xlim([0, 1])
    _ = ax[2].set_ylabel("accuracy")
    _ = ax[2].set_title("Recourse tradeoff")
    _ = ax[2].axhline(y=recourse_benchmark, linewidth=3, linestyle="--", color="gray", label="Benchmark")
    _ = ax[2].plot(recourses, recourse_accuracies, 'o',  label="SERM")
    _ = ax[2].legend()


def show_visualizations():
    animation1 = Visualization.visualize("./Results/visualization", "slope1")
    animation2 = Visualization.visualize("./Results/visualization", "slope40")
    return animation1, animation2


def slope_discussion():
    credit_results = pd.read_csv("Results/slopes/credit_results.csv")
    distress_results = pd.read_csv("Results/slopes/distress_results.csv")
    fraud_results = pd.read_csv("Results/slopes/fraud_results.csv")
    slopes = credit_results["slopes"].tolist()
    
    results = [credit_results, distress_results, fraud_results]
    
    fig, axes = plt.subplots(3, figsize=(15, 9))
    _ = fig.suptitle("Accuracy for various datasets and cost scales")
    _ = axes[0].set_title("credit")
    _ = axes[1].set_title("fin. distress")
    _ = axes[2].set_title("fraud")

    for j in range(3):
        result = results[j]
        SERM = result["SERM"]
        blind = result["blind"]
        _ = axes[j].plot(slopes, blind, label="Blind")
        _ = axes[j].plot(slopes, SERM, label="SERM")
        # _ = axes[j].label_outer()

    lines, labels = fig.axes[-1].get_legend_handles_labels()    
    _ = fig.legend(lines, labels, loc="lower center", ncol=3)

