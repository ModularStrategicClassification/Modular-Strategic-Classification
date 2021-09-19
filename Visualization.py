import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

# We have three phases: phase 0 shows the batch x's and classifier's line, phase 1 moves the xs to x_opts, phase 2 changes the classifier's line.
NUM_PHASES = 3
# On phase 2 the previous line and x_opt are drawn with this alpha
SMALL_ALPHA = 0.25

x_colors = {1: "#8aff8a", -1: "#ff8a8a"}
x_opt_colors = {1: "#45e645", -1: "#e64545"}


def get_color_for_prediction(y_pred):
    pred = 1 if y_pred > 0 else -1
    return x_opt_colors[pred]


def visualize(path, model_name, llim=-3, rlim=3, dlim=-3, ulim=3, interval=1000):
    progression = pd.read_csv(f"{path}/{model_name}_progress.csv")
    epochs_list = progression["epoch"].unique()
    epoch_num = max(epochs_list)
    batches_list = progression["batch"].unique()
    batch_num = max(batches_list)

    fig, ax = plt.subplots(figsize=(9, 9))

    x = np.linspace(llim, rlim, 10)

    def plot_line(w0, w1, b, linestyle="-", color="black", alpha=1):
        if w1 == 0:
            ax.axvline(-b / w0, linestyle=linestyle, color=color, alpha=alpha)
        else:
            ax.plot(x, -((w0 / w1) * x) - (b / w1), linestyle="-", color=color, alpha=alpha)

    w0, w1, b = None, None, None

    def init():
        nonlocal w0, w1, b

        init_data = progression[progression.epoch == -1]
        w0, w1, b = init_data["w0"][0], init_data["w1"][0], init_data["b"][0]

        ax.set_xlim(llim, rlim)
        ax.set_ylim(dlim, ulim)
        return fig,

    def animate_aux(epoch, batch, phase):
        nonlocal w0, w1, b

        ax.clear()
        ax.grid()
        ax.set_xlim(llim, rlim)
        ax.set_ylim(dlim, ulim)

        # Extract epoch and batch data.
        epoch_data = progression[progression.epoch == epoch]
        batch_data = epoch_data[epoch_data.batch == batch]

        # Draw the entire dataset in the background.
        x0, x1, y = epoch_data["x0"].to_list(), epoch_data["x1"].to_list(), epoch_data["y"].to_list()
        edgecolor = [x_colors[t] for t in y]
        ax.scatter(x0, x1, color="white", edgecolor=edgecolor)

        # Draw the points in the current batch.
        xbatch0, xbatch1, ybatch = batch_data["x0"].to_list(), batch_data["x1"].to_list(), batch_data["y"].to_list()
        ax.scatter(xbatch0, xbatch1, s=6, color="gray")

        # Draw the classifier's line (update if phase == 2 and draw the old line with small alpha).
        if phase == 2:
            plot_line(w0, w1, b, alpha=SMALL_ALPHA)
            w0, w1, b = batch_data["w0"].to_list()[0], batch_data["w1"].to_list()[0], batch_data["b"].to_list()[0]
        plot_line(w0, w1, b)

        if phase >= 1:
            # Draw user's responses.
            alpha = 1 if phase == 1 else SMALL_ALPHA
            xbatch_opt0, xbatch_opt1, ybatch_pred = batch_data["x_opt0"].to_list(), batch_data["x_opt1"].to_list(), batch_data["y_pred"]
            pred_colors = [get_color_for_prediction(t) for t in ybatch_pred]
            ax.scatter(xbatch_opt0, xbatch_opt1, s=6, c=pred_colors, alpha=alpha)
            # Draw arrows between user and response (X and X_opt).
            for i in range(len(xbatch0)):
                ax.arrow(xbatch0[i], xbatch1[i], xbatch_opt0[i] - xbatch0[i], xbatch_opt1[i] - xbatch1[i], color="gray", alpha=alpha)
            # Maybe write cost in figure.
            # Maybe draw circles on figure.

        # Write additional data
        model_text = f"w0 = {w0:.5f}, w1 = {w1:.5f}, b = {b:.5f}"
        ax.text(llim + 0.1, ulim - 0.2, model_text)
        epoch_batch_text = f"epoch = {epoch}/{epoch_num}, batch = {batch}/{batch_num}"
        ax.text(llim + 0.1, dlim + 0.1, epoch_batch_text)

    def animate(i):
        phase = i % NUM_PHASES
        eff_i = i // NUM_PHASES
        epoch, batch = eff_i // batch_num + 1, eff_i % batch_num + 1
        animate_aux(epoch, batch, phase)

    print("Creating animation.")
    animation = FuncAnimation(fig, animate, frames=range(NUM_PHASES * epoch_num * batch_num), init_func=init, interval=interval, repeat=False)
    print("Animation created, saving.")
    animation.save(f"{path}/{model_name}_animation.gif")
    print("Saving complete, closing plot and returning.")
    plt.close()
    return animation
