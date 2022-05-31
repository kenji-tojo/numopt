import numpy as np
import matplotlib.pyplot as plt
import os


def plot(values, residuals, l, save_prefix, title):
    print(save_prefix)
    iter = np.arange(len(values)) + 1
    plt.clf()
    plt.title(title)
    plt.plot(iter, values, label="values")
    plt.plot(iter, residuals, label="residuals")
    plt.xlabel("iteration number $k$")
    plt.ylabel("$f(w_k)$")
    plt.legend()
    plt.savefig(save_prefix + "_" + str(l) + ".png")

def plot_residuals(residuals, lamb, save_prefix, title, log_scale=True):
    print("save_prefix:", save_prefix)

    plt.clf()
    plt.title(title)
    if log_scale:
        residuals = np.log(np.clip(np.array(residuals), a_min=np.exp(-30), a_max=None))
        plt.ylabel("$\log\left( f(w_k) - f(w^*) \\right)$")
    else:
        plt.ylabel("$f(w_k) - f(w^*)$")
    plt.plot(np.arange(len(residuals))+1, residuals)
    plt.xlabel("iteration number $k$")

    plt.savefig(save_prefix + "_" + str(int(lamb)) + ".png")


def read_and_plot(fname, title):
    lamb, fmin, vals = None, None, None
    with open(fname) as file:
        lines = file.readlines()
        for line in lines:
            if "lambda: " in line:
                lamb = float(line.split(" ")[1])
            if "f_star: " in line:
                fmin = float(line.split(" ")[1])
            if "residuals: " in line:
                vals = [ float(v) for v in line.split(" ")[1].split(",") ]
            if "END" in line:
                assert lamb is not None
                assert fmin is not None
                assert vals is not None
                plot_residuals(vals, lamb, fname.split(".")[0], title+" $\lambda={}$".format(lamb))
                lamb, fmin, vals = None, None, None


if __name__ == "__main__":
    read_and_plot("data/q1.txt", "Normal")
    read_and_plot("data/q2.txt", "Armijo")
    read_and_plot("data/q3.txt", "Nesterov")
