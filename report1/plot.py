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

def read_and_plot(fname, title):
    values = dict()
    residuals = dict()

    niter, l, f = None, None, None
    with open(fname) as file:
        lines = file.readlines()
        for line in lines:
            if "niter: " in line:
                if niter is not None:
                    plot(values[niter], residuals[niter], l, fname.split(".")[0], title)
                for ent in line.split(","):
                    if "niter: " in ent:
                        niter = int(ent.split(" ")[1])
                    elif "lambda: " in ent:
                        l = float(ent.split(" ")[1])
                    elif "f_star: " in ent:
                        f = float(ent.split(" ")[1])
                print("loading")
                print("niter: ", niter)
                print("lambda: ", l)
                print("f_star: ", f)
                values[niter]    = list()
                residuals[niter] = list()
            else:
                values[niter]   .append(float(line.split(",")[0]))
                residuals[niter].append(float(line.split(",")[1]))
        plot(values[niter], residuals[niter], l, fname.split(".")[0], title)


if __name__ == "__main__":
    read_and_plot("data/q1.txt", "Normal")
    read_and_plot("data/q2.txt", "Armijo")
    read_and_plot("data/q3.txt", "Nesterov")
