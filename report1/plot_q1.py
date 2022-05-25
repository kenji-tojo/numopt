from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


def plot(values, residuals, l):
    iter = np.arange(len(values)) + 1
    plt.clf()
    plt.plot(iter, values, label="values")
    plt.plot(iter, residuals, label="residuals")
    plt.legend()
    plt.savefig("data/plot_lambda{}.png".format(l))

if __name__ == "__main__":
    values = dict()
    residuals = dict()

    niter, l, f = None, None, None
    with open("data/q1.txt") as f:
        lines = f.readlines()
        for line in lines:
            if "niter: " in line:
                if niter is not None:
                    plot(values[niter], residuals[niter], l)
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
        plot(values[niter], residuals[niter], l)
