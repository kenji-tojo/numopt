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

    ks, ls, fs = list(), list(), list()
    with open(fname) as file:
        lines = file.readlines()
        for line in lines:
            if "lambda: " in line:
                for param_str in line.split(","):
                    if "niter: " in param_str:
                        ks.append(int(param_str.split(" ")[1]))
                    elif "lambda: " in param_str:
                        ls.append(float(param_str.split(" ")[1]))
                    elif "f_star: " in param_str:
                        fs.append(float(param_str.split(" ")[1]))
                print("loading")
                print("niter: ",  ks[-1])
                print("lambda: ", ls[-1])
                print("f_star: ", fs[-1])
                values   [ls[-1]] = list()
                residuals[ls[-1]] = list()
            else:
                values   [ls[-1]].append(float(line.split(",")[0]))
                residuals[ls[-1]].append(float(line.split(",")[1]))
    
    for l in ls:
        log_res = np.log(np.clip(np.array(residuals[l]), a_min=np.exp(-30), a_max=None))
        plot(values[l], log_res, l, fname.split(".")[0], title+"$ \lambda = {}$".format(l))


if __name__ == "__main__":
    read_and_plot("data/q1.txt", "Normal")
    read_and_plot("data/q2.txt", "Armijo")
    read_and_plot("data/q3.txt", "Nesterov")
