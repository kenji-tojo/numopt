import numpy as np
import matplotlib.pyplot as plt


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

def plot_q1(result, save_prefix, log_scale=True):
    print("save_prefix:", save_prefix)

    plt.clf()
    plt.title("Convergence speed of fixed step-size")
    for lamb in result:
        vals = result[lamb]["residuals"]
        if log_scale:
            vals = np.log(np.clip(np.array(vals), a_min=np.exp(-30), a_max=None))
        plt.plot(np.arange(len(vals))+1, vals, label="$\lambda = {}$".format(lamb))
    plt.xlabel("iteration number $k$")
    plt.ylabel("$\log\left( f(w_k) - f(w^*) \\right)$" if log_scale else "$f(w_k) - f(w^*)$")
    plt.legend()
    plt.savefig(save_prefix + ".png")


def plot_comparison(res_fixed, res_advanced, name_advanced, lamb, save_prefix, log_scale=True):
    print("save_prefix:", save_prefix)
    print("lambda:", lamb)
    print("name of method:", name_advanced)

    plt.clf()
    plt.title("Comparison of fixed and {} steepest descent".format(name_advanced))
    vals = res_fixed[lamb]["residuals"]
    if log_scale:
        vals = np.log(np.clip(np.array(vals), a_min=np.exp(-30), a_max=None))
    plt.plot(np.arange(len(vals))+1, vals, label="$Fixed, \lambda = {}$".format(lamb))
    vals = res_advanced[lamb]["residuals"]
    if log_scale:
        vals = np.log(np.clip(np.array(vals), a_min=np.exp(-30), a_max=None))
    plt.plot(np.arange(len(vals))+1, vals, label="${}, \lambda = {}$".format(name_advanced, lamb))

    plt.xlabel("iteration number $k$")
    plt.ylabel("$\log\left( f(w_k) - f(w^*) \\right)$" if log_scale else "$f(w_k) - f(w^*)$")
    plt.legend()
    plt.savefig(save_prefix + ".png")


def read_result(fname):
    result = dict()
    lamb = None
    with open(fname) as file:
        lines = file.readlines()
        for line in lines:
            if "lambda: " in line:
                lamb = float(line.split(" ")[1])
                result[lamb] = dict()
            if "f_star: " in line:
                result[lamb]["f_star"] = float(line.split(" ")[1])
            if "residuals: " in line:
                result[lamb]["residuals"] = [ float(v) for v in line.split(" ")[1].split(",") ]
            if "END" in line:
                lamb = None
    return result


if __name__ == "__main__":
    res_fixed    = read_result("data/q1.txt")
    res_Armijo   = read_result("data/q2.txt")
    res_Nesterov = read_result("data/q3.txt")
    plot_q1(res_fixed, "./data/fig_q1")
    plot_comparison(res_fixed, res_Armijo, "Armijo", lamb=1.0, save_prefix="./data/fig_q2")
    plot_comparison(res_fixed, res_Nesterov, "Nesterov", lamb=1.0, save_prefix="./data/fig_q3")
