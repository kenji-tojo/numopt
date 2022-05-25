#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include "Eigen/Dense"

inline double eval_f(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double lambda,
        const Eigen::VectorXd &w)
{
    return (b - A * w).squaredNorm() + lambda * w.squaredNorm();
}

template <bool armijo = false>
void steepest_descent(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double lambda,
        const int niter,
        std::ofstream &ofs)
{
    const auto n = A.cols();

    auto C = A.transpose() * A + (2.0 * lambda) * Eigen::MatrixXd::Identity(n,n);
    auto d = A.transpose() * b;

    auto alpha = 1.0;
    if constexpr (!armijo) {
        std::cout << "using constant alpha" << std::endl;
        alpha = 1.0 / C.operatorNorm();
    } else {
        std::cout << "using Armijo's rule to determine alpha" << std::endl;
    }

    Eigen::VectorXd w_star = C.colPivHouseholderQr().solve(d);
    const auto f_star = eval_f(A, b, lambda, w_star);
    ofs << "lambda: " << lambda << std::endl;
    ofs << "f_star: " << f_star << std::endl;
    Eigen::VectorXd w(n); w.setZero();
    for (int iter=0;iter<niter;iter++) {
        auto grad = C * w - d;
        w = w - alpha * grad;
        auto f = eval_f(A, b, lambda, w);
        ofs << f << ", " << std::abs(f_star - f) << std::endl;
    }
}

int main() {

    constexpr int m = 5;
    constexpr int n = 20;
    constexpr int niter = 50;

    std::mt19937 gen(0);
    std::normal_distribution<> dis(0,1);

    Eigen::MatrixXd A(m,n);
    Eigen::VectorXd b(m);
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            A(i,j) = dis(gen);
        }
        b(i) = (double)i;
    }
    const auto lambdas = std::vector<double>{ 0, 1, 10.0 };

    std::ofstream ofs;
    ofs.open("data/question1.txt", std::ios::out);
    for (const auto lambda : lambdas) {
        steepest_descent(A, b, lambda, niter, ofs);
    }
    ofs.close();

    return 0;
}