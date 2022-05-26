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

void steepest_descent(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double lambda,
        const int niter,
        std::ofstream &ofs)
{
    const auto n = A.cols();

    auto C = 2.0 * (A.transpose() * A + lambda * Eigen::MatrixXd::Identity(n,n));
    auto d = 2.0 * A.transpose() * b;

    auto alpha = 1.0 / C.operatorNorm();
    std::cout << "using constant alpha" << std::endl;

    Eigen::VectorXd w_star = C.colPivHouseholderQr().solve(d);
    const auto f_star = eval_f(A, b, lambda, w_star);
    ofs << "niter: "  << niter << ",";
    ofs << "lambda: " << lambda << ",";
    ofs << "f_star: " << f_star << std::endl;
    Eigen::VectorXd w(n); w.setZero();
    for (int iter=0;iter<niter;iter++) {
        auto grad = C * w - d;
        w = w - alpha * grad;
        auto f = eval_f(A, b, lambda, w);
        ofs << f << "," << std::abs(f_star - f) << std::endl;
    }
}

void steepest_descent_armijo(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double lambda,
        const int niter,
        const double xi,
        const double tau,
        const int nbacktrack,
        std::ofstream &ofs)
{
    const auto n = A.cols();

    auto C = 2.0 * (A.transpose() * A + lambda * Eigen::MatrixXd::Identity(n,n));
    auto d = 2.0 * A.transpose() * b;

    std::cout << "using Armijo criteria" << std::endl;

    Eigen::VectorXd w_star = C.colPivHouseholderQr().solve(d);
    const auto f_star = eval_f(A, b, lambda, w_star);
    ofs << "niter: "  << niter << ",";
    ofs << "lambda: " << lambda << ",";
    ofs << "f_star: " << f_star << std::endl;
    Eigen::VectorXd w(n); w.setZero();
    Eigen::VectorXd w_tmp = w;
    for (int iter=0;iter<niter;iter++) {
        auto grad = C * w - d;
        Eigen::VectorXd dir = -grad;
        dir.normalize();
        auto gdd = grad.dot(dir);
        auto alpha = 1.0;
        auto f = eval_f(A, b, lambda, w);
        auto f_tmp = f;
        for (int l=0;l<nbacktrack;l++) {
            w_tmp = w + alpha * dir;
            f_tmp = eval_f(A, b, lambda, w_tmp);
            if (f_tmp <= f + xi * alpha * gdd) {
                break;
            }
            alpha *= tau;
        }
        w = w_tmp;
        f = f_tmp;
        ofs << f << "," << std::abs(f_star - f) << std::endl;
    }
}

void steepest_descent_nesterov(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const double lambda,
        const int niter,
        std::ofstream &ofs)
{
    const auto n = A.cols();

    auto C = 2.0 * (A.transpose() * A + lambda * Eigen::MatrixXd::Identity(n,n));
    auto d = 2.0 * A.transpose() * b;

    auto alpha = 1.0 / C.operatorNorm();
    std::cout << "using Nesterov acceleration" << std::endl;

    Eigen::VectorXd w_star = C.colPivHouseholderQr().solve(d);
    const auto f_star = eval_f(A, b, lambda, w_star);
    ofs << "niter: "  << niter << ",";
    ofs << "lambda: " << lambda << ",";
    ofs << "f_star: " << f_star << std::endl;
    Eigen::VectorXd w(n); w.setZero();
    Eigen::VectorXd w_prev = w;
    Eigen::VectorXd y = w;
    for (int iter=0;iter<niter;iter++) {
        auto grad = C * y - d;
        auto beta = (double)(iter+1) / (double)(iter+4);
        w_prev = w;
        w = y - alpha * grad;
        y = w + beta * (w - w_prev);
        auto f = eval_f(A, b, lambda, w);
        ofs << f << "," << std::abs(f_star - f) << std::endl;
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
    ofs.open("data/q1.txt", std::ios::out);
    for (const auto lambda : lambdas) {
        steepest_descent(A, b, lambda, niter, ofs);
    }
    ofs.close();

    ofs.open("data/q2.txt", std::ios::out);
    for (const auto lambda : lambdas) {
        steepest_descent_armijo(A, b, lambda, niter, 1e-3, 0.5, 50, ofs);
    }
    ofs.close();

    ofs.open("data/q3.txt", std::ios::out);
    for (const auto lambda : lambdas) {
        steepest_descent_nesterov(A, b, lambda, niter, ofs);
    }
    ofs.close();

    return 0;
}