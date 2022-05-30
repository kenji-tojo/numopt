#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include "Eigen/Dense"

class SteepestDescent {
public:
    SteepestDescent(Eigen::MatrixXd A, Eigen::VectorXd b)
            : m_A(std::move(A))
            , m_b(std::move(b))
    {
        init();
    }

    [[nodiscard]]
    inline double eval(const Eigen::VectorXd &w) const {
        return (m_b - m_A * w).squaredNorm() + m_lambda * w.squaredNorm();
    }

    inline void init() {
        m_C = 2.0 * (m_A.transpose() * m_A + m_lambda * Eigen::MatrixXd::Identity(m_A.cols(),m_A.cols()));
        m_d = 2.0 * m_A.transpose() * m_b;
        m_inv_L = 1.0 / m_C.operatorNorm();
        m_w_star = m_C.colPivHouseholderQr().solve(m_d);
        m_f_star = eval(m_w_star);
    }

    inline void set_lambda(double lambda) {
        m_lambda = lambda < 0 ? 0 : lambda;
        init();
    }

    [[nodiscard]]
    double lambda() const { return m_lambda; }

    void step(Eigen::VectorXd &w) {
        auto grad = m_C * w - m_d;
        w = w - m_inv_L * grad;
    }

    struct ArmijoParams {
        double alpha_zero = 1;
        double tau = 0.5;
        double xi = 1e-3;
        int iter_max = 50;
    };

    void step_Armijo(Eigen::VectorXd &w, const ArmijoParams &prm)
    {
        auto grad = m_C * w - m_d;
        auto grad2 = grad.squaredNorm();
        auto f = eval(w);
        auto f_tmp = f;
        auto w_tmp = w;
        auto alpha = prm.alpha_zero;
        for (int l=0;l<prm.iter_max;l++) {
            w_tmp = w - alpha * grad;
            f_tmp = eval(w_tmp);
            if (f_tmp <= f - prm.xi * alpha * grad2) {
                break;
            }
            alpha *= prm.tau;
        }
        w = w_tmp;
    }

    void step_Nesterov(Eigen::VectorXd &w, Eigen::VectorXd &y, const int k) {
        auto grad = m_C * y - m_d;
        auto beta = (double)k / (double)(k+3);
        auto w_prev = w;
        w = y - m_inv_L * grad;
        y = w + beta * (w - w_prev);
    }

    [[nodiscard]]
    inline double f_star() const { return m_f_star; }

protected:
    Eigen::MatrixXd m_A;
    Eigen::VectorXd m_b;
    double m_lambda = 0;
    double m_inv_L = 1;

    Eigen::MatrixXd m_C;
    Eigen::VectorXd m_d;

    Eigen::VectorXd m_w_star;
    double m_f_star = 0;

};

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
        b(i) = (double)(i+1);
    }
    const auto lambdas = std::vector<double>{ 0, 1, 10.0 };

    SteepestDescent optim(A, b);
    Eigen::VectorXd w(n), y(n);

    std::ofstream ofs;
    ofs.open("data/q1.txt", std::ios::out);
    for (const auto lambda : lambdas) {
        optim.set_lambda(lambda);
        ofs << "niter: "  << niter << ",";
        ofs << "lambda: " << optim.lambda() << ",";
        ofs << "f_star: " << optim.f_star() << std::endl;

        w.setZero();
        for (int i=0;i<niter;i++) {
            optim.step(w);
            auto f = optim.eval(w);
            ofs << f << "," << f - optim.f_star() << std::endl;
        }
    }
    ofs.close();

    ofs.open("data/q2.txt", std::ios::out);
    {
        SteepestDescent::ArmijoParams prm;
        prm.alpha_zero = 1;
        prm.tau = 0.5;
        prm.xi = 1e-3;
        for (const auto lambda : lambdas) {
            optim.set_lambda(lambda);
            ofs << "niter: "  << niter << ",";
            ofs << "lambda: " << optim.lambda() << ",";
            ofs << "f_star: " << optim.f_star() << std::endl;

            w.setZero();
            for (int i=0;i<niter;i++) {
                optim.step_Armijo(w, prm);
                auto f = optim.eval(w);
                ofs << f << "," << f - optim.f_star() << std::endl;
            }
        }
    }
    ofs.close();

    ofs.open("data/q3.txt", std::ios::out);
    for (const auto lambda : lambdas) {
        optim.set_lambda(lambda);
        ofs << "niter: "  << niter << ",";
        ofs << "lambda: " << optim.lambda() << ",";
        ofs << "f_star: " << optim.f_star() << std::endl;

        w.setZero(); y.setZero();
        for (int i=0;i<niter;i++) {
            optim.step_Nesterov(w, y, i+1);
            auto f = optim.eval(w);
            ofs << f << "," << f - optim.f_star() << std::endl;
        }
    }
    ofs.close();

    return 0;
}