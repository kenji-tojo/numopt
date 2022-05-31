#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>

#include "Eigen/Dense"


// Minimize a function f = \| b - A w \|^2 + \lambda \| w \|^2
// using variants of the steepest descent algorithm
class SteepestDescent {
public:
    SteepestDescent(Eigen::MatrixXd A, Eigen::VectorXd b)
            : m_A(std::move(A))
            , m_b(std::move(b))
    {
        init();
    }

    [[nodiscard]]
    inline double eval_f(const Eigen::VectorXd &w) const {
        return (m_b - m_A * w).squaredNorm() + m_lambda * w.squaredNorm();
    }

    inline void init() {
        m_C = 2.0 * (m_A.transpose() * m_A + m_lambda * Eigen::MatrixXd::Identity(m_A.cols(),m_A.cols()));
        m_d = 2.0 * m_A.transpose() * m_b;
        m_inv_L = 1.0 / m_C.operatorNorm();
        m_w_star = m_C.colPivHouseholderQr().solve(m_d);
        m_f_star = eval_f(m_w_star);
    }

    inline void set_lambda(double lambda) {
        m_lambda = lambda < 0 ? 0 : lambda;
        init();
    }

    [[nodiscard]]
    double lambda() const { return m_lambda; }

    inline void step(Eigen::VectorXd &w) {
        auto grad = m_C * w - m_d;
        w = w - m_inv_L * grad;
    }

    struct ArmijoParams {
        double alpha_zero = 1;
        double tau = 0.5;
        double xi = 1e-3;
        int iter_max = 50;
    };

    inline void step_Armijo(Eigen::VectorXd &w, const ArmijoParams &prm)
    {
        auto grad = m_C * w - m_d;
        auto grad2 = grad.squaredNorm();
        auto f = eval_f(w);
        double f_tmp;
        Eigen::VectorXd w_tmp;
        // backtracking
        auto alpha = prm.alpha_zero;
        for (int l=0;l<prm.iter_max;l++) {
            w_tmp = w - alpha * grad;
            f_tmp = eval_f(w_tmp);
            if (f_tmp <= f - prm.xi * alpha * grad2) {
                break;
            }
            alpha *= prm.tau;
        }
        w = w_tmp;
    }

    inline void step_Nesterov(Eigen::VectorXd &w, Eigen::VectorXd &y, const int k) {
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
    double m_lambda = 0; // \in \mathbb{R}_{ \geq 0 }

    Eigen::MatrixXd m_C; // := 2 ( A^T A + \lambda I )
    Eigen::VectorXd m_d; // := 2 ( A^T b )
    double m_inv_L = 1;  // := 1 / \| C \|_2

    Eigen::VectorXd m_w_star; // \arg\min f
    double m_f_star = 0; // \min f

};

#define WRITE_HEADER(out, sd) \
    out << "lambda: " << (sd).lambda() << std::endl \
        << "f_star: " << (sd).f_star() << std::endl

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
    const auto lambda_list = std::vector<double>{ 0, 1, 10.0 };

    SteepestDescent sd(A, b);
    Eigen::VectorXd w(n), y(n);

    std::ofstream ofs;
    ofs.open("data/q1.txt", std::ios::out);
    for (const auto lambda : lambda_list) {
        sd.set_lambda(lambda);
        WRITE_HEADER(ofs, sd);
        w.setZero();
        std::string delim = "residuals: ";
        for (int i=0;i<niter;i++) {
            sd.step(w);
            auto f = sd.eval_f(w);
            ofs << delim << f - sd.f_star();
            if (delim != ",") delim = ",";
        }
        ofs << std::endl << "END" << std::endl;
    }
    ofs.close();

    ofs.open("data/q2.txt", std::ios::out);
    {
        SteepestDescent::ArmijoParams prm;
        prm.alpha_zero = 1; prm.tau = 0.5; prm.xi = 1e-3; prm.iter_max = 50;
        for (const auto lambda : lambda_list) {
            sd.set_lambda(lambda);
            WRITE_HEADER(ofs, sd);
            w.setZero();
            std::string delim = "residuals: ";
            for (int i=0;i<niter;i++) {
                sd.step_Armijo(w, prm);
                auto f = sd.eval_f(w);
                ofs << delim << f - sd.f_star();
                if (delim != ",") delim = ",";
            }
            ofs << std::endl << "END" << std::endl;
        }
    }
    ofs.close();

    ofs.open("data/q3.txt", std::ios::out);
    for (const auto lambda : lambda_list) {
        sd.set_lambda(lambda);
        WRITE_HEADER(ofs, sd);
        w.setZero(); y.setZero();
        std::string delim = "residuals: ";
        for (int i=0;i<niter;i++) {
            sd.step_Nesterov(w, y, i+1);
            auto f = sd.eval_f(w);
            ofs << delim << f - sd.f_star();
            if (delim != ",") delim = ",";
        }
        ofs << std::endl << "END" << std::endl;
    }
    ofs.close();

    return 0;
}