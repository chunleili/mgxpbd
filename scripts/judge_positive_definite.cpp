#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <stdexcept>

int main()
{
    Eigen::SparseMatrix<float> A;
    Eigen::loadMarket(A, "E:/Dev/mgxpbd/data/misc/A_10.mtx");
    // std::cout << "The matrix A is" << std::endl << A << std::endl;
    
    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> lltOfA(A); // compute the Cholesky decomposition of A
    if(lltOfA.info() == Eigen::NumericalIssue)
    {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }    
    else if(lltOfA.info() != Eigen::Success)
    {
        throw std::runtime_error("Decomposition failed!");
    }
    else
    {
        std::cout << "The matrix A is positive definite" << std::endl;
    }
}