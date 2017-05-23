#ifndef ARITHMETICS_H
#define ARITHMETICS_H

#include <QtCore>
#include "matrixutil.hpp"
#include "object.h"
#include "database.h"

namespace bnu = boost::numeric::ublas;

typedef struct FisherResult{
    double FLD;
    std::vector<int> bestFeatures;
}FisherResult;

typedef struct FisherData {

    std::vector<double> classAAvg;
    std::vector<double> classBAvg;
    std::vector<Object> classAMembers;
    std::vector<Object> classBMembers;
    bnu::matrix<double> a;
    bnu::matrix<double> b;
    std::vector<int> bestFeatures;
    std::vector<double> fisherCoeffs;


}FisherData;

int determinant_sign(const bnu::permutation_matrix<std::size_t>& pm);
double determinant( bnu::matrix<double>& m );
double fisher_2C(FisherData& calcData);
FisherResult SFS(Database &InputSet, int N);
float fisher_1D(Database& database, FisherData &calcData);
std::vector<std::vector<int>> comb(int N, int K);
bool isFeautureOriginal(const std::vector<int> &featureList, int feature);

#endif
