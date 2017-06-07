#include "arithmetics.h"


bool isFeautureOriginal(const std::vector<int> &featureList, int feature)
{
    if(featureList.size() == 0)
        return true;

    for(unsigned int i = 0; i< featureList.size(); ++i)
    {
        if(feature == featureList.at(i))
            return false;
    }
    return true;

}

double fisher_2C(FisherData& calcData)
{
    const size_t mSize = calcData.bestFeatures.size()+1;
    bnu::matrix <double> aResult(mSize,mSize);
    bnu::matrix<double> bResult(mSize,mSize);
    double detA = 0;
    double detB = 0;
    double FLD = 0;
    int currentFeature =0;
    double maxFLD;
    calcData.a.resize(calcData.a.size1()+1, calcData.a.size2());
    calcData.b.resize(calcData.b.size1()+1,calcData.b.size2());

    calcData.bestFeatures.push_back(-1);
    calcData.fisherCoeffs.push_back(-1);
    for(int i = 0; i<calcData.classAAvg.size(); ++i)
    {
        detA = 0;
        detB = 0;
        FLD = 0;

        if(isFeautureOriginal(calcData.bestFeatures,i)) ///pomijamy Featury ktore juz sa na liscie najlepszych
        {
            currentFeature = i;
            for(int j = 0; j<calcData.classAMembers.size(); ++j)
            {
                calcData.a(mSize-1,j) = calcData.classAMembers.at(j).getFeatures()[i]
                        - calcData.classAAvg.at(i);
            }
            for(int j = 0; j< calcData.classBMembers.size(); ++j)
            {
                calcData.b(mSize-1,j) = calcData.classBMembers.at(j).getFeatures()[i]
                        - calcData.classBAvg.at(i);
            }
            ///dodany row dla obu matrixow

            for(int j = 0; j<calcData.bestFeatures.size()-1; ++j)
            {
                FLD+= (calcData.classAAvg.at(calcData.bestFeatures[j])-calcData.classBAvg.at(calcData.bestFeatures[j]))
                     *(calcData.classAAvg.at(calcData.bestFeatures[j])-calcData.classBAvg.at(calcData.bestFeatures[j]));
            }
            FLD+= (calcData.classAAvg.at(currentFeature)-calcData.classBAvg.at(currentFeature))
                 *(calcData.classAAvg.at(currentFeature)-calcData.classBAvg.at(currentFeature));

            FLD=sqrt(FLD);
            //qDebug()<<"norm: "<<FLD;
            axpy_prod(calcData.a, trans(calcData.a), aResult, true); ///ares, bres = S matrix
            axpy_prod(calcData.b, trans(calcData.b), bResult, true);

            detA = determinant(aResult);
            detB = determinant(bResult);
            //qDebug()<<"DetA+ b ="<<detA+detB;

            FLD/=(detA+detB);
           // qDebug()<<"FLD:"<<FLD<<"kl:"<<currentFeature<<" Prev: "<<calcData.fisherCoeffs.back()<<"\n";
            if(FLD> calcData.fisherCoeffs.back())
            {
               //qDebug()<<"FLD:"<<FLD<<"kl:"<<currentFeature<<" Prev: "<<calcData.fisherCoeffs.back()<<"\n";
               maxFLD = FLD;
               calcData.fisherCoeffs.back() = FLD;
               calcData.bestFeatures.back() = currentFeature; ///dodaje obecny feature do listy najlepszych
            }
        }
    }
    for(int i = 0; i < calcData.classAMembers.size();++i) /// dopisuje wybrany najlepszy feature do wierszy macierzy a i b
    {
        calcData.a(mSize-1,i) = calcData.classAMembers.at(i).getFeatures()[calcData.bestFeatures.back()]
                - calcData.classAAvg.at(calcData.bestFeatures.back());
    }
    for(int i = 0; i < calcData.classBMembers.size();++i)
    {
        calcData.b(mSize-1,i) = calcData.classBMembers.at(i).getFeatures()[calcData.bestFeatures.back()]
                - calcData.classBAvg.at(calcData.bestFeatures.back());
    }
    return maxFLD;
}

FisherResult SFS(Database& inputSet, int N)
{
    FisherData calcData;
    FisherResult result;

    fisher_1D(inputSet,calcData); ///pierwszy feature liczony ze standardowego Fishera
    //calcData.bestFeatures.front() = 15;
    for (auto const &ob : inputSet.getObjects())
    {
        if(ob.getClassName() == inputSet.getClassNames()[0])
        {
            calcData.classAMembers.push_back(ob);
        }else{
            calcData.classBMembers.push_back(ob);
        }

    } // klasy rozdzielone

    calcData.a.resize(1, calcData.classAMembers.size());
    calcData.b.resize(1,calcData.classBMembers.size()); ///resize macierzy a i b aby pomiescic 1 najlepszy feature

    for(int i = 0; i < calcData.classAMembers.size();++i) ///dopisuje pierwszy feature do macierzy a i b
    {
        calcData.a(0,i) = calcData.classAMembers.at(i).getFeatures()[calcData.bestFeatures.front()]
                - calcData.classAAvg.at(calcData.bestFeatures.front());
    }
    for(int i = 0; i < calcData.classBMembers.size();++i)
    {
        calcData.b(0,i) = calcData.classBMembers.at(i).getFeatures()[calcData.bestFeatures.front()]
                - calcData.classBAvg.at(calcData.bestFeatures.front());
    }
    //macierze zainicjalizowane
    double max;

    while(calcData.bestFeatures.size() < N) /// dopoki liczba najlepszych featurow jest mniejsza od oczekiwanej szukaj dalej
    {
        max = fisher_2C(calcData);
        qDebug()<<"Feature: "<<calcData.bestFeatures.size()<<"from "<<N;
    }
    //qDebug()<<"min:"<<max;
    result.bestFeatures = calcData.bestFeatures;
    result.FLD = calcData.fisherCoeffs.back();

    return result; ///zwraca zestaw najlepszych featurow razem z ich wspolczynnikami
}

float fisher_1D(Database& database, FisherData& calcData) ///zwykly Fisher z mainwindow.cpp
{
    float FLD = 0, tmp;
    int max_ind = -1;
    for (uint i = 0; i < database.getNoFeatures(); ++i)
    {
        std::map<std::string, float> classAverages;
        std::map<std::string, float> classStds;

        for (auto const &ob : database.getObjects())
        {
            classAverages[ob.getClassName()] += ob.getFeatures()[i];
            classStds[ob.getClassName()] += ob.getFeatures()[i] * ob.getFeatures()[i];
        }

        std::for_each(database.getClassCounters().begin(), database.getClassCounters().end(), [&](const std::pair<std::string, int> &it)
        {
            classAverages[it.first] /= it.second;
            classStds[it.first] = std::sqrt(classStds[it.first] / it.second - classAverages[it.first] * classAverages[it.first]);
        }
        );

        tmp = std::abs(classAverages[ database.getClassNames()[0] ] - classAverages[database.getClassNames()[1]]) / (classStds[database.getClassNames()[0]] + classStds[database.getClassNames()[1]]);

        calcData.classAAvg.push_back(classAverages[database.getClassNames()[0]]);
        calcData.classBAvg.push_back(classAverages[database.getClassNames()[1]]);

        if (tmp > FLD)
        {
            FLD = tmp;
            max_ind = i;
        }
    }
    calcData.bestFeatures.push_back(max_ind);
    calcData.fisherCoeffs.push_back(FLD);
    return FLD;
}

int determinant_sign(const bnu::permutation_matrix<std::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}

double determinant( bnu::matrix<double>& m ) {
    bnu::permutation_matrix<std::size_t> pm(m.size1());
    double det = 1.0;
    if( bnu::lu_factorize(m,pm) ) /// LU factorization- decomposition of matrix NxN A into a product of L lower triagular and U upper triangular matrix
    {
        det = 0.0;
    }
    else
    {
        for(int i = 0; i < m.size1(); i++)
        det *= m(i,i); // multiply by elements on diagonal
        det = det * determinant_sign( pm );
    }
    return det;
}

std::vector<std::vector<int>> comb(int N, int K)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    std::vector<std::vector<int>> vect;
    int counter = 0;

    // print integers and permute bitmask
    do {
        vect.push_back(std::vector<int>());
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i])
            {
               // std::cout << " " << i;
                vect.at(counter).push_back(i);
            }

        }
        //std::cout << std::endl;
        counter++;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return vect;
}
