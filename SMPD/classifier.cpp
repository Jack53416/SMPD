#include "Classifier.h"

std::vector<int> Classifier::selectedFeatures;
void Classifier::divideDatabase( Database &data) ///Dzieli baze na zbior testowy i treningowy
{
    unsigned int rN;
    qsrand(QTime::currentTime().msec());

    testSeq.clear();
    trainingSeq.clear();
    trainIndexes.clear();

    testSeq = data.getObjects(); //Zbior testowy zaczyna jako cala baza
    while(trainIndexes.size() < trainingSize) //losuje indexy do zbioru treningowego
    {
        rN= qrand()%(data.getNoObjects()-1);
        if(checkIfIndexOriginal(rN))
        {
            trainIndexes.push_back(rN);
            trainingSeq.push_back(testSeq[rN]); //od razu je przypisuje do treningowego
           // this->trainingSet.addObject(testSeq[rN]);// debug
        }
    }

    std::sort(trainIndexes.begin(), trainIndexes.end(), std::greater<unsigned int>()); //sort malejacy, zeby sie dobrze usuwalo pozniej

    for(unsigned int i = 0; i< trainingSize; i++) //Usuwa z testowego, te ktore zostaly wziete do treningowego
    {
        deleteIndex(trainIndexes[i], testSeq);
    }
}


void Classifier::deleteIndex(unsigned int index, std::vector<Object> & vec)
{
    vec.at(index) = vec.back();
    vec.pop_back();
}


bool Classifier::checkIfIndexOriginal(unsigned int index)
{
    if(trainIndexes.size() == 0)
        return true;
    for(unsigned int i = 0; i< trainIndexes.size(); i++)
    {
        if(index == trainIndexes.at(i))
            return false;
    }
    return true;

}

double Classifier::calculateDistance(Object& startVec, Object& endVec) // liczy pierwiastek z kwadratow roznic miedzy wektorami
{
    double result = 0.0;
    double sum=0.0;
    std::vector<float> startFeatures = startVec.getFeatures();
    std::vector<float> endFeatures = endVec.getFeatures();
    if(selectedFeatures.empty())
    {
        for(unsigned int i =0; i< startVec.getFeaturesNumber();i++)
        {
            sum=startFeatures.at(i) - endFeatures.at(i);
            result+=sum*sum;
        }
    }
    else
    {
        for(unsigned int i =0; i< Classifier::selectedFeatures.size(); i++)
        {
            sum = startFeatures.at(Classifier::selectedFeatures.at(i)) - endFeatures.at(Classifier::selectedFeatures.at(i));
            result+=sum*sum;
        }
    }
    result = sqrt(result);

    return result;
}

double Classifier::performBootstrap(int K)
{
    double avgFailRate = 0.0;

    for(int i =0; i<K; i++)
    {
        this->execute();
        avgFailRate += this->failureRate;
        this->train();
    }
    avgFailRate /= K;

    return avgFailRate;
}

double Classifier::performCrossValidation(int K)
{
    double avgFailRate = 0.0;
    auto vect = this->originalSet.getObjects();
    std::vector<Object> mixedObj;

    srand(QTime::currentTime().msec());
    int rN =0;
    while(vect.size() > 1)  ///Przelosowuje wektor obiektów z bazy danych aby dane byly losowo rozlozone
    {
        rN = qrand()%(vect.size()-1);
        mixedObj.push_back(vect.at(rN));
        deleteIndex(rN,vect);
    }
    mixedObj.push_back(vect.back());
    //qDebug()<<"mixedObj size:"<<mixedObj.size();

    int lastIndx = 0;
    std::vector<Object> tmpSeq;
    int chunk = (int)mixedObj.size()/K;
    this->crossValidation = true;

    for(int i =0; i<K; i++) /// wycina kawałki wektora potrzebne do test i training sequence a następnie wykonuje na nich dany classifier
    {
        this->testSeq.assign(mixedObj.begin()+lastIndx,mixedObj.begin()+chunk+lastIndx); /// wycięcie małego kawałka do test Seq
        tmpSeq = mixedObj;
        tmpSeq.erase(tmpSeq.begin()+lastIndx,tmpSeq.begin()+chunk+lastIndx); /// unięcie wcześniej wyciętego kawałka i utworzenie train seq
        this->trainingSeq = tmpSeq;
        lastIndx+= chunk;
        this->train();
        this->execute();
        avgFailRate += this->failureRate;
        qDebug()<<"FR:"<<this->failureRate;
    }

    this->crossValidation = false;
    avgFailRate/= K;
    qDebug()<<"FR avg:"<<avgFailRate;
    return avgFailRate;
}
