#include "Classifier.h"

std::vector<int> Classifier::selectedFeatures;
void Classifier::divideDatabase( Database &data)
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

    //Debug
   /* for(unsigned int i =0; i<testSeq.size(); i++)
    {
        this->testSet.addObject(testSeq[i]); //dodano petle dla debugu
    }
    trainingSet.save("trainSet.txt"); //dodano dla debugu
    testSet.save("testSet.txt");
    qDebug()<<"TrainingSeq:"<<trainingSeq.size() << "TestSeq:" << testSeq.size()<<"BaseSize" << data.getNoObjects();*/
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
