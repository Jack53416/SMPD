#include "nearestneighbour.h"
#include <functional>

NearestNeighbour::NearestNeighbour(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.2;
}

bool NearestNeighbour::checkIfIndexOriginal(unsigned int index)
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

void NearestNeighbour::divideDatabase( Database &data)
{
    unsigned int rN;
    qsrand(QTime::currentTime().msec());

    testSeq.clear();
    trainingSeq.clear();
    trainIndexes.clear();

    testSeq = data.getObjects(); //Zbior testowy zaczyna jako cala baza
    while(trainIndexes.size() <= trainingSize) //losuje indexy do zbioru treningowego
    {
        rN= qrand()%(data.getNoObjects()-1);
        if(checkIfIndexOriginal(rN))
            trainIndexes.push_back(rN);
    }

    std::sort(trainIndexes.begin(), trainIndexes.end(), std::greater<unsigned int>()); //sort malejacy, zeby sie dobrze usuwalo pozniej

    for(unsigned int i = 0; i< trainingSize; i++) // przypisuje do zbioru treningowego
    {
        qDebug()<< trainIndexes[i];
        trainingSeq.push_back(testSeq[trainIndexes[i]]);
    }

    for(unsigned int i = 0; i< trainingSize; i++) //Usuwa z testowego, te ktore zostaly wziete do treningowego
    {
        deleteIndex(trainIndexes[i], testSeq);
    }

    //Debug
    qDebug()<<"TrainingSeq:"<<trainingSeq.size() << "TestSeq:" << testSeq.size()<<"BaseSize" << data.getNoObjects();
}

void NearestNeighbour::deleteIndex(unsigned int index, std::vector<Object> & vec)
{
    vec.at(index) = vec.back();
    vec.pop_back();
}

void NearestNeighbour::train(){
    if(originalSet.getNoObjects() > 0)
        divideDatabase(originalSet);

}

void NearestNeighbour::execute(){
/*TO DO*/
}
