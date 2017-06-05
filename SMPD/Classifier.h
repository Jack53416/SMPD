#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "database.h"
#include <functional>
#include <QtCore>
typedef struct ClosestObject {
    double distance;
    Object* obj;
} ClosestObject;


class Classifier{

public:
    Classifier(Database &data): originalSet(data){}
    unsigned int getTestSize(){
        return (unsigned int) testSeq.size();
    }
    unsigned int getTrainSize(){
        return (unsigned int) trainingSeq.size();
    }
    void setTrainSize(int number)
    {
        trainingSize=number*originalSet.getNoObjects()/100;
    }
    double getFailRate(){return failureRate;}

    double calculateDistance(Object& startVec, Object& endVec);

    double performBootstrap(int K);
    virtual double performCrossValidation(int K);
    virtual std::string dumpLog(bool full) = 0;

    virtual void train()=0;
    virtual void execute()=0;
    static std::vector<int> selectedFeatures;
protected:
    void divideDatabase(Database &data);
    bool checkIfIndexOriginal(unsigned int index);
    static void deleteIndex(unsigned int index, std::vector<Object> & vec);

    //variables
    int k;
    unsigned int trainingSize;
    double failureRate;

    std::vector<unsigned int> trainIndexes;
    std::vector<Object> trainingSeq;
    std::vector<Object> testSeq;

    Database trainingSet;
    Database testSet;
    Database& originalSet;

};

#endif // CLASSIFIER_H
