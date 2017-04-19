#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "database.h"

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

    virtual void train()=0;
    virtual void execute()=0;
    int k;
protected:
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
