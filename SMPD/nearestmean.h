#ifndef NEARESTMEAN_H
#define NEARESTMEAN_H

#include "Classifier.h"
#include "database.h"
#include <QTCore>

class NearestMean : public Classifier
{
public:
    Database data1;
    NearestMean(Database &data);
    void train();
    void execute();
    std::string dumpLog(bool full);

    ClosestObject classifyObject(Object obj);
    std::map<Object*, ClosestObject> log; //zeby wyswietlic pelne dane w GUI
    void calculateMean(Database &data);

private:
};

#endif // NEARESTMEAN_H
