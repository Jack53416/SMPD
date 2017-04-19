#ifndef NEARESTMEAN_H
#define NEARESTMEAN_H

#include "Classifier.h"
#include "database.h"
#include <QTCore>

class NearestMean : public Classifier
{
public:
    NearestMean(Database &data);
    void train();
    void execute();
    double calculateDistance(Object& startVec, Object& endVec);
    ClosestObject classifyObject(Object obj);
    std::map<Object*, ClosestObject> log; //zeby wyswietlic pelne dane w GUI
    void calculateMean(Database &data);

private:
    void divideDatabase(Database &data);
    bool checkIfIndexOriginal(unsigned int index);

    static void deleteIndex(unsigned int index, std::vector<Object> & vec);
};

#endif // NEARESTMEAN_H
