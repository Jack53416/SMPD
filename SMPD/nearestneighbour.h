#ifndef NEARESTNEIGHBOUR_H
#define NEARESTNEIGHBOUR_H

#include "classifier.h"
#include "database.h"
#include <QTCore>


typedef struct ClosestObject {
    double distance;
    Object* obj;
} ClosestObject;

class NearestNeighbour:public Classifier
{
public:
    NearestNeighbour(Database &data);
    void train();
    void execute();

    std::map<Object*, ClosestObject> log; //zeby wyswietlic pelne dane w GUI

private:
    void divideDatabase(Database &data);
    bool checkIfIndexOriginal(unsigned int index);
    double calculateDistance(Object& startVec, Object& endVec);
    ClosestObject classifyObject(Object obj);
    static void deleteIndex(unsigned int index, std::vector<Object> & vec);
};

#endif // NEARESTNEIGHBOUR_H
