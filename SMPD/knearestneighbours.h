#ifndef KNEARESTNEIGHBOURS_H
#define KNEARESTNEIGHBOURS_H

#include "Classifier.h"
#include "database.h"
#include <QTCore>


class KNearestNeighbours: public Classifier
{
public:
    KNearestNeighbours(Database &data);
    void train();
    void execute();
    void execute(Database &data);
    double calculateDistance(Object& startVec, Object& endVec);
    ClosestObject classifyObject(Object obj, Database &data);
    std::map<Object*, ClosestObject> log; //zeby wyswietlic pelne dane w GUI
    int k;

private:
    void divideDatabase(Database &data);
    bool checkIfIndexOriginal(unsigned int index);
    static void deleteIndex(unsigned int index, std::vector<Object> & vec);
};


#endif // KNEARESTNEIGHBOURS_H
