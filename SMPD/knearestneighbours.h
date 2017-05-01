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
    ClosestObject classifyObject(Object obj, Database &data);
    std::map<Object*, ClosestObject> log; //zeby wyswietlic pelne dane w GUI
    int k;

private:

};


#endif // KNEARESTNEIGHBOURS_H
