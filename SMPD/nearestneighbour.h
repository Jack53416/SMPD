#ifndef NEARESTNEIGHBOUR_H
#define NEARESTNEIGHBOUR_H

#include "classifier.h"
#include "database.h"
#include <QTCore>

class NearestNeighbour: public Classifier
{
public:
    NearestNeighbour(Database &data);
    void train();
    void execute();
    std::string dumpLog(bool full);

    std::map<Object*, ClosestObject> log; ///pelne dane dla  GUI

private:
    ClosestObject classifyObject(Object obj);
};


#endif // NEARESTNEIGHBOUR_H
