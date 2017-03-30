#ifndef NEARESTNEIGHBOUR_H
#define NEARESTNEIGHBOUR_H

#include "classifier.h"
#include "database.h"
#include <QTCore>

class NearestNeighbour:public Classifier
{
public:
    NearestNeighbour(Database &data);
    void train();
    void execute();

private:
    void divideDatabase(Database &data);
    bool checkIfIndexOriginal(unsigned int index);
    static void deleteIndex(unsigned int index, std::vector<Object> & vec);
};

#endif // NEARESTNEIGHBOUR_H
