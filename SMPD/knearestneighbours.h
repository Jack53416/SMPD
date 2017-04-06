#ifndef KNEARESTNEIGHBOURS_H
#define KNEARESTNEIGHBOURS_H
#include "Classifier.h"
#include "database.h"
#include <QTCore>

class KNearestNeighbours : public Classifier
{
public:
    KNearestNeighbours(Database &data);
};

#endif // KNEARESTNEIGHBOURS_H
