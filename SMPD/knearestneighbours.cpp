#include "knearestneighbours.h"

KNearestNeighbours::KNearestNeighbours(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.1; //ustawianie wielkosci treningowego
    failureRate= 0.0;
}
