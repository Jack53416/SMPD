#ifndef KNEARESTMEAN_H
#define KNEARESTMEAN_H

#include "Classifier.h"
#include <QTCore>

class KNearestMean : public Classifier
{
public:
    KNearestMean(Database &data, int subclasses);
    void train();
    void execute();
    std::string dumpLog(bool full);

    std::vector<std::string> log;

private:
   const unsigned int maxIterations;
   std::vector<float> generateRandomClassCenter(int seed);
   ClosestObject classifyObject(Object obj,  std::vector<Object> &relativeSequence);
   std::vector<Object> calculateMean(Database &data);
   bool isOriginal(std::vector<float> &featureVect, std::vector<Object> &sourceVect);

   Database getOneClass(std::vector<Object> &objVector, std::string className);
   std::vector<Object> performKNM(Database& dataSet);
   template <typename T> void concatVect(std::vector<T>& a, const std::vector<T>& b);

};

#endif // KNEARESTMEAN_H
