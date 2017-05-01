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
    std::vector<std::string> log;

private:
   const unsigned int maxIterations;
   void initClassifier();
   std::vector<float> generateRandomClassCenter(int seed);
   ClosestObject classifyObject(Object obj,  std::vector<Object> &relativeSequence);
   std::vector<Object> calculateMean(Database &data);
   bool isOriginal(std::vector<float> featureVect, std::vector<Object> sourceVect);
};

#endif // KNEARESTMEAN_H
