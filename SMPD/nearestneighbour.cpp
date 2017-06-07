#include "nearestneighbour.h"
#include <functional>


NearestNeighbour::NearestNeighbour(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.8; //ustawianie wielkosci treningowego
    failureRate= 0.0;
}

void NearestNeighbour::train(){
    if(originalSet.getNoObjects() > 0 && !crossValidation) /// Kroswalidacja wykonuje własny podział danych
        divideDatabase(originalSet);

}

void NearestNeighbour::execute(){
    ClosestObject obj;
    for(unsigned int i = 0; i<testSeq.size(); i++ )
    {
        obj=classifyObject(this->testSeq.at(i));
        log.insert(std::make_pair(&testSeq.at(i), obj)); //tworzy log, do przekazania dla gui
        if(testSeq.at(i).getClassName() != obj.obj->getClassName())
            failureRate++;
    }
    failureRate /= testSeq.size();
}


ClosestObject NearestNeighbour::classifyObject(Object obj) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
{
    double tmpDist;
    ClosestObject result;
    result.distance=calculateDistance(obj,trainingSeq.at(0));
    result.obj=&trainingSeq.at(0);

    for(unsigned int i=1; i<trainingSeq.size(); i++)
    {
        tmpDist=calculateDistance(obj,trainingSeq.at(i));
        if(tmpDist < result.distance)
        {
            result.distance=tmpDist;
            result.obj=&trainingSeq.at(i);
        }
    }
    return result;
}

std::string NearestNeighbour::dumpLog(bool full)
{
    std::string result;
    std::map<Object*, ClosestObject>::iterator it;
    if(full){
        it = log.begin();
        while(it!= log.end())
        {

            result += "Orig class :"+ it->first->getClassName()
                    + "\nClass found:" + it->second.obj->getClassName() + "\n-----\n";
            it++;
        }
    }

    result += "Failure Rate: " + std::to_string(this->failureRate) + "\n";
    return result;
}
