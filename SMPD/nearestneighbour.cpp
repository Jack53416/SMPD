#include "nearestneighbour.h"
#include <functional>


NearestNeighbour::NearestNeighbour(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.8; //ustawianie wielkosci treningowego
    failureRate= 0.0;
}

void NearestNeighbour::train(){
    if(originalSet.getNoObjects() > 0)
        divideDatabase(originalSet);

}

void NearestNeighbour::execute(){
    ClosestObject obj;
    for(unsigned int i = 0; i<trainingSeq.size(); i++ )
    {
        obj=classifyObject(this->trainingSeq.at(i));
        log.insert(std::make_pair(&trainingSeq.at(i), obj)); //tworzy log, do przekazania dla gui
        if(trainingSeq.at(i).getClassName() != obj.obj->getClassName())
            failureRate++;
    }
    failureRate /= trainingSeq.size();
}


ClosestObject NearestNeighbour::classifyObject(Object obj) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
{
    double tmpDist;
    ClosestObject result;
    result.distance=calculateDistance(obj,testSeq.at(0));
    result.obj=&testSeq.at(0);

    for(unsigned int i=1; i<testSeq.size(); i++)
    {
        tmpDist=calculateDistance(obj,testSeq.at(i));
        if(tmpDist < result.distance)
        {
            result.distance=tmpDist;
            result.obj=&testSeq.at(i);
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
