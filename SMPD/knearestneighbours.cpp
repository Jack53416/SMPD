#include "knearestneighbours.h"

KNearestNeighbours::KNearestNeighbours(Database &data, int input):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.1; //ustawianie wielkosci treningowego
    failureRate= 0.0;
    k = input;
}


void KNearestNeighbours::train(){
    if(originalSet.getNoObjects() > 0)
        divideDatabase(originalSet);

}

ClosestObject KNearestNeighbours::classifyObject(Object obj, Database &data) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
{
    ClosestObject result;
    ClosestObject* results = new ClosestObject[k];

    int* classes = new int[data.getNoClass()];

    double tmpDist;
    int maxIndex = 0;
    int maxValue = 0;
    std::string className;

    result.distance=calculateDistance(obj,testSeq.at(0));
    result.obj=&testSeq.at(0);


    for(unsigned int i = 0; i<data.getNoClass();i++)
    {
        classes[i] = 0;
    }

    for(int i = 0; i<k; i++)
    {
        results[i].distance=2147483647;
        results[i].obj=&testSeq.at(i);
    }

    for(unsigned int i = 0; i<testSeq.size(); i++)
    {
        tmpDist=calculateDistance(obj,testSeq.at(i));
        if(tmpDist < results[k-1].distance)
        {
            results[k-1].distance=tmpDist;
            results[k-1].obj=&testSeq.at(i);
            std::sort(results, results + k,
                      [](ClosestObject const & a, ClosestObject const & b) -> bool
                      { return a.distance < b.distance; } );

        }
    }
    for(int i = 0; i<k; i++)
    {
       // qDebug()<<"klasa sąsiada nr:"<<k<< "to: " <<results[i].obj->getClassName().c_str();

            for(unsigned int j = 0; j<data.getNoClass(); j++)
            {
                if(!results[i].obj->getClassName().compare(data.getClassNames().at(j)))
                {
                    classes[j]++;
                }
            }

    }
    result.distance = 0;
    maxValue = classes[0];
    if(maxValue<classes[1])
    {
        maxValue = classes[1];
        maxIndex = 1;
        className = data.getClassNames().at(maxIndex);
    }else{
        if(maxValue == classes[1])
        {
            className = results[0].obj->getClassName();

        }else{
            className = data.getClassNames().at(maxIndex);
        }
    }
    result.obj = results[0].obj;
    result.obj->setClassName(className);

    if(results)
        free(results);
    if(classes)
        free(classes);
    return result;
}
void KNearestNeighbours::execute()
{
    ClosestObject obj;
    for(unsigned int i = 0; i<trainingSeq.size(); i++ )
    {

        obj=classifyObject(this->trainingSeq.at(i),originalSet);
        log.insert(std::make_pair(&trainingSeq.at(i), obj)); //tworzy log, do przekazania dla gui
        qDebug()<<"Oryginalna klasa:"<<trainingSeq.at(i).getClassName().c_str();
        qDebug()<<"zdobyta klasa:"<<obj.obj->getClassName().c_str();
        qDebug()<<"------------";
        if(trainingSeq.at(i).getClassName() != obj.obj->getClassName())
            failureRate++;
    }
    failureRate /= trainingSeq.size();

}

std::string KNearestNeighbours::dumpLog(bool full)
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
    result += "K: " + std::to_string(this->k) + "\n-----\n";
    result += "Failure Rate: " + std::to_string(this->failureRate) + "\n";

    return result;
}
