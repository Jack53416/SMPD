#include "nearestmean.h"
#include <functional>

NearestMean::NearestMean(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.1; //ustawianie wielkosci treningowego
    failureRate= 0.0;
    data1 = data;
}

void NearestMean::calculateMean(Database &data){

    float** sumFeatures = NULL;
    int* numberOfObjectsFromClass = new int[data.getNoClass()];
    int classId = 0;
    std::vector<std::vector<float>> dataVector(data.getNoClass(), std::vector<float>(data.getNoFeatures()));

    sumFeatures = new float*[data.getNoClass()];
    for(unsigned int i = 0; i<data.getNoClass();i++)
    {
        sumFeatures[i] = new float[data.getNoFeatures()];
    }

    for(unsigned int j = 0; j<data.getNoClass();j++) //inicializacja tablic
    {

        for(unsigned int i = 0; i<data.getNoFeatures();i++)
        {
           sumFeatures[j][i] = 0;
        }
          numberOfObjectsFromClass[j]= 0;
    }

    for(unsigned int i = 0; i<trainingSeq.size();i++)//sumujemy featury w danej klasie
    {
        for(int j = 0; j<data.getClassNames().size();j++)//sprawdzenie w ktorej jest klasie
        {
            if(!data.getClassNames().at(j).compare(trainingSeq.at(i).getClassName())){
                classId = j;
                break;
            }
        }
        for(unsigned int m = 0; m<data.getNoFeatures();m++) //sumujemy featury w danej klasie
        {
            sumFeatures[classId][m]+= trainingSeq.at(i).getFeatures().at(m);

        }
        numberOfObjectsFromClass[classId]++;

        classId = -1;
    }

    for(unsigned int j = 0; j<data.getNoClass();j++)//dzielimy przez liczbe obiektow z klasy
     {
         for(unsigned int i = 0; i<data.getNoFeatures();i++)
         {
            sumFeatures[j][i] = sumFeatures[j][i]/ numberOfObjectsFromClass[j];
         }
    }

    trainingSeq.clear(); //czyscimy zbior
    for(unsigned int i = 0; i< data.getNoClass();i++)//dodajemy srednie wartosci z klas
     {
        dataVector.at(i).assign(sumFeatures[i], sumFeatures[i] + data.getNoFeatures());
        trainingSeq.push_back(Object(data.getClassNames().at(i),dataVector.at(i)));
     }

    for(unsigned int i = 0; i < data.getNoClass() ; i++)
    {
        delete sumFeatures[i];
    }
    delete sumFeatures;
    delete numberOfObjectsFromClass;
}

void NearestMean::train(){
    if(originalSet.getNoObjects() > 0)
    {
        if(!crossValidation) /// Kroswalidacja wykonuje własny podział danych
            divideDatabase(originalSet);

        calculateMean(data1);
    }

}

void NearestMean::execute(){
    ClosestObject obj; //
    for(unsigned int i = 0; i<testSeq.size(); i++ )
    {
        obj=classifyObject(this->testSeq.at(i));
        log.insert(std::make_pair(&testSeq.at(i), obj)); //tworzy log, do przekazania dla gui
        if(testSeq.at(i).getClassName() != obj.obj->getClassName())
            failureRate++;
    }
    failureRate /= testSeq.size();
}


ClosestObject NearestMean::classifyObject(Object obj) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
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
    //debug
   qDebug()<<"najmnieszy euklides ="<<result.distance;
    qDebug()<<"Oryginalna klasa:"<<obj.getClassName().c_str()
           <<"Znaleziona:"<<result.obj->getClassName().c_str();
    return result;
}



std::string NearestMean::dumpLog(bool full)
{
    std::string result;
    std::map<Object*, ClosestObject>::iterator it;
    it = log.begin();
    if(full)
    {
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

