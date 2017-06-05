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

    for(unsigned int i = 0; i<testSeq.size();i++)//sumujemy featury w danej klasie
    {
        for(int j = 0; j<data.getClassNames().size();j++)//sprawdzenie w ktorej jest klasie
        {
            if(!data.getClassNames().at(j).compare(testSeq.at(i).getClassName())){
                classId = j;
                break;
            }
        }
        for(unsigned int m = 0; m<data.getNoFeatures();m++) //sumujemy featury w danej klasie
        {
            sumFeatures[classId][m]+= testSeq.at(i).getFeatures().at(m);

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

    testSeq.clear(); //czyscimy zbior
    for(unsigned int i = 0; i< data.getNoClass();i++)//dodajemy srednie wartosci z klas
     {
        dataVector.at(i).assign(sumFeatures[i], sumFeatures[i] + data.getNoFeatures());
        testSeq.push_back(Object(data.getClassNames().at(i),dataVector.at(i)));
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
        divideDatabase(originalSet);
        calculateMean(data1);
    }

}

void NearestMean::execute(){
    ClosestObject obj; //
    for(unsigned int i = 0; i<trainingSeq.size(); i++ )
    {
        obj=classifyObject(this->trainingSeq.at(i));
        log.insert(std::make_pair(&trainingSeq.at(i), obj)); //tworzy log, do przekazania dla gui
        if(trainingSeq.at(i).getClassName() != obj.obj->getClassName())
            failureRate++;
    }
    failureRate /= trainingSeq.size();
}


ClosestObject NearestMean::classifyObject(Object obj) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
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
    //debug
   qDebug()<<"najmnieszy euklides ="<<result.distance;
    qDebug()<<"Oryginalna klasa:"<<obj.getClassName().c_str()
           <<"Znaleziona:"<<result.obj->getClassName().c_str();
    return result;
}


double NearestMean::performCrossValidation(int K)
{
    double avgFailRate = 0.0;
    auto vect = this->originalSet.getObjects();
    std::vector<Object> mixedObj;

    srand(QTime::currentTime().msec());
    int rN =0;
    while(vect.size() > 1)  //Przelosowuje wektor aby dane byly losowo rozlozone
    {
        rN = qrand()%(vect.size()-1);
        mixedObj.push_back(vect.at(rN));
        deleteIndex(rN,vect);
    }
    mixedObj.push_back(vect.back());
    qDebug()<<"mixedObj size:"<<mixedObj.size();

    int lastIndx = 0;
    int chunk = (int)mixedObj.size()/K;
    this->trainingSize = chunk;
    this->train();

    for(int i =0; i<K; i++) // wycina kawałki wektora potrzebne do test i training sequence a następnie wykonuje na nich dany classifier
    {
        this->trainingSeq.assign(mixedObj.begin()+lastIndx,mixedObj.begin()+chunk+lastIndx);
        lastIndx+= chunk;
        this->execute();
        avgFailRate += this->failureRate;
        qDebug()<<"FR:"<<this->failureRate;
    }

    avgFailRate/= K;
    qDebug()<<"FR avg:"<<avgFailRate;
    return avgFailRate;
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

