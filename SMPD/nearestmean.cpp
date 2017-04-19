#include "nearestmean.h"
#include <functional>

NearestMean::NearestMean(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.1; //ustawianie wielkosci treningowego
    failureRate= 0.0;
}

bool NearestMean::checkIfIndexOriginal(unsigned int index)
{
    if(trainIndexes.size() == 0)
        return true;
    for(unsigned int i = 0; i< trainIndexes.size(); i++)
    {
        if(index == trainIndexes.at(i))
            return false;
    }
    return true;

}

void NearestMean::divideDatabase( Database &data)
{
    unsigned int rN;
    qsrand(QTime::currentTime().msec());

    testSeq.clear();
    trainingSeq.clear();
    trainIndexes.clear();

    testSeq = data.getObjects(); //Zbior testowy zaczyna jako cala baza
    while(trainIndexes.size() < trainingSize) //losuje indexy do zbioru treningowego
    {
        rN= qrand()%(data.getNoObjects()-1);
        if(checkIfIndexOriginal(rN))
        {
            trainIndexes.push_back(rN);
            trainingSeq.push_back(testSeq[rN]); //od razu je przypisuje do treningowego
           // this->trainingSet.addObject(testSeq[rN]);// debug
        }
    }

    std::sort(trainIndexes.begin(), trainIndexes.end(), std::greater<unsigned int>()); //sort malejacy, zeby sie dobrze usuwalo pozniej

    for(unsigned int i = 0; i< trainingSize; i++) //Usuwa z testowego, te ktore zostaly wziete do treningowego
    {
        deleteIndex(trainIndexes[i], testSeq);
    }

    calculateMean(data);

    //Debug
   /* for(unsigned int i =0; i<testSeq.size(); i++)
    {
        this->testSet.addObject(testSeq[i]); //dodano petle dla debugu
    }
    trainingSet.save("trainSet.txt"); //dodano dla debugu
    testSet.save("testSet.txt");
    qDebug()<<"TrainingSeq:"<<trainingSeq.size() << "TestSeq:" << testSeq.size()<<"BaseSize" << data.getNoObjects();*/
}

void NearestMean::deleteIndex(unsigned int index, std::vector<Object> & vec)
{
    vec.at(index) = vec.back();
    vec.pop_back();
}

void NearestMean::calculateMean(Database &data){

    float sumFeatures[data.getNoClass()][data.getNoFeatures()];
    int numberOfObjectsFromClass[data.getNoClass()];
    int classId = 0;
    std::vector<std::vector<float>> dataVector(data.getNoClass(), std::vector<float>(data.getNoFeatures()));

    for(int j = 0; j<data.getNoClass();j++) //inicializacja tablic
    {

        for(int i = 0; i<data.getNoFeatures();i++)
        {
           sumFeatures[j][i] = 0;
        }
          numberOfObjectsFromClass[j]= 0;
    }

    for(int i = 0; i<testSeq.size();i++)//sumujemy featury w danej klasie
    {
        for(int j = 0; j<data.getClassNames().size();j++)//sprawdzenie w ktorej jest klasie
        {
            if(!data.getClassNames().at(j).compare(testSeq.at(i).getClassName())){
                classId = j;
                break;
            }
        }
        for(int m = 0; m<data.getNoFeatures();m++) //sumujemy featury w danej klasie
        {
            sumFeatures[classId][m]+= testSeq.at(i).getFeatures().at(m);


        }
        numberOfObjectsFromClass[classId]++;

        classId = -1;
    }

    for(int j = 0; j<data.getNoClass();j++)//dzielimy przez liczbe obiektow z klasy
     {
         for(int i = 0; i<data.getNoFeatures();i++)
         {
            sumFeatures[j][i] = sumFeatures[j][i]/ numberOfObjectsFromClass[j];
         }
    }

    testSeq.clear(); //czyscimy zbior
    for(int i = 0; i< data.getNoClass();i++)//dodajemy srednie wartosci z klas
     {
        dataVector.at(i).assign(sumFeatures[i], sumFeatures[i] + data.getNoFeatures());
        testSeq.push_back(Object(data.getClassNames().at(i),dataVector.at(i)));
     }
}

void NearestMean::train(){
    if(originalSet.getNoObjects() > 0)
        divideDatabase(originalSet);

}

void NearestMean::execute(){
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

double NearestMean::calculateDistance(Object& startVec, Object& endVec) // liczy pierwiastek z kwadratow roznic miedzy wektorami
{
    double result = 0.0;
    double sum=0.0;
    std::vector<float> startFeatures = startVec.getFeatures();
    std::vector<float> endFeatures = endVec.getFeatures();

    for(unsigned int i =0; i< startVec.getFeaturesNumber();i++)
    {
        sum=startFeatures.at(i) - endFeatures.at(i);
        result+=sum*sum;
    }
    result = sqrt(result);

    return result;
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
