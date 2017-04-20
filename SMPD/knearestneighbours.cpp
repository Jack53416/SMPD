#include "knearestneighbours.h"

KNearestNeighbours::KNearestNeighbours(Database &data):
    Classifier(data)
{
    trainingSize=data.getNoObjects()*0.1; //ustawianie wielkosci treningowego
    failureRate= 0.0;
    k = 3;
}

bool KNearestNeighbours::checkIfIndexOriginal(unsigned int index)
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

void KNearestNeighbours::divideDatabase( Database &data)
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

    //Debug
   /* for(unsigned int i =0; i<testSeq.size(); i++)
    {
        this->testSet.addObject(testSeq[i]); //dodano petle dla debugu
    }
    trainingSet.save("trainSet.txt"); //dodano dla debugu
    testSet.save("testSet.txt");
    qDebug()<<"TrainingSeq:"<<trainingSeq.size() << "TestSeq:" << testSeq.size()<<"BaseSize" << data.getNoObjects();*/
}

void KNearestNeighbours::deleteIndex(unsigned int index, std::vector<Object> & vec)
{
    vec.at(index) = vec.back();
    vec.pop_back();
}

void KNearestNeighbours::train(){
    if(originalSet.getNoObjects() > 0)
        divideDatabase(originalSet);

}

double KNearestNeighbours::calculateDistance(Object& startVec, Object& endVec) // liczy pierwiastek z kwadratow roznic miedzy wektorami
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

ClosestObject KNearestNeighbours::classifyObject(Object obj, Database &data) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
{
    double tmpDist;
    ClosestObject result;
    ClosestObject* results = new ClosestObject[k]; ///delete!
    result.distance=calculateDistance(obj,testSeq.at(0));
    result.obj=&testSeq.at(0);
    int* classes = new int[data.getNoClass()]; /// delete!
    std::string className;
    int maxIndex = 0;
    int maxValue = 0;



    for(int i = 0; i<data.getNoClass();i++)
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

            for(int j = 0; j<data.getNoClass(); j++)
            {
                if(!results[i].obj->getClassName().compare(data.getClassNames().at(j)))
                {
                    classes[j]++;
                }
            }

    }
    result.distance = 0;
    maxValue = classes[0];
    for(int i = 0; i< data.getNoClass(); i++)
    {
        if(maxValue<classes[i])
        {
            maxValue = classes[i];
            maxIndex = i;
        }
    }
        className = data.getClassNames().at(maxIndex);
    result.obj = results[0].obj;
    result.obj->setClassName(className);
    return result;
}
void KNearestNeighbours::execute(Database &data){
    ClosestObject obj;
    for(unsigned int i = 0; i<trainingSeq.size(); i++ )
    {

        obj=classifyObject(this->trainingSeq.at(i),data);
        log.insert(std::make_pair(&trainingSeq.at(i), obj)); //tworzy log, do przekazania dla gui
        qDebug()<<"Oryginalna klasa:"<<trainingSeq.at(i).getClassName().c_str();
        qDebug()<<"zdobyta klasa:"<<obj.obj->getClassName().c_str();
        qDebug()<<"------------";
        if(trainingSeq.at(i).getClassName() != obj.obj->getClassName())
            failureRate++;
    }
    failureRate /= trainingSeq.size();
}
void KNearestNeighbours::execute(){}
