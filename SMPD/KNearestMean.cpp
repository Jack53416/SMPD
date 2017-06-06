#include "KNearestMean.h"

KNearestMean::KNearestMean(Database &data, int subclasses):
    Classifier(data), maxIterations(100)
{
    k=subclasses;
    failureRate = 0.0;

}

void KNearestMean::train(){
    if(originalSet.getNoObjects() > 0 && !crossValidation)
        divideDatabase(originalSet);

    std::vector<std::string> classNames = originalSet.getClassNames();
    std::vector<Database> separateClasses;
    std::vector<Object> meanSubclasses;

    for(int i =0; i < classNames.size(); i++)
    {
        separateClasses.push_back(getOneClass(trainingSeq,classNames.at(i)));
        concatVect(meanSubclasses,performKNM(separateClasses.at(i)));
    }

    trainingSeq.clear();
    trainingSeq = meanSubclasses;
    qDebug()<<"meansnr: "<<meanSubclasses.size();

 }

template <typename T> void KNearestMean::concatVect(std::vector<T>& a, const std::vector<T>& b)
{
    a.reserve(a.size() + b.size());
    a.insert(a.end(), b.begin(), b.end());
}

Database KNearestMean::getOneClass(std::vector<Object> &objVector, std::string className)
{
    Database result;
    for(int i = 0; i < objVector.size();i++)
    {
        if(objVector.at(i).getClassName() == className)
        {
            result.addObject(objVector.at(i));
        }
    }
    return result;
}

std::vector<Object> KNearestMean::performKNM(Database &dataSet){ //zwraca wektor srednich z podklas po KnM dla bazy danych z jedną klasą

    std::vector<Object> databaseObjects;
    std::vector<Object> subclassAverages;
    unsigned int swapsNr = 0;
    ClosestObject obj;
    Database kNMtmp;
    unsigned int iterations =0;
    //losuj randomowe srodki;
    std::vector<float> randomFeatures;
    int RN;
    qsrand(QTime::currentTime().msec());;
    for(int i =0; i< k; i++) //wygeneruj nowe srodki
    {
       do{
            RN = rand()%(dataSet.getNoObjects()-1);
            randomFeatures = dataSet.getObjects()[RN].getFeatures();
       }while(!isOriginal(randomFeatures,subclassAverages));
        subclassAverages.push_back(Object(dataSet.getClassNames().at(0)+ std::to_string(i), randomFeatures));
    }

    //wstepnie przydziel
    databaseObjects = dataSet.getObjects();

    for(int i = 0; i < databaseObjects.size(); i++)
    {
       obj = classifyObject(databaseObjects.at(i),subclassAverages);
       kNMtmp.addObject(Object(obj.obj->getClassName(),databaseObjects.at(i).getFeatures()));
       if(databaseObjects.at(i).getClassName() != obj.obj->getClassName())
           swapsNr++;
    }
    //klasyfikuj dopoki cos sie zmienia

    do{
        swapsNr = 0;
        subclassAverages.clear();
        subclassAverages = calculateMean(kNMtmp);
        databaseObjects.clear();
        databaseObjects = kNMtmp.getObjects();
        kNMtmp.clear();

        for(int i = 0; i < databaseObjects.size(); i++)
        {
           obj = classifyObject(databaseObjects.at(i),subclassAverages);
           kNMtmp.addObject(Object(obj.obj->getClassName(),databaseObjects.at(i).getFeatures()));
           if(databaseObjects.at(i).getClassName() != obj.obj->getClassName())
               swapsNr++;
        }

        iterations++;
        if(iterations >= maxIterations)
        {
            break;
        }
    }while(swapsNr > 0);

    qDebug()<<"iterations:"<<iterations;
    return subclassAverages;
}


void KNearestMean::execute()
{
    ClosestObject obj;
    std::string subclassName;
    std::string originalClassName;
    std::string currentClassName;
    for(unsigned int i = 0; i<testSeq.size(); i++ )
    {
        obj=classifyObject(testSeq.at(i),trainingSeq);
        originalClassName = testSeq.at(i).getClassName();
        currentClassName = obj.obj->getClassName();
        qDebug()<<"orginal : "<<QString::fromStdString(originalClassName)<<" Found:" << QString::fromStdString(currentClassName);
        subclassName =  obj.obj->getClassName();
        subclassName = subclassName.substr(0,subclassName.size()-1);
        if(obj.obj->getClassName().find(testSeq.at(i).getClassName()) == std::string::npos)
            failureRate++;
    }
    failureRate /= testSeq.size();
    log.push_back("Failure Rate:"+ std::to_string(failureRate));
    qDebug()<<"fRate = "<<failureRate;
}

std::vector<Object> KNearestMean::calculateMean(Database &data){

    float** sumFeatures = NULL;
    int* numberOfObjectsFromClass = new int[data.getNoClass()];
    int classId = 0;
    std::vector<std::vector<float>> dataVector(data.getNoClass(), std::vector<float>(data.getNoFeatures()));
    std::vector<Object> databaseObjects;
    std::vector<Object> result;

    databaseObjects = data.getObjects();

    sumFeatures = new float*[data.getNoClass()];
    for(unsigned int i = 0; i<data.getNoClass();i++) //preAlokacja wektora sumfeatures
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

    for(unsigned int i = 0; i<databaseObjects.size();i++)//sumujemy featury w danej klasie
    {
        for(int j = 0; j<data.getClassNames().size();j++)//sprawdzenie w ktorej jest klasie
        {
            if(!data.getClassNames().at(j).compare(databaseObjects.at(i).getClassName())){
                classId = j;
                break;
            }
        }
        for(unsigned int m = 0; m<data.getNoFeatures();m++) //sumujemy featury w danej klasie
        {
            sumFeatures[classId][m]+= databaseObjects.at(i).getFeatures().at(m);
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

    for(unsigned int i = 0; i< data.getNoClass();i++)//dodajemy srednie wartosci z klas
     {
        dataVector.at(i).assign(sumFeatures[i], sumFeatures[i] + data.getNoFeatures());
        result.push_back(Object(data.getClassNames().at(i),dataVector.at(i)));
     }

    for(unsigned int i = 0; i < data.getNoClass() ; i++)
    {
        delete sumFeatures[i];
    }
    delete sumFeatures;
    delete numberOfObjectsFromClass;

    return result;
}

ClosestObject KNearestMean::classifyObject(Object obj, std::vector<Object> &relativeSequence) //znajduje obiekt z najmniejsza odlegloscia i zwaraca jego ptr z wartoscia
{
    double tmpDist;
    ClosestObject result;
    result.distance=calculateDistance(obj,relativeSequence.at(0));
    result.obj=&relativeSequence.at(0);

    for(unsigned int i=1; i<relativeSequence.size(); i++)
    {
        tmpDist=calculateDistance(obj,relativeSequence.at(i));
        if(tmpDist < result.distance)
        {
            result.distance=tmpDist;
            result.obj=&relativeSequence.at(i);
        }
    }
    return result;
}


bool KNearestMean::isOriginal(std::vector<float> &featureVect, std::vector<Object> &sourceVect)
{
    unsigned int similarities = 0;
    const unsigned int maxSimilarities = 0.2* featureVect.size();

    for(unsigned int i = 0; i < sourceVect.size(); i++)
    {
        similarities = 0;
        for(unsigned int j = 0; j < featureVect.size(); j++)
        {
            if(featureVect.at(j) == sourceVect.at(i).getFeatures().at(j))
            {
                similarities++;
                //qDebug()<<"simi: "<<similarities;
                if(similarities > maxSimilarities)
                    return false;
            }
        }
    }
    return true;

}


std::string KNearestMean::dumpLog(bool full)
{
    std::string result;
    result += "Failure Rate: " + std::to_string(this->failureRate)+ "\n";
    return result;
}
