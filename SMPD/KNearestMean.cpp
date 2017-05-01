#include "KNearestMean.h"

KNearestMean::KNearestMean(Database &data, int subclasses):
    Classifier(data), maxIterations(100)
{
    k=subclasses;
    failureRate = 0.0;

}

void KNearestMean::train(){
    unsigned int iterations = 0;
    while(testSet.getNoClass() != k) // inicjalizuj dopoki nie bedzie takich srodkow podklas, zeby przynajmniej jeden element sie do nich zaklasyfikowal
    {
        initClassifier();
        if(iterations >= 10) // tylko 10, bo mega wolne
        {
            k--;
            iterations = 0;
            log.push_back("Couldn't subdivide into " + std::to_string(k) + "classes\n decreasing k");
        }
        iterations++;
        qDebug()<<"iteration: "<<iterations;
    }
    for(int i =0 ; i< testSeq.size(); i++)
    {
        trainingSet.addObject(testSeq.at(i));
    }
    trainingSet.save("averages.txt");

}

void KNearestMean::execute()
{
    ClosestObject obj;
    unsigned int iterations = 0;

    while(failureRate > 0)
    {
        failureRate = 0;
        testSeq = calculateMean(testSet);

        trainingSeq.clear();
        trainingSeq = testSet.getObjects();

        testSet.clear();

        for(int i = 0; i < trainingSeq.size(); i++)
        {
           obj = classifyObject(trainingSeq.at(i),testSeq);
           testSet.addObject(Object(obj.obj->getClassName(),trainingSeq.at(i).getFeatures()));
           if(trainingSeq.at(i).getClassName() != obj.obj->getClassName())
               failureRate++;

        }
        qDebug()<<"fR:"<<failureRate;
        iterations ++;
        if(iterations > maxIterations)
        {
            log.push_back("Exceeded max iteration nr, terminated");
            break;
        }
    }
    testSet.save("kNM.txt");
    log.push_back("Database generated in kNM.txt");
    log.push_back("Iterations: " + std::to_string(iterations));
    log.push_back("Resulting number of classes: " + std::to_string(testSet.getNoClass()));
    qDebug()<<"iterations: "<<iterations;
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


void KNearestMean::initClassifier()
{
    trainingSeq.clear();
    testSeq.clear();
    testSet.clear();
    ClosestObject obj;
    std::vector<float> randomFeatures;

    for(int i =0; i< k; i++) //wygeneruj nowe srodki
    {
        do{
        randomFeatures = generateRandomClassCenter(QTime::currentTime().msec()); // relatywnie unikalny srodek
        }while(!isOriginal(randomFeatures,testSeq));

        testSeq.push_back(Object("k"+ std::to_string(i), generateRandomClassCenter(QTime::currentTime().msec())));
    }

    trainingSeq = originalSet.getObjects();

    for(int i = 0; i < trainingSeq.size(); i++) // zaklasyfikuj obiekty do podklas
    {
       obj = classifyObject(trainingSeq.at(i),testSeq);
       testSet.addObject(Object(obj.obj->getClassName(),trainingSeq.at(i).getFeatures()));
       if(trainingSeq.at(i).getClassName() != obj.obj->getClassName())
           failureRate++;

    }
    testSet.save("kNM.txt");
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



std::vector<float> KNearestMean::generateRandomClassCenter(int seed)
{
    std::vector<float> result;
    std::vector<Object> databaseObjects = originalSet.getObjects();

    unsigned int rObj = 0.0;
    unsigned int rFeat = 0.0;

    qsrand(seed);

    for(unsigned int i = 0; i < originalSet.getNoFeatures(); i++)
    {
        rObj = rand() % (databaseObjects.size()-1);
        rFeat = rand() % (originalSet.getNoFeatures() - 1);
        result.push_back(databaseObjects.at(rObj).getFeatures().at(rFeat));
    }
    return result;
}

bool KNearestMean::isOriginal(std::vector<float> featureVect, std::vector<Object> sourceVect)
{
    unsigned int similarities = 0;
    const unsigned int maxSimilarities = 0.9 * featureVect.size();

    for(unsigned int i = 0; i < sourceVect.size(); i++)
    {
        similarities = 0;
        for(unsigned int j = 0; j < featureVect.size(); j++)
        {
            if(featureVect.at(j) == sourceVect.at(i).getFeatures().at(j))
            {
                similarities++;
                if(similarities > maxSimilarities)
                    return false;
            }
        }
    }
    return true;

}
