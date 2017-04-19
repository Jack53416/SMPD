#ifndef OBJECT_H
#define OBJECT_H

#include <string>
#include <vector>

class Object
{
private:
    std::string className;
    std::vector<float> features;


public:

    Object(const std::string &className, const std::vector<float> &features) :classID(-1), className(className), features(features)
    {
    }
    int classID;
    void setClassName(std::string name);
    std::string getClassName() const;
    size_t getFeaturesNumber() const;
    const std::vector<float> &getFeatures() const;
};



#endif // OBJECT_H
