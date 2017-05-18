#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "matrixutil.hpp"

namespace bnu = boost::numeric::ublas;

std::vector<std::vector<int>> comb(int N, int K)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    std::vector<std::vector<int>> vect;
    int counter = 0;

    // print integers and permute bitmask
    do {
        vect.push_back(std::vector<int>());
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i])
            {
               // std::cout << " " << i;
                vect.at(counter).push_back(i);
            }

        }
        //std::cout << std::endl;
        counter++;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return vect;
}


int determinant_sign(const bnu::permutation_matrix<std::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}

double determinant( bnu::matrix<double>& m ) {
    bnu::permutation_matrix<std::size_t> pm(m.size1());
    double det = 1.0;
    if( bnu::lu_factorize(m,pm) ) {
        det = 0.0;
    } else {
        for(int i = 0; i < m.size1(); i++)
            det *= m(i,i); // multiply by elements on diagonal
        det = det * determinant_sign( pm );
    }
    return det;
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    classifier(NULL)
{
    ui->setupUi(this);
    FSupdateButtonState();
    ui->CcomboBoxClassifiers->addItem("NN");
    ui->CcomboBoxClassifiers->addItem("kNN");
    ui->CcomboBoxClassifiers->addItem("NM");
    ui->CcomboBoxClassifiers->addItem("kNM");
    ui->CcomboBoxK->addItem("3");
    ui->CcomboBoxK->addItem("5");
    ui->CcomboBoxK->addItem("7");
    ui->CcomboBoxK->addItem("9");

    for(int i = 10; i<100; i+=10)
    {
        ui->comboBoxTrainingPart->addItem(QString::number(i));
    }
}

MainWindow::~MainWindow()
{
    delete ui;
    if(classifier)
        delete classifier;
}

void MainWindow::updateDatabaseInfo(QTextBrowser* textBrowser)
{
    textBrowser->setText("noClass: " +  QString::number(database.getNoClass()));
    textBrowser->append("noObjects: "  +  QString::number(database.getNoObjects()));
    textBrowser->append("noFeatures: "  +  QString::number(database.getNoFeatures()));
}

void MainWindow::FSupdateButtonState(void)
{
    if(database.getNoObjects()==0)
    {
        FSsetButtonState(false);
    }
    else
        FSsetButtonState(true);

}


void MainWindow::FSsetButtonState(bool state)
{
   ui->FScomboBox->setEnabled(state);
   ui->FSpushButtonCompute->setEnabled(state);
   ui->FSpushButtonSaveFile->setEnabled(state);
   ui->FSradioButtonFisher->setEnabled(state);
   ui->FSradioButtonSFS->setEnabled(state);
}

void MainWindow::on_FSpushButtonOpenFile_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open TextFile"), "", tr("Texts Files (*.txt)"));

    if ( !database.load(fileName.toStdString()) )
        QMessageBox::warning(this, "Warning", "File corrupted !!!");
    else
        QMessageBox::information(this, fileName, "File loaded !!!");

    FSupdateButtonState();
    ui->FScomboBox->clear();
    for(unsigned int i=1; i<=database.getNoFeatures(); ++i)
        ui->FScomboBox->addItem(QString::number(i));

    updateDatabaseInfo(ui->FStextBrowserDatabaseInfo);
}

void MainWindow::on_FSpushButtonCompute_clicked()
{
    int dimension = ui->FScomboBox->currentText().toInt();

    bnu::matrix<double> m(3, 3);
        for (unsigned i = 0; i < m.size1() ; ++i) {
            for (unsigned j = 0; j < m.size2() ; ++j) {
                m (i, j) = 3 * i + sqrt(j+1); // fill matrix
                m(i,j) = m(i,j)*m(i,j);       // with some numbers
            }
        }
        double k = determinant(m);
        std::cout<<k<<std::endl;


    if( ui->FSradioButtonFisher ->isChecked())
    {
    if (dimension == 1 && database.getNoClass() == 2)
        {
            float FLD = 0, tmp;
            int max_ind = -1;

            //std::map<std::string, int> classNames = database.getClassNames();
            for (uint i = 0; i < database.getNoFeatures(); ++i)
            {
                std::map<std::string, float> classAverages;
                std::map<std::string, float> classStds;

                for (auto const &ob : database.getObjects())
                {
                    classAverages[ob.getClassName()] += ob.getFeatures()[i];
                    classStds[ob.getClassName()] += ob.getFeatures()[i] * ob.getFeatures()[i];
                }

                std::for_each(database.getClassCounters().begin(), database.getClassCounters().end(), [&](const std::pair<std::string, int> &it)
                {
                    classAverages[it.first] /= it.second;
                    classStds[it.first] = std::sqrt(classStds[it.first] / it.second - classAverages[it.first] * classAverages[it.first]);
                }
                );

                tmp = std::abs(classAverages[ database.getClassNames()[0] ] - classAverages[database.getClassNames()[1]]) / (classStds[database.getClassNames()[0]] + classStds[database.getClassNames()[1]]);

                if (tmp > FLD)
                {
                    FLD = tmp;
                    max_ind = i;
                }


              }

            ui->FStextBrowserDatabaseInfo->append("max_ind: "  +  QString::number(max_ind) + " " + QString::number(FLD));
          }else{

                    int max_ind = 0;
                    double FLD = 0;
                    double tmp;
                    std::vector<int> bestFeatures(dimension);


                    std::vector<double> classAAvg;
                    std::vector<double> classBAvg;

                    std::vector<Object> classAMembers;
                    std::vector<Object> classBMembers;
                    std::vector<double> classAStd;
                    std::vector<double> classBStd;


                    for (uint i = 0; i < database.getNoFeatures(); ++i)
                    {
                        std::map<std::string, float> classAverages;
                        std::map<std::string, float> classStds;

                        for (auto const &ob : database.getObjects())
                        {
                            classAverages[ob.getClassName()] += ob.getFeatures()[i];
                            classStds[ob.getClassName()] += ob.getFeatures()[i] * ob.getFeatures()[i];

                        }

                        std::for_each(database.getClassCounters().begin(), database.getClassCounters().end(), [&](const std::pair<std::string, int> &it)
                        {
                            classAverages[it.first] /= it.second;
                            classStds[it.first] = std::sqrt(classStds[it.first] / it.second - classAverages[it.first] * classAverages[it.first]);
                        }
                        );


                        tmp = std::abs(classAverages[ database.getClassNames()[0] ] - classAverages[database.getClassNames()[1]]) / (classStds[database.getClassNames()[0]] + classStds[database.getClassNames()[1]]);

                        //std::cout<<classAverages[database.getClassNames()[0]]<<std::endl;
                        //std::cout<<classStds[database.getClassNames()[0]]<<std::endl;

                        classAAvg.push_back(classAverages[database.getClassNames()[0]]);
                        classBAvg.push_back(classAverages[database.getClassNames()[1]]);


                        classAStd.push_back(classStds[database.getClassNames()[0]]);
                        classBStd.push_back(classStds[database.getClassNames()[1]]);

                    }


                    for (auto const &ob : database.getObjects())
                    {
                        if(ob.getClassName() == database.getClassNames()[0])
                        {
                            classAMembers.push_back(ob);
                        }else{
                            classBMembers.push_back(ob);
                        }

                    }


                    bnu::matrix<double> a(dimension,classAMembers.size());
                    bnu::matrix<double> b(dimension,classBMembers.size());
                    bnu::matrix <double> aResult(dimension,dimension);
                    bnu::matrix<double> bResult(dimension,dimension);
                    double detA = 0;
                    double detB = 0;
                    double z = 0;

                    std::vector<std::vector<int>> combinations;
                    combinations = comb(database.getNoFeatures(),dimension);

                    std::cout<<combinations.size()<<std::endl;

                    for(int m = 0; m<combinations.size(); m++)
                    {
                            detA = 0;
                            detB = 0;
                            z = 0;


                            for(int j = 0; j< dimension ; j++)
                            {
                                //std::cout<<combinations.at(m).at(j)<<std::endl;
                                for(int i = 0; i<classAMembers.size(); i++)
                                {
                                    //std::cout<<classAAvg.at(combinations.at(m).at(j))<<std::endl;
                                    //std::cout<<classAMembers.at(i).getFeatures()[combinations.at(m).at(j)]<<std::endl;
                                    a(j,i) = classAMembers.at(i).getFeatures()[combinations.at(m).at(j)] - classAAvg.at(combinations.at(m).at(j));
                                }
                            }
                            for(int j = 0; j< dimension ; j++)
                            {

                                for(int i = 0; i<classBMembers.size(); i++)
                                {
                                    b(j,i) = classBMembers.at(i).getFeatures()[combinations.at(m).at(j)] - classBAvg.at(combinations.at(m).at(j));
                                }
                            }

                            for(int i = 0; i<dimension; i++)
                            {
                                //std::cout<<"klasa a srednia "<< combinations.at(m).at(i) <<classAAvg.at(combinations.at(m).at(i))<<std::endl;
                                //std::cout<<"klasa b srednia"<< combinations.at(m).at(i) <<classBAvg.at(combinations.at(m).at(i))<<std::endl;
                                z+= (classAAvg.at(combinations.at(m).at(i))-classBAvg.at(combinations.at(m).at(i)))*(classAAvg.at(combinations.at(m).at(i))-classBAvg.at(combinations.at(m).at(i)));
                            }

                            z=sqrt(z);
                            //std::cout<<"roznica srednich"<<z<<std::endl;
                            //std::cout<<b<<std::endl;
                            axpy_prod(a, trans(a), aResult, true);
                            axpy_prod(b, trans(b), bResult, true);

                            //std::cout<<aResult<<std::endl;
                            //std::cout<<bResult<<std::endl;

                            detA = determinant(aResult);
                            detB = determinant(bResult);

                            z/=(detA+detB);
                            //std::cout<<"wspolczynnik fiszera "<<z<<std::endl;
                            if(z>FLD)
                            {
                               //std::cout<<"asdasd"<<std::endl;
                               FLD = z;
                               bestFeatures.clear();
                               bestFeatures.insert( bestFeatures.end(), combinations.at(m).begin(), combinations.at(m).end() );
                            }


                      }
                     ui->FStextBrowserDatabaseInfo->append("bestFeatures:");
                    for(int i = 0; i<bestFeatures.size(); i++)
                    {
                        ui->FStextBrowserDatabaseInfo->append(QString::number(bestFeatures.at(i)));
                    }

                    ui->FStextBrowserDatabaseInfo->append("Fisher coefficient:" + QString::number(FLD));

                }



    }
}



void MainWindow::on_FSpushButtonSaveFile_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
    tr("Open TextFile"), "D:\\Users\\Krzysiu\\Documents\\Visual Studio 2015\\Projects\\SMPD\\SMPD\\Debug\\", tr("Texts Files (*.txt)"));

        QMessageBox::information(this, "My File", fileName);
        database.save(fileName.toStdString());
}

void MainWindow::on_PpushButtonSelectFolder_clicked()
{
}

void MainWindow::on_CpushButtonOpenFile_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open TextFile"), "", tr("Texts Files (*.txt)"));

    if ( !database.load(fileName.toStdString()) )
    {
        ui->CpushButtonTrain->setEnabled(false);
        ui->CpushButtonExecute->setEnabled(false);
        QMessageBox::warning(this, "Warning", "File corrupted !!!");
    }
    else
    {
        QMessageBox::information(this, fileName, "File loaded !!!");
        ui->CpushButtonTrain->setEnabled(true);
    }

    updateDatabaseInfo(ui->CtextBrowser);
    if(classifier)
        delete classifier;
   // classifier = new NearestNeighbour(database);


}

void MainWindow::on_CpushButtonSaveFile_clicked()
{

}

void MainWindow::on_CpushButtonTrain_clicked()
{
    int classifierChoice = ui->CcomboBoxClassifiers->currentIndex();
    switch(classifierChoice)
    {
        case 0: //nn
        classifier = new NearestNeighbour(database);
        break;
        case 1: //knn
        classifier = new KNearestNeighbours(database);
        break;
        case 2: //nm
        classifier = new NearestMean(database);
        break;
        case 3: //knM
        classifier = new KNearestMean(database, ui->CcomboBoxK->currentText().toInt());
        break;

    }

    if(classifier)
    {
        classifier->setTrainSize(ui->comboBoxTrainingPart->currentText().toInt());
        classifier->train();
        ui->CpushButtonExecute->setEnabled(true);
        ui->CtextBrowser->append("Training Successful!\nTrain Size:" + QString::number(classifier->getTrainSize()));
        ui->CtextBrowser->append("Test Size:" + QString::number(classifier->getTestSize()));
        ui->CtextBrowser->append(QString::number(ui->comboBoxTrainingPart->currentText().toInt()));
    }
}

void MainWindow::on_CpushButtonExecute_clicked()
{
    std::map<Object* , ClosestObject>::iterator it;
    int classifierChoice = ui->CcomboBoxClassifiers->currentIndex();

    NearestNeighbour* ptrNN;
    KNearestNeighbours* ptrKNN;
    NearestMean* ptrNM;
    KNearestMean* ptrKNM;
    if(classifier)
    {
        switch(classifierChoice)
        {
            case 0: //nn

                ptrNN = (NearestNeighbour*) classifier;
                ptrNN->execute();
                it = ptrNN->log.begin();
                while(it != ptrNN->log.end()) // dla wyswietlania pelnego loga
                {
                    ui->CtextBrowser->append("Orig cs:"+ QString::fromStdString(it->first->getClassName())
                                             + " Cs found:" + QString::fromStdString(it->second.obj->getClassName())
                                             + " dist = " + QString::number(it->second.distance));
                    it++;
                }
                ui->CtextBrowser->append("failure rate =" + QString::number(classifier->getFailRate()));
                break;

             case 1: //knn

                ptrKNN = (KNearestNeighbours*) classifier;
                ui->CtextBrowser->append(QString::number(classifierChoice));
                ptrKNN->k = ui->CcomboBoxK->currentText().toInt();
                ptrKNN->execute(database);
                it = ptrKNN->log.begin();
                while(it != ptrKNN->log.end()) // dla wyswietlania pelnego loga
                {
                   ui->CtextBrowser->append("Orig cs:"+ QString::fromStdString(it->first->getClassName())
                                             + " Cs found:" + QString::fromStdString(it->second.obj->getClassName())
                                             + " dist = " + QString::number(it->second.distance));
                    it++;
                }
                ui->CtextBrowser->append("knnfailure rate =" + QString::number(classifier->getFailRate()));
                break;

            case 2: //nm

                ptrNM = (NearestMean*) classifier;
                ptrNM->execute();
                it = ptrNM->log.begin();
                while(it != ptrNM->log.end()) // dla wyswietlania pelnego loga
                {
                    ui->CtextBrowser->append("Orig cs:"+ QString::fromStdString(it->first->getClassName())
                                             + " Cs found:" + QString::fromStdString(it->second.obj->getClassName())
                                             + " dist = " + QString::number(it->second.distance));
                    it++;
                }
                ui->CtextBrowser->append("failure rate =" + QString::number(classifier->getFailRate()));
                break;
        case 3: //knM
            ptrKNM = (KNearestMean*) classifier;
            classifier->execute();
            for(int i = 0; i <ptrKNM->log.size(); i++ )
            {
                ui->CtextBrowser->append(QString::fromStdString(ptrKNM->log.at(i)));
            }
            break;

        }
    }
    delete classifier;
    classifier = NULL;
    ui->CpushButtonExecute->setEnabled(false);

}

void MainWindow::on_comboBoxTrainingPart_currentTextChanged(const QString &arg1)
{
    if(classifier)
    {
        classifier->setTrainSize(arg1.toInt());
    }
}
