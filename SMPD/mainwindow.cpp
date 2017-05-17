#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "matrixutil.hpp"

namespace bnu = boost::numeric::ublas;

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
        std::cout<<m;


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
