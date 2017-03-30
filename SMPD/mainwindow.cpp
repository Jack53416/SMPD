#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    classifier(NULL)
{
    ui->setupUi(this);
    FSupdateButtonState();
    ui->CplainTextEditTrainingPart->appendPlainText("20%");
    ui->CcomboBoxClassifiers->addItem("NN");
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
    classifier = new NearestNeighbour(database);

}

void MainWindow::on_CpushButtonSaveFile_clicked()
{

}

void MainWindow::on_CpushButtonTrain_clicked()
{
    if(classifier)
    {
        classifier->train();
        ui->CpushButtonExecute->setEnabled(true);
        ui->CtextBrowser->append("Training Successful!\nTrain Size:" + QString::number(classifier->getTrainSize()));
        ui->CtextBrowser->append("Test Size:" + QString::number(classifier->getTestSize()));
    }
}

void MainWindow::on_CpushButtonExecute_clicked()
{

}
