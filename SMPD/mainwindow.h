#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <fstream>
#include <QCheckBox>
#include <QFileDialog>
#include <QtCore>
#include <QtGui>
#include <QMessageBox>

#include <QImage>
#include <QDebug>
#include <QTextBrowser>

#include"database.h"
#include "object.h"
#include "nearestneighbour.h"
#include "knearestneighbours.h"
#include "classifier.h"
#include "nearestmean.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    bool loadFile(const std::string &fileName);
    void updateDatabaseInfo(QTextBrowser* textBrowser);
    void saveFile(const std::string &fileName);

    void FSupdateButtonState(void);
    void FSsetButtonState(bool state);

    void divideDatabase();
    bool checkIfIndexOriginal(unsigned int index);


private slots:
    void on_FSpushButtonOpenFile_clicked();

    void on_FSpushButtonCompute_clicked();

    void on_FSpushButtonSaveFile_clicked();

    void on_PpushButtonSelectFolder_clicked();


    void on_CpushButtonOpenFile_clicked();

    void on_CpushButtonSaveFile_clicked();

    void on_CpushButtonTrain_clicked();

    void on_CpushButtonExecute_clicked();

    void on_comboBoxTrainingPart_currentTextChanged(const QString &arg1);

private:
    Ui::MainWindow *ui;

private:
     Database database;
     Classifier* classifier;

};

#endif // MAINWINDOW_H
