#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    customPlot = new QCustomPlot(this); // Initialize custom plot
    setCentralWidget(customPlot); // Set QCustomPlot as the central widget of the window

    // Add a graph to the plot
    customPlot->addGraph();
    
    // Set some sample data (x, y values)
    QVector<double> x(101), y(101);
    for (int i = 0; i < 101; ++i)
    {
        x[i] = i / 50.0 - 1; // x goes from -1 to 1
        y[i] = x[i] * x[i];   // y = x^2
    }
    
    // Set graph data
    customPlot->graph(0)->setData(x, y);
    
    // Set labels for the axes
    customPlot->xAxis->setLabel("X Axis");
    customPlot->yAxis->setLabel("Y Axis");

    // Set range of the axes
    customPlot->xAxis->setRange(-1, 1);
    customPlot->yAxis->setRange(0, 1);
    
    // Replot the graph to update
    customPlot->replot();
}

MainWindow::~MainWindow()
{
    delete ui;
}
