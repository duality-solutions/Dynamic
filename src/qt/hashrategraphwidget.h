// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2011-2018 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_HASHRATEGRAPHWIDGET_H
#define DYNAMIC_QT_HASHRATEGRAPHWIDGET_H

#include <QWidget>
#include <QQueue>

QT_BEGIN_NAMESPACE
class QPaintEvent;
class QTimer;
QT_END_NAMESPACE

class HashRateGraphWidget : public QWidget
{
    Q_OBJECT

public:
    explicit HashRateGraphWidget(QWidget *parent = 0);
    
public Q_SLOTS:
    void updateHashRateGraph();
    void stopHashMeter();
    void startHashMeter();

private:
    void initGraph(QPainter& painter);
    void plotMyHashRate(QPainter& painter);
    void plotNetworkHashRate(QPainter& painter);
    void clear();

    int64_t iMaxMyHashRate;
    int64_t iMaxNetworkHashRate;
    QQueue<int64_t> vSampleMyHashRate;
    QQueue<int64_t> vSampleNetworkHashRate;
    bool fPlotHashRate;

protected :
    void paintEvent(QPaintEvent *);
};

#endif // DYNAMIC_QT_HASHRATEGRAPHWIDGET_H