// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2011-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_HASHRATEGRAPHWIDGET_H
#define DYNAMIC_QT_HASHRATEGRAPHWIDGET_H

#include <QQueue>
#include <QWidget>

QT_BEGIN_NAMESPACE
class QPaintEvent;
class QTimer;
QT_END_NAMESPACE

class HashRateGraphWidget : public QWidget
{
    Q_OBJECT

public:
    explicit HashRateGraphWidget(QWidget* parent = 0);

    enum GraphType {
        MINER_CPU_HASHRATE = 0,
        MINER_GPU_HASHRATE,
        NETWORK_HASHRATE
    };

    enum SampleTime {
        FIVE_MINUTES = 0,
        TEN_MINUTES,
        THIRTY_MINUTES,
        ONE_HOUR,
        EIGHT_HOURS,
        TWELVE_HOURS,
        ONE_DAY
    };

    GraphType graphType;

public Q_SLOTS:
    void StopHashMeter();
    void StartHashMeter();
    void UpdateSampleTime(SampleTime time);
    void clear();

private:
    void timerEvent(QTimerEvent* event);
    void updateHashRateGraph();
    void initGraph(QPainter& painter);
    void drawHashRate(QPainter& painter);
    void truncateSampleQueue();
    int64_t getHashRate();

    unsigned int iDesiredSamples;
    int64_t iMaxHashRate;
    QQueue<int64_t> vSampleHashRate;
    bool fPlotHashRate;

protected:
    void paintEvent(QPaintEvent*);
};

#endif // DYNAMIC_QT_HASHRATEGRAPHWIDGET_H