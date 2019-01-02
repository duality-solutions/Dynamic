// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2011-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <qt/hashrategraphwidget.h>

#include "guiutil.h"
#include "util.h"

#include <QColor>
#include <QPainter>
#include <QTimer>

#include <cmath>

#define DEFAULT_DESIRED_SAMPLES 60 * 5 // 5 minutes

#define XMARGIN 10
#define YMARGIN 10
#define GRID_HEIGHT 30
#define MAX_SAMPLES 60 * 60 * 24 // 24 hours

HashRateGraphWidget::HashRateGraphWidget(QWidget* parent) : QWidget(parent),
                                                            graphType(GraphType::MINER_CPU_HASHRATE),
                                                            iDesiredSamples(DEFAULT_DESIRED_SAMPLES),
                                                            iMaxHashRate(0),
                                                            vSampleHashRate(),
                                                            fPlotHashRate(false)
{
    startTimer(1000);
}

void HashRateGraphWidget::initGraph(QPainter& painter)
{
    int m_height = height();
    int m_width = width();

    QColor axisCol(Qt::gray);
    QColor firstCol(Qt::yellow);
    painter.setPen(firstCol);

    //Compute height and width steps
    int stepm_height = m_height / GRID_HEIGHT;
    //Draw horizontal lines
    for (int i = 0; i < stepm_height + 1; i++) {
        if (i > 0)
            painter.setPen(axisCol);

        painter.drawLine(4, GRID_HEIGHT * i + 2, m_width - 4, GRID_HEIGHT * i + 2);
    }
}

int64_t HashRateGraphWidget::getHashRate()
{
    switch (graphType) {
    case GraphType::MINER_CPU_HASHRATE:
        return GetCPUHashRate();
    case GraphType::MINER_GPU_HASHRATE:
        return GetGPUHashRate();
    default:
        return GetHashRate();
    }
}

void HashRateGraphWidget::drawHashRate(QPainter& painter)
{
    QPainterPath path;
    int sampleCount = vSampleHashRate.size();
    if (sampleCount > 0 && iMaxHashRate > 0) {
        int h = height() - YMARGIN * 2, w = width() - XMARGIN * 2;
        int x = XMARGIN + w;
        path.moveTo(x, YMARGIN + h);
        for (int i = 0; i < sampleCount; ++i) {
            int64_t rate = vSampleHashRate.at(i);
            x = XMARGIN + w - w * i / iDesiredSamples;
            int y = YMARGIN + h - (int)(h * rate / iMaxHashRate);
            path.lineTo(x, y);
        }
        path.lineTo(x, YMARGIN + h);
        if (graphType == MINER_CPU_HASHRATE || graphType == MINER_GPU_HASHRATE) {
            painter.fillPath(path, QColor(0, 255, 0, 128)); //green
            painter.setPen(Qt::red);
        } else if (graphType == NETWORK_HASHRATE) {
            painter.fillPath(path, QColor(255, 0, 0, 128)); //red
            painter.setPen(Qt::green);
        }
        painter.drawPath(path);
    }
    // Write axis hashrate label
    painter.setPen(Qt::yellow);
    painter.drawText(XMARGIN, YMARGIN + (GRID_HEIGHT / 2), QString("%1").arg(GUIUtil::FormatHashRate(iMaxHashRate)));
}

void HashRateGraphWidget::paintEvent(QPaintEvent*)
{
    if (!fPlotHashRate)
        return;

    QPainter painter(this);

    initGraph(painter);
    drawHashRate(painter);
}

void HashRateGraphWidget::truncateSampleQueue()
{
    while (vSampleHashRate.size() > MAX_SAMPLES) {
        vSampleHashRate.pop_back();
    }
}

void HashRateGraphWidget::updateHashRateGraph()
{
    int64_t iCurrentHashRate = 0;
    if (graphType == GraphType::MINER_CPU_HASHRATE || graphType == GraphType::MINER_GPU_HASHRATE) {
        iCurrentHashRate = getHashRate();
    } else if (graphType == GraphType::NETWORK_HASHRATE) {
        iCurrentHashRate = GUIUtil::GetNetworkHashPS(120, -1);
    }

    vSampleHashRate.push_front(iCurrentHashRate);

    if (vSampleHashRate.size() > MAX_SAMPLES + DEFAULT_DESIRED_SAMPLES) {
        truncateSampleQueue();
    }

    if (iMaxHashRate < iCurrentHashRate)
        iMaxHashRate = iCurrentHashRate;

    update();
}

void HashRateGraphWidget::timerEvent(QTimerEvent*)
{
    if (fPlotHashRate)
        updateHashRateGraph();
}

void HashRateGraphWidget::clear()
{
    iMaxHashRate = 0;
    vSampleHashRate.clear();
}

void HashRateGraphWidget::UpdateSampleTime(SampleTime time)
{
    if (time == SampleTime::FIVE_MINUTES) {
        iDesiredSamples = 60 * 5;
    } else if (time == SampleTime::TEN_MINUTES) {
        iDesiredSamples = 60 * 10;
    } else if (time == SampleTime::THIRTY_MINUTES) {
        iDesiredSamples = 60 * 30;
    } else if (time == SampleTime::ONE_HOUR) {
        iDesiredSamples = 60 * 60;
    } else if (time == SampleTime::EIGHT_HOURS) {
        iDesiredSamples = 60 * 60 * 8;
    } else if (time == SampleTime::TWELVE_HOURS) {
        iDesiredSamples = 60 * 60 * 12;
    } else if (time == SampleTime::ONE_DAY) {
        iDesiredSamples = 60 * 60 * 24;
    }
}

void HashRateGraphWidget::StopHashMeter()
{
    fPlotHashRate = false;
    update();
}

void HashRateGraphWidget::StartHashMeter()
{
    fPlotHashRate = true;
    update();
}