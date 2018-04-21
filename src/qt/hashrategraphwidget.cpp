// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2011-2018 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <qt/hashrategraphwidget.h>

#include "guiutil.h"
#include "util.h"

#include <QPainter>
#include <QColor>
#include <QTimer>

#include <cmath>

#define DESIRED_SAMPLES         800

#define XMARGIN                 10
#define YMARGIN                 10

HashRateGraphWidget::HashRateGraphWidget(QWidget *parent) :
    QWidget(parent),
    iMaxMyHashRate(0),
    iMaxNetworkHashRate(0),
    vSampleMyHashRate(),
    vSampleNetworkHashRate(),
    fPlotHashRate(true)
{

}

void HashRateGraphWidget::initGraph(QPainter& painter)
{
    int m_height = height();
    int m_width = width();
    int m_gridHeightStep = 30;

    QColor axisCol(Qt::gray);
    painter.setPen(axisCol);

    //Compute height and width steps
    int stepm_height = m_height/m_gridHeightStep;
    //Draw horizontal lines
    for (int i = 0; i < stepm_height +1; i++)
    {
      painter.drawLine(4,m_gridHeightStep*i+2,m_width-4,
                  m_gridHeightStep*i+2);
    }
}

void HashRateGraphWidget::plotMyHashRate(QPainter& painter)
{
    QPainterPath path;
    int sampleCount = vSampleMyHashRate.size();
    if(sampleCount > 0) {
        int h = height() - YMARGIN * 2, w = width() - XMARGIN * 2;
        int x = XMARGIN + w;
        path.moveTo(x, YMARGIN + h);
        for(int i = 0; i < sampleCount; ++i) {
            if (vSampleMyHashRate.at(i) > 0) {
                x = XMARGIN + w - w * i / DESIRED_SAMPLES;
                int y = YMARGIN + h - (int)(h * vSampleMyHashRate.at(i) / iMaxMyHashRate);
                path.lineTo(x, y);
            }
        }
        path.lineTo(x, YMARGIN + h);
        painter.fillPath(path, QColor(0, 255, 0, 128));
        painter.setPen(Qt::green);
        painter.drawPath(path);
    }
}

void HashRateGraphWidget::plotNetworkHashRate(QPainter& painter)
{
    QPainterPath path;
    int sampleCount = vSampleNetworkHashRate.size();
    if(sampleCount > 0) {
        int h = height() - YMARGIN * 2, w = width() - XMARGIN * 2;
        int x = XMARGIN + w;
        path.moveTo(x, YMARGIN + h);
        for(int i = 0; i < sampleCount; ++i) {
            if (vSampleNetworkHashRate.at(i) > 0) {
                x = XMARGIN + w - w * i / DESIRED_SAMPLES;
                int y = YMARGIN + h - (int)(h * vSampleNetworkHashRate.at(i) / iMaxNetworkHashRate);
                path.lineTo(x, y);
            }
        }
        path.lineTo(x, YMARGIN + h);
        painter.fillPath(path, QColor(255, 0, 0, 128));
        painter.setPen(Qt::red);
        painter.drawPath(path);
    }
}

void HashRateGraphWidget::paintEvent(QPaintEvent *)
{
    if (!fPlotHashRate)
        return;

    QPainter painter(this);
    
    initGraph(painter);
    plotMyHashRate(painter);
    plotNetworkHashRate(painter);
}

void HashRateGraphWidget::updateHashRateGraph()
{
    if (fPlotHashRate) {
        int64_t currentHashRate = GUIUtil::GetHashRate();
        int64_t currentNetworkHashRate = GUIUtil::GetHashRate();
        
        vSampleMyHashRate.push_front(currentHashRate);
        vSampleNetworkHashRate.push_front(currentNetworkHashRate);

        if (iMaxMyHashRate < currentHashRate)
            iMaxMyHashRate = currentHashRate;

        if (iMaxNetworkHashRate < currentNetworkHashRate)
            iMaxNetworkHashRate = currentNetworkHashRate;

        if (iMaxMyHashRate == 0)
            return;

        update();
    }
}

void HashRateGraphWidget::stopHashMeter()
{
    fPlotHashRate = false;
    clear();
    update();
}

void HashRateGraphWidget::startHashMeter()
{
    fPlotHashRate = true;
    clear();
    update();
}

void HashRateGraphWidget::clear()
{
    iMaxMyHashRate = 0;
    iMaxNetworkHashRate = 0;
    vSampleMyHashRate.clear();
    vSampleNetworkHashRate.clear();
}