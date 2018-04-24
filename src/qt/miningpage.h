// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef MININGPAGE_H
#define MININGPAGE_H

#include "platformstyle.h"

#include "walletmodel.h"

#include <QWidget>
#include <memory>

namespace Ui {
class MiningPage;
}

class MiningPage : public QWidget
{
    Q_OBJECT

public:
    explicit MiningPage(const PlatformStyle *platformStyle, QWidget *parent = 0);
    ~MiningPage();

    void setModel(WalletModel *model);

private:
    Ui::MiningPage *ui;
    WalletModel *model;
    std::unique_ptr<WalletModel::UnlockContext> unlockContext;
    bool hasMiningprivkey;

    void timerEvent(QTimerEvent *event);
    void updateUI();
    void StartMiner();
    void StopMiner();
    void showHashMeterControls(bool show);

private Q_SLOTS:

    void changeNumberOfCores(int i);
    void switchMining();
    void showHashRate(int i);
    void changeSampleTime(int i);
    void clearHashRateData();
};

#endif // MININGPAGE_H
