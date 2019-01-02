// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef MININGPAGE_H
#define MININGPAGE_H

#include "platformstyle.h"

#include "walletmodel.h"

#include <QPushButton>
#include <QWidget>

#include <memory>

namespace Ui
{
class MiningPage;
}

class MiningPage : public QWidget
{
    Q_OBJECT

public:
    explicit MiningPage(const PlatformStyle* platformStyle, QWidget* parent = 0);
    ~MiningPage();

    void setModel(WalletModel* model);

private:
    Ui::MiningPage* ui;
    WalletModel* model;
    std::unique_ptr<WalletModel::UnlockContext> unlockContext;
    bool hasMiningprivkey;
    bool fGPUMinerOn;
    bool fCPUMinerOn;
    void timerEvent(QTimerEvent* event);
    void updateUI();
    void StartCPUMiner();
    void StopCPUMiner();
    void showCPUHashMeterControls(bool show);
    void updateCPUPushSwitch();
#ifdef ENABLE_GPU
    void StartGPUMiner();
    void StopGPUMiner();
    void showGPUHashMeterControls(bool show);
    void updateGPUPushSwitch();
#endif

    void updatePushSwitch(QPushButton* pushSwitch, bool minerOn);

    bool isMinerOn();

private Q_SLOTS:

    void changeNumberOfCPUThreads(int i, bool shutdown = false);
    void switchCPUMining();
    void showCPUHashRate(int i);
    void changeCPUSampleTime(int i);
    void clearCPUHashRateData();


#ifdef ENABLE_GPU
    void changeNumberOfGPUThreads(int i, bool shutdown = false);
    void switchGPUMining();
    void showGPUHashRate(int i);
    void changeGPUSampleTime(int i);
    void clearGPUHashRateData();
#endif
};

#endif // MININGPAGE_H
