#include "miningpage.h"
#include "ui_miningpage.h"

#include "dynode-sync.h"
#include "guiutil.h"
#include "miner/miner.h"
#include "net.h"
#include "util.h"
#include "utiltime.h"
#include "validation.h"
#include "walletmodel.h"

#include <boost/thread.hpp>
#include <stdio.h>

MiningPage::MiningPage(const PlatformStyle* platformStyle, QWidget* parent) : QWidget(parent),
                                                                              ui(new Ui::MiningPage),
                                                                              hasMiningprivkey(false)
{
    ui->setupUi(this);

    int nCPUMaxUseThreads = GUIUtil::CPUMaxThreads();
#ifdef ENABLE_GPU
    int nGPUMaxUseThreads = GUIUtil::GPUMaxThreads();
#endif
    std::string PrivAddress = GetArg("-miningprivkey", "");
    if (!PrivAddress.empty()) {
        CDynamicSecret Secret;
        Secret.SetString(PrivAddress);
        if (Secret.IsValid()) {
            CDynamicAddress Address;
            Address.Set(Secret.GetKey().GetPubKey().GetID());
            ui->labelAddress->setText(QString("All mined coins will go to %1").arg(Address.ToString().c_str()));
            hasMiningprivkey = true;
        }
    }

    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->sliderCPUCores->setVisible(false);
        ui->labelNCPUCores->setText(tr("Slider will show once Dynamic has finished syncing"));
    } else {
        ui->sliderCPUCores->setVisible(true);
        ui->labelNCPUCores->setText(QString("%1").arg(nCPUMaxUseThreads));
    }

    ui->sliderCPUCores->setMinimum(0);
    ui->sliderCPUCores->setMaximum(nCPUMaxUseThreads);
    ui->sliderCPUCores->setValue(nCPUMaxUseThreads);

#ifdef ENABLE_GPU
    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->sliderGPUCores->setVisible(false);
        ui->labelNGPUCores->setText(tr("Slider will show once Dynamic has finished syncing"));
    } else {
        ui->sliderGPUCores->setVisible(true);
        ui->labelNGPUCores->setText(QString("%1").arg(nGPUMaxUseThreads));
    }

    ui->sliderGPUCores->setMinimum(0);
    ui->sliderGPUCores->setMaximum(nGPUMaxUseThreads);
    ui->sliderGPUCores->setValue(nGPUMaxUseThreads);
    ui->pushSwitchGPUMining->setVisible(true);
    ui->checkBoxShowGPUGraph->setVisible(true);
#else
    ui->sliderGPUCores->setVisible(false);
    ui->labelNGPUCores->setText(tr("GPU mining is not supported in this version of Dynamic"));
    ui->pushSwitchGPUMining->setVisible(false);
    ui->checkBoxShowGPUGraph->setVisible(false);
#endif

    ui->sliderCPUGraphSampleTime->setMaximum(0);
    ui->sliderCPUGraphSampleTime->setMaximum(6);

#ifdef ENABLE_GPU
    ui->sliderGPUGraphSampleTime->setMaximum(0);
    ui->sliderGPUGraphSampleTime->setMaximum(6);
#else
    ui->sliderGPUGraphSampleTime->setVisible(false);
#endif

#ifdef ENABLE_GPU
    ui->labelGPUGraphSampleSize->setVisible(true);
#else
    ui->labelGPUGraphSampleSize->setVisible(false);
#endif

    ui->sliderCPUCores->setToolTip(tr("Use the slider to select the amount of CPU threads to use"));
#ifdef ENABLE_GPU
    ui->sliderGPUCores->setToolTip(tr("Use the slider to select the amount of GPU devices to use"));
#endif
    ui->labelCPUMinerHashRate->setToolTip(tr("This shows the hashrate of your CPU whilst mining"));
#ifdef ENABLE_GPU
    ui->labelGPUMinerHashRate->setToolTip(tr("This shows the hashrate of your GPU whilst mining"));
#endif
    ui->labelNetHashRateCPU->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelNetHashRateGPU->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelNextCPUBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));
#ifdef ENABLE_GPU
    ui->labelNextGPUBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));
#endif

    connect(ui->sliderCPUCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfCPUThreads(int)));
#ifdef ENABLE_GPU
    connect(ui->sliderGPUCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfGPUThreads(int)));
#endif
    connect(ui->sliderCPUGraphSampleTime, SIGNAL(valueChanged(int)), this, SLOT(changeCPUSampleTime(int)));
#ifdef ENABLE_GPU
    connect(ui->sliderGPUGraphSampleTime, SIGNAL(valueChanged(int)), this, SLOT(changeGPUSampleTime(int)));
#endif
    connect(ui->pushSwitchCPUMining, SIGNAL(clicked()), this, SLOT(switchCPUMining()));
#ifdef ENABLE_GPU
    connect(ui->pushSwitchGPUMining, SIGNAL(clicked()), this, SLOT(switchGPUMining()));
#endif
    connect(ui->pushButtonClearCPUData, SIGNAL(clicked()), this, SLOT(clearCPUHashRateData()));
#ifdef ENABLE_GPU
    connect(ui->pushButtonClearGPUData, SIGNAL(clicked()), this, SLOT(clearGPUHashRateData()));
#else
    ui->pushButtonClearGPUData->setVisible(false);
#endif
    connect(ui->checkBoxShowCPUGraph, SIGNAL(stateChanged(int)), this, SLOT(showCPUHashRate(int)));
#ifdef ENABLE_GPU
    connect(ui->checkBoxShowGPUGraph, SIGNAL(stateChanged(int)), this, SLOT(showGPUHashRate(int)));
#endif

    ui->minerCPUHashRateWidget->graphType = HashRateGraphWidget::GraphType::MINER_CPU_HASHRATE;
    ui->minerCPUHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);

#ifdef ENABLE_GPU
    ui->minerGPUHashRateWidget->graphType = HashRateGraphWidget::GraphType::MINER_GPU_HASHRATE;
    ui->minerGPUHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);
#endif

    showCPUHashMeterControls(false);
#ifdef ENABLE_GPU
    showGPUHashMeterControls(false);
#endif
    fCPUMinerOn = false;
    fGPUMinerOn = false;
    updateUI();
    startTimer(3511);
}

MiningPage::~MiningPage()
{
    delete ui;
}

void MiningPage::setModel(WalletModel* model)
{
    this->model = model;
}

void MiningPage::updateUI()
{
    if (dynodeSync.IsSynced() && dynodeSync.IsBlockchainSynced()) {
#ifdef ENABLE_GPU
        if (ui->sliderGPUCores->isHidden()) {
            int nThreads = ui->sliderGPUCores->value();
            ui->sliderGPUCores->setVisible(true);
            ui->labelNGPUCores->setText(QString("%1").arg(nThreads));
        }
#endif
        if (ui->sliderCPUCores->isHidden()) {
            int nThreads = ui->sliderCPUCores->value();
            ui->sliderCPUCores->setVisible(true);
            ui->labelNCPUCores->setText(QString("%1").arg(nThreads));
        }
    }
    qint64 networkHashrate = GUIUtil::GetNetworkHashPS(120, -1);
    qint64 hashrate = GetHashRate();

    ui->labelNetHashRateCPU->setText(GUIUtil::FormatHashRate(networkHashrate));
    ui->labelNetHashRateGPU->setText(GUIUtil::FormatHashRate(networkHashrate));
    ui->labelCPUMinerHashRate->setText(GUIUtil::FormatHashRate(GetCPUHashRate()));
#ifdef ENABLE_GPU
    ui->labelGPUMinerHashRate->setText(GUIUtil::FormatHashRate(GetGPUHashRate()));
#endif

    QString nextBlockTime;
    if (hashrate == 0) {
        nextBlockTime = QChar(L'∞');
    } else {
        arith_uint256 target;
        target.SetCompact(chainActive.Tip()->nBits);
        arith_uint256 expectedTime = (arith_uint256(1) << 256) / (target * hashrate);
        nextBlockTime = GUIUtil::FormatTimeInterval(expectedTime);
    }

    ui->labelNextCPUBlock->setText(nextBlockTime);
#ifdef ENABLE_GPU
    ui->labelNextGPUBlock->setText(nextBlockTime);
    updateGPUPushSwitch();
#endif
    updateCPUPushSwitch();
}

void MiningPage::updatePushSwitch(QPushButton* pushSwitch, bool minerOn)
{
    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        pushSwitch->setToolTip(tr("Blockchain/Dynodes are not synced, please wait until fully synced before mining!"));
        pushSwitch->setText(tr("Disabled"));
        pushSwitch->setEnabled(false);
        return;
    }
    if (minerOn) {
        pushSwitch->setToolTip(tr("Click 'Stop mining' to stop mining!"));
        pushSwitch->setText(tr("Stop mining"));
    } else if (!minerOn) {
        pushSwitch->setToolTip(tr("Click 'Start mining' to begin mining!"));
        pushSwitch->setText(tr("Start mining"));
    }
    pushSwitch->setEnabled(true);
}

void MiningPage::updateCPUPushSwitch()
{
    updatePushSwitch(ui->pushSwitchCPUMining, fCPUMinerOn);
}

#ifdef ENABLE_GPU
void MiningPage::updateGPUPushSwitch()
{
    updatePushSwitch(ui->pushSwitchGPUMining, fGPUMinerOn);
}
#endif

void MiningPage::StartCPUMiner()
{
    LogPrintf("StartCPUMiner %d (%s)", ui->sliderCPUCores->value(), fCPUMinerOn);
    fCPUMinerOn = true;
    InitMiners(Params(), *g_connman);
    changeNumberOfCPUThreads(ui->sliderCPUCores->value());
    updateUI();
}

#ifdef ENABLE_GPU
void MiningPage::StartGPUMiner()
{
    fGPUMinerOn = true;
    InitMiners(Params(), *g_connman);
    changeNumberOfGPUThreads(ui->sliderGPUCores->value());
    updateUI();
}
#endif

void MiningPage::StopCPUMiner()
{
    LogPrintf("StopCPUMiner %d (%s)", ui->sliderCPUCores->value(), fCPUMinerOn);
    fCPUMinerOn = false;
    changeNumberOfCPUThreads(0, true);
    ShutdownCPUMiners();
    updateUI();
}

#ifdef ENABLE_GPU
void MiningPage::StopGPUMiner()
{
    fGPUMinerOn = false;
    changeNumberOfGPUThreads(0, true);
    ShutdownGPUMiners();
    updateUI();
}
#endif

bool MiningPage::isMinerOn()
{
#ifdef ENABLE_GPU
    return fCPUMinerOn || fGPUMinerOn;
#else
    return fCPUMinerOn;
#endif
}

void MiningPage::changeNumberOfCPUThreads(int i, bool shutdown)
{
    if (!shutdown)
        ui->labelNCPUCores->setText(QString("%1").arg(i));
    ForceSetArg("-gen", isMinerOn() ? "1" : "0");
    ForceSetArg("-genproclimit-cpu", isMinerOn() ? i : 0);
    InitMiners(Params(), *g_connman);
    SetCPUMinerThreads(i);
    if (fCPUMinerOn)
        StartMiners();
}

#ifdef ENABLE_GPU
void MiningPage::changeNumberOfGPUThreads(int i, bool shutdown)
{
    if (!shutdown)
        ui->labelNGPUCores->setText(QString("%1").arg(i));
    ForceSetArg("-gen", isMinerOn() ? "1" : "0");
    ForceSetArg("-genproclimit-gpu", isMinerOn() ? i : 0);
    InitMiners(Params(), *g_connman);
    SetGPUMinerThreads(i);
    if (fGPUMinerOn)
        StartMiners();
}
#endif

void MiningPage::switchCPUMining()
{
    fCPUMinerOn = !fCPUMinerOn;
    updateCPUPushSwitch();
    if (fCPUMinerOn) {
        StartCPUMiner();
    } else {
        StopCPUMiner();
    }
}

#ifdef ENABLE_GPU
void MiningPage::switchGPUMining()
{
    fGPUMinerOn = !fGPUMinerOn;
    updateGPUPushSwitch();
    if (fGPUMinerOn) {
        StartGPUMiner();
    } else {
        StopGPUMiner();
    }
}
#endif

void MiningPage::timerEvent(QTimerEvent*)
{
    updateUI();
}

void MiningPage::showCPUHashRate(int i)
{
    if (i == 0) {
        ui->minerCPUHashRateWidget->StopHashMeter();
        showCPUHashMeterControls(false);
    } else {
        ui->minerCPUHashRateWidget->StartHashMeter();
        showCPUHashMeterControls(true);
    }
}

#ifdef ENABLE_GPU
void MiningPage::showGPUHashRate(int i)
{
    if (i == 0) {
        ui->minerGPUHashRateWidget->StopHashMeter();
        showGPUHashMeterControls(false);
    } else {
        ui->minerGPUHashRateWidget->StartHashMeter();
        showGPUHashMeterControls(true);
    }
}
#endif

void MiningPage::showCPUHashMeterControls(bool show)
{
    ui->sliderCPUGraphSampleTime->setVisible(show);
    ui->labelCPUGraphSampleSize->setVisible(show);
    ui->pushButtonClearCPUData->setVisible(show);
}

#ifdef ENABLE_GPU
void MiningPage::showGPUHashMeterControls(bool show)
{
    ui->sliderGPUGraphSampleTime->setVisible(show);
    ui->labelGPUGraphSampleSize->setVisible(show);
    ui->pushButtonClearGPUData->setVisible(show);
}
#endif

void MiningPage::clearCPUHashRateData()
{
    ui->minerCPUHashRateWidget->clear();
}

#ifdef ENABLE_GPU
void MiningPage::clearGPUHashRateData()
{
    ui->minerGPUHashRateWidget->clear();
}
#endif

void setSampleTimeLabel(QLabel* labelSize, HashRateGraphWidget* hashRate, int i)
{
    switch (i) {
    case 0:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);
        labelSize->setText(QString("5 minutes"));
        break;
    case 1:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::TEN_MINUTES);
        labelSize->setText(QString("10 minutes"));
        break;
    case 2:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::THIRTY_MINUTES);
        labelSize->setText(QString("30 minutes"));
        break;
    case 3:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::ONE_HOUR);
        labelSize->setText(QString("1 hour"));
        break;
    case 4:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::EIGHT_HOURS);
        labelSize->setText(QString("8 hours"));
        break;
    case 5:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::TWELVE_HOURS);
        labelSize->setText(QString("12 hours"));
        break;
    case 6:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::ONE_DAY);
        labelSize->setText(QString("1 day"));
        break;
    default:
        hashRate->UpdateSampleTime(HashRateGraphWidget::SampleTime::ONE_DAY);
        labelSize->setText(QString("1 day"));
        break;
    }
}

void MiningPage::changeCPUSampleTime(int i)
{
    QLabel* labelSize = ui->labelCPUGraphSampleSize;
    HashRateGraphWidget* hashRate = ui->minerCPUHashRateWidget;
    setSampleTimeLabel(labelSize, hashRate, i);
}

#ifdef ENABLE_GPU
void MiningPage::changeGPUSampleTime(int i)
{
    QLabel* labelSize = ui->labelGPUGraphSampleSize;
    HashRateGraphWidget* hashRate = ui->minerGPUHashRateWidget;
    setSampleTimeLabel(labelSize, hashRate, i);
}
#endif
