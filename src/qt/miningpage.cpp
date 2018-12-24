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
        ui->labelNCPUCores->setText(QString("Slider will show once Dynamic has finished syncing"));
    } else {
        ui->sliderCPUCores->setVisible(true);
        ui->labelNCPUCores->setText(QString("%1").arg(nCPUMaxUseThreads));
    }

    ui->sliderCPUCores->setMinimum(1);
    ui->sliderCPUCores->setMaximum(nCPUMaxUseThreads);
    ui->sliderCPUCores->setValue(nCPUMaxUseThreads);

#ifdef ENABLE_GPU
    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->sliderGPUCores->setVisible(false);
        ui->labelNGPUCores->setText(QString("Slider will show once Dynamic has finished syncing"));
    } else {
        ui->sliderGPUCores->setVisible(true);
        ui->labelNGPUCores->setText(QString("%1").arg(nGPUMaxUseThreads));
    }

    ui->sliderGPUCores->setMinimum(1);
    ui->sliderGPUCores->setMaximum(nGPUMaxUseThreads);
    ui->sliderGPUCores->setValue(nGPUMaxUseThreads);
    ui->pushSwitchGPUMining->setVisible(true);
    ui->checkBoxShowGPUGraph->setVisible(true);
#else
    ui->sliderGPUCores->setVisible(false);
    ui->labelNGPUCores->setText(QString("GPU mining is not supported in this version of Dynamic"));
    ui->pushSwitchGPUMining->setVisible(false);
    ui->checkBoxShowGPUGraph->setVisible(false);
#endif

    ui->sliderCPUGraphSampleTime->setMaximum(0);
    ui->sliderCPUGraphSampleTime->setMaximum(6);

#ifdef ENABLE_GPU
    ui->sliderGPUGraphSampleTime->setMaximum(0);
    ui->sliderGPUGraphSampleTime->setMaximum(6);
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

    showHashMeterControls(false, false);
    showHashMeterControls(false, true);
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
#endif

    updatePushSwitch(true);
    updatePushSwitch(false);
}

void MiningPage::updatePushSwitch(bool fGPU)
{
    QPushButton* pushSwitch = fGPU ? ui->pushSwitchGPUMining : ui->pushSwitchCPUMining;
    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        pushSwitch->setToolTip(tr("Blockchain/Dynodes are not synced, please wait until fully synced before mining!"));
        pushSwitch->setText(tr("Disabled"));
        pushSwitch->setEnabled(false);
        return;
    }

    if (fGPU && fGPUMinerOn) {
        pushSwitch->setToolTip(tr("Click 'Stop mining' to stop mining!"));
        pushSwitch->setText(tr("Stop mining"));
    } else if (fGPU && !fGPUMinerOn) {
        pushSwitch->setToolTip(tr("Click 'Start mining' to begin mining!"));
        pushSwitch->setText(tr("Start mining"));
    } else if (!fGPU && fCPUMinerOn) {
        pushSwitch->setToolTip(tr("Click 'Stop mining' to stop mining!"));
        pushSwitch->setText(tr("Stop mining"));
    } else if (!fGPU && !fCPUMinerOn) {
        pushSwitch->setToolTip(tr("Click 'Start mining' to begin mining!"));
        pushSwitch->setText(tr("Start mining"));
    }
    pushSwitch->setEnabled(true);
}

void MiningPage::StartMiner(bool fGPU)
{
#ifdef ENABLE_GPU
    if (fGPU) {
        fGPUMinerOn = true;
    } else {
        fCPUMinerOn = true;
    }
    startMining();
#else
    fCPUMinerOn = true;
    startMining();
#endif
}

void MiningPage::StopMiner(bool fGPU)
{
#ifdef ENABLE_GPU
    if (fGPU) {
        fGPUMinerOn = false;
        ShutdownGPUMiners();
    } else {
        fCPUMinerOn = false;
        ShutdownCPUMiners();
    }
    updateUI();
#else
    fCPUMinerOn = false;
    ShutdownCPUMiners();
    updateUI();
#endif
}

void MiningPage::changeNumberOfCPUThreads(int i)
{
    fCPUMinerOn = (i > 0);
    ui->labelNCPUCores->setText(QString("%1").arg(i));
}

#ifdef ENABLE_GPU
void MiningPage::changeNumberOfGPUThreads(int i)
{
    fGPUMinerOn = (i > 0);
    ui->labelNGPUCores->setText(QString("%1").arg(i));
}
#endif

void MiningPage::startMining()
{
#ifdef ENABLE_GPU
    if (fGPUMinerOn) {
        SetGPUMinerThreads(ui->sliderGPUCores->value());
        StartGPUMiners();
    }
#endif
    if (fCPUMinerOn) {
        SetCPUMinerThreads(ui->sliderCPUCores->value());
        StartCPUMiners();
    }
    updateUI();
}

void MiningPage::switchCPUMining()
{
    switchMining(false);
}

#ifdef ENABLE_GPU
void MiningPage::switchGPUMining()
{
    switchMining(true);
}
#endif

void MiningPage::switchMining(bool fGPU)
{
#ifdef ENABLE_GPU
    QPushButton* pushSwitch = fGPU ? ui->pushSwitchGPUMining : ui->pushSwitchCPUMining;
    QSlider* coreSlider = fGPU ? ui->sliderGPUCores : ui->sliderCPUCores;
    QLabel* labelCores = fGPU ? ui->labelNGPUCores : ui->labelNCPUCores;
    int nThreads = (int)(fGPU ? ui->sliderGPUCores->value() : ui->sliderCPUCores->value());
#else
    int64_t hashRate = GetCPUHashRate();
    int nThreads = (int)ui->sliderCPUCores->value();
#endif

#ifdef ENABLE_GPU
    if (fGPU && !fGPUMinerOn) {
        if (nThreads == 0)
            coreSlider->setValue(1);
        coreSlider->setVisible(true);
        labelCores->setText(QString("%1").arg(nThreads));
        pushSwitch->setText(tr("Starting"));
        StartMiner(fGPU);
    } else if (fGPU && fGPUMinerOn) {
        pushSwitch->setText(tr("Stopping"));
        coreSlider->setVisible(false);
        StopMiner(fGPU);
    } else if (!fGPU && !fCPUMinerOn) {
        if (nThreads == 0)
            coreSlider->setValue(1);
        coreSlider->setVisible(true);
        labelCores->setText(QString("%1").arg(nThreads));
        pushSwitch->setText(tr("Starting"));
        StartMiner(fGPU);
    } else if (!fGPU && fCPUMinerOn) {
        pushSwitch->setText(tr("Stopping"));
        coreSlider->setVisible(false);
        StopMiner(fGPU);
    }
#else
    if (hashRate > 0) {
        ui->pushSwitchCPUMining->setText(tr("Stopping"));
        StopMiner(fGPU);
    }
    else if (nThreads == 0 && hashRate == 0){
        ui->sliderCPUCores->setValue(1);
        ui->pushSwitchCPUMining->setText(tr("Starting"));
        StartMiner(fGPU);
    }
    else {
        ui->pushSwitchCPUMining->setText(tr("Starting"));
        StartMiner(fGPU);
    }
#endif
}

void MiningPage::timerEvent(QTimerEvent*)
{
    updateUI();
}

void MiningPage::showCPUHashRate(int i)
{
    showHashRate(i, false);
}

#ifdef ENABLE_GPU
void MiningPage::showGPUHashRate(int i)
{
    showHashRate(i, true);
}
#endif

void MiningPage::showHashRate(int i, bool fGPU)
{
    HashRateGraphWidget* minerHashRateWidget = fGPU ? ui->minerGPUHashRateWidget : ui->minerCPUHashRateWidget;
    if (i == 0) {
        minerHashRateWidget->StopHashMeter();
        showHashMeterControls(false, fGPU);
    } else {
        minerHashRateWidget->StartHashMeter();
        showHashMeterControls(true, fGPU);
    }
}

void MiningPage::showHashMeterControls(bool show, bool fGPU)
{
    if (fGPU) {
        ui->sliderGPUGraphSampleTime->setVisible(show);
        ui->labelGPUGraphSampleSize->setVisible(show);
        ui->pushButtonClearGPUData->setVisible(show);
    } else {
        ui->sliderCPUGraphSampleTime->setVisible(show);
        ui->labelCPUGraphSampleSize->setVisible(show);
        ui->pushButtonClearCPUData->setVisible(show);
    }
}

void MiningPage::changeCPUSampleTime(int i)
{
    changeSampleTime(i, false);
}

#ifdef ENABLE_GPU
void MiningPage::changeGPUSampleTime(int i)
{
    changeSampleTime(i, true);
}
#endif

void MiningPage::changeSampleTime(int i, bool fGPU)
{
#ifdef ENABLE_GPU
    QLabel* labelSize = fGPU ? ui->labelGPUGraphSampleSize : ui->labelCPUGraphSampleSize;
    HashRateGraphWidget* hashRate = fGPU ? ui->minerGPUHashRateWidget : ui->minerCPUHashRateWidget;
#else
    QLabel* labelSize = ui->labelCPUGraphSampleSize;
    HashRateGraphWidget* hashRate = ui->minerCPUHashRateWidget;
#endif
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
