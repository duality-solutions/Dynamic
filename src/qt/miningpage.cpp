#include "miningpage.h"
#include "ui_miningpage.h"

#include "dynode-sync.h"
#include "guiutil.h"
#include "miner.h"
#include "net.h"
#include "util.h"
#include "utiltime.h"
#include "validation.h"
#include "walletmodel.h"

#include <boost/thread.hpp>
#include <stdio.h>

MiningPage::MiningPage(const PlatformStyle *platformStyle, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MiningPage),
    hasMiningprivkey(false)
{
    ui->setupUi(this);

    int nCPUMaxUseThreads = GUIUtil::CPUMaxThreads();
    int nGPUMaxUseThreads = GUIUtil::GPUMaxThreads();

    std::string PrivAddress = GetArg("-miningprivkey", "");
    if (!PrivAddress.empty())
    {
        CDynamicSecret Secret;
        Secret.SetString(PrivAddress);
        if (Secret.IsValid())
        {
            CDynamicAddress Address;
            Address.Set(Secret.GetKey().GetPubKey().GetID());
            ui->labelAddress->setText(QString("All mined coins will go to %1").arg(Address.ToString().c_str()));
            hasMiningprivkey = true;
        }
    }

    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->sliderCPUCores->setVisible(false);
        ui->labelNCPUCores->setText(QString("Slider will show once Dynamic has finished syncing").arg(nCPUMaxUseThreads));
    }
    else {
        ui->sliderCPUCores->setVisible(true);
        ui->labelNCPUCores->setText(QString("%1").arg(nCPUMaxUseThreads));
    }

    ui->sliderCPUCores->setMinimum(1);
    ui->sliderCPUCores->setMaximum(nCPUMaxUseThreads);
    ui->sliderCPUCores->setValue(nCPUMaxUseThreads);

    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->sliderGPUCores->setVisible(false);
        ui->labelNGPUCores->setText(QString("Slider will show once Dynamic has finished syncing").arg(nGPUMaxUseThreads));
    }
    else {
        ui->sliderGPUCores->setVisible(true);
        ui->labelNGPUCores->setText(QString("%1").arg(nGPUMaxUseThreads));
    }

    ui->sliderGPUCores->setMinimum(1);
    ui->sliderGPUCores->setMaximum(nGPUMaxUseThreads);
    ui->sliderGPUCores->setValue(nGPUMaxUseThreads);

    ui->sliderCPUGraphSampleTime->setMaximum(0);
    ui->sliderCPUGraphSampleTime->setMaximum(6);

    ui->sliderGPUGraphSampleTime->setMaximum(0);
    ui->sliderGPUGraphSampleTime->setMaximum(6);

    ui->sliderCPUCores->setToolTip(tr("Use the slider to select the amount of CPU threads to use"));
    ui->sliderGPUCores->setToolTip(tr("Use the slider to select the amount of GPU threads to use"));
    ui->labelCPUMinerHashRate->setToolTip(tr("This shows the hashrate of your CPU whilst mining"));
    ui->labelGPUMinerHashRate->setToolTip(tr("This shows the hashrate of your GPU whilst mining"));
    ui->labelNetHashRateCPU->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelNetHashRateGPU->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelNextCPUBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));
    ui->labelNextGPUBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));

    connect(ui->sliderCPUCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfCPUThreads(int)));
    connect(ui->sliderGPUCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfGPUThreads(int)));
    connect(ui->sliderCPUGraphSampleTime, SIGNAL(valueChanged(int)), this, SLOT(changeCPUSampleTime(int)));
    connect(ui->sliderGPUGraphSampleTime, SIGNAL(valueChanged(int)), this, SLOT(changeGPUSampleTime(int)));
    connect(ui->pushSwitchCPUMining, SIGNAL(clicked()), this, SLOT(switchCPUMining()));
    connect(ui->pushSwitchGPUMining, SIGNAL(clicked()), this, SLOT(switchGPUMining()));
    connect(ui->pushButtonClearCPUData, SIGNAL(clicked()), this, SLOT(clearCPUHashRateData()));
    connect(ui->pushButtonClearGPUData, SIGNAL(clicked()), this, SLOT(clearGPUHashRateData()));
    connect(ui->checkBoxShowCPUGraph, SIGNAL(stateChanged(int)), this, SLOT(showCPUHashRate(int)));
    connect(ui->checkBoxShowGPUGraph, SIGNAL(stateChanged(int)), this, SLOT(showGPUHashRate(int)));

    ui->minerCPUHashRateWidget->graphType = HashRateGraphWidget::GraphType::MINER_CPU_HASHRATE;
    ui->minerCPUHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);

    ui->minerGPUHashRateWidget->graphType = HashRateGraphWidget::GraphType::MINER_GPU_HASHRATE;
    ui->minerGPUHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);

    showHashMeterControls(false, false);
    showHashMeterControls(false, true);
    updateUI();
    startTimer(3511);
}

MiningPage::~MiningPage()
{
    delete ui;
}

void MiningPage::setModel(WalletModel *model)
{
    this->model = model;
}

void MiningPage::updateUI()
{
    qint64 networkHashrate = GUIUtil::GetNetworkHashPS(120, -1);
    qint64 hashrate = GetHashRate();

    ui->labelNetHashRateCPU->setText(GUIUtil::FormatHashRate(networkHashrate));
    ui->labelNetHashRateGPU->setText(GUIUtil::FormatHashRate(networkHashrate));
    ui->labelCPUMinerHashRate->setText(GUIUtil::FormatHashRate(GetCPUHashRate()));
    ui->labelGPUMinerHashRate->setText(GUIUtil::FormatHashRate(GetGPUHashRate()));

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
    ui->labelNextGPUBlock->setText(nextBlockTime);

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
    } else if (dynodeSync.IsSynced() && dynodeSync.IsBlockchainSynced() && GetHashRate() == 0) {
        pushSwitch->setToolTip(tr("Click 'Start mining' to begin mining!"));
        pushSwitch->setText(tr("Start mining"));
        pushSwitch->setEnabled(true);
    } else {
        pushSwitch->setToolTip(tr("Click 'Stop mining' to finish mining!"));
        pushSwitch->setText(tr("Stop mining"));
        pushSwitch->setEnabled(true);
    }
}

void MiningPage::StartMiner()
{
    int nCPUThreads = (int)ui->sliderCPUCores->value();
    int nGPUThreads = (int)ui->sliderGPUCores->value();
    GenerateDynamics(nCPUThreads, nGPUThreads, Params(), *g_connman);
    updateUI();
}

void MiningPage::StopMiner(bool fGPU)
{
    if (fGPU) {
        ShutdownGPUMiners();
    } else {
        ShutdownCPUMiners();
    }
    updateUI();
}

void MiningPage::changeNumberOfCPUThreads(int i)
{
    ui->labelNCPUCores->setText(QString("%1").arg(i));
    if (i == 0) {
        StopMiner(false);
    } else if (i > 0 && GetCPUHashRate() <= 0) {
        StartMiner();
    }
}

void MiningPage::changeNumberOfGPUThreads(int i)
{
    ui->labelNGPUCores->setText(QString("%1").arg(i));
    if (i == 0) {
        StopMiner(true);
    } else if (i > 0 && GetGPUHashRate() <= 0) {
        StartMiner();
    }
}

void MiningPage::switchCPUMining()
{
    switchMining(false);
}

void MiningPage::switchGPUMining()
{
    switchMining(true);
}

void MiningPage::switchMining(bool fGPU)
{
    QPushButton* pushSwitch = fGPU ? ui->pushSwitchGPUMining : ui->pushSwitchCPUMining;
    QSlider* coreSlider = fGPU ? ui->sliderGPUCores : ui->sliderCPUCores;
    int nThreads = (int)(fGPU ? ui->sliderGPUCores->value() : ui->sliderCPUCores->value());
    int64_t hashRate = fGPU ? GetGPUHashRate() : GetCPUHashRate();

    if (hashRate > 0) {
        pushSwitch->setText(tr("Stopping"));
        StopMiner(fGPU);
    } else if (nThreads == 0 && hashRate == 0) {
        coreSlider->setValue(1);
        pushSwitch->setText(tr("Starting"));
        StartMiner();
    } else {
        pushSwitch->setText(tr("Starting"));
        StartMiner();
    }
}

void MiningPage::timerEvent(QTimerEvent *)
{
    updateUI();
}

void MiningPage::showCPUHashRate(int i)
{
    showHashRate(i, false);
}

void MiningPage::showGPUHashRate(int i)
{
    showHashRate(i, true);
}

void MiningPage::showHashRate(int i, bool fGPU)
{
    HashRateGraphWidget* widget = fGPU ? ui->minerGPUHashRateWidget : ui->minerCPUHashRateWidget;
    if (i == 0) {
        widget->StopHashMeter();
        showHashMeterControls(false, fGPU);
    } else {
        widget->StartHashMeter();
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

void MiningPage::changeGPUSampleTime(int i)
{
    changeSampleTime(i, true);
}

void MiningPage::changeSampleTime(int i, bool fGPU)
{
    QLabel* labelSize = fGPU ? ui->labelGPUGraphSampleSize : ui->labelCPUGraphSampleSize;
    HashRateGraphWidget* hashRate = fGPU ? ui->minerGPUHashRateWidget : ui->minerCPUHashRateWidget;
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

void MiningPage::clearGPUHashRateData()
{
    ui->minerGPUHashRateWidget->clear();
}
