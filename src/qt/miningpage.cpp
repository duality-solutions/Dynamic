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

    int nMaxUseThreads = GUIUtil::MaxThreads();

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

    ui->sliderCores->setMinimum(0);
    ui->sliderCores->setMaximum(nMaxUseThreads);
    ui->sliderCores->setValue(nMaxUseThreads);
    ui->labelNCores->setText(QString("%1").arg(nMaxUseThreads));
    ui->sliderGraphSampleTime->setMaximum(0);
    ui->sliderGraphSampleTime->setMaximum(6);

    ui->sliderCores->setToolTip(tr("Use the slider to select the amount of CPU threads to use"));
    ui->labelNetHashRate->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelMinerHashRate->setToolTip(tr("This shows the hashrate of your CPU whilst mining"));
    ui->labelNextBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));

    connect(ui->sliderCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfCores(int)));
    connect(ui->sliderGraphSampleTime, SIGNAL(valueChanged(int)), this, SLOT(changeSampleTime(int)));
    connect(ui->pushSwitchMining, SIGNAL(clicked()), this, SLOT(switchMining()));
    connect(ui->pushButtonClearData, SIGNAL(clicked()), this, SLOT(clearHashRateData()));
    connect(ui->checkBoxShowGraph, SIGNAL(stateChanged(int)), this, SLOT(showHashRate(int)));
    //
    ui->minerHashRateWidget->graphType = HashRateGraphWidget::GraphType::MINER_HASHRATE;
    ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);
    
    showHashMeterControls(false);
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
    qint64 NetworkHashrate = GUIUtil::GetNetworkHashPS(120, -1);
    qint64 Hashrate = GUIUtil::GetHashRate();

    ui->labelNetHashRate->setText(GUIUtil::FormatHashRate(NetworkHashrate));
    ui->labelMinerHashRate->setText(GUIUtil::FormatHashRate(Hashrate));
    
    QString NextBlockTime;
    if (Hashrate == 0)
        NextBlockTime = QChar(L'∞');
    else
    {
        arith_uint256 Target;
        Target.SetCompact(chainActive.Tip()->nBits);
        arith_uint256 ExpectedTime = (arith_uint256(1) << 256)/(Target*Hashrate);
        NextBlockTime = GUIUtil::FormatTimeInterval(ExpectedTime);
    }
    
    ui->labelNextBlock->setText(NextBlockTime);

    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->pushSwitchMining->setToolTip(tr("Blockchain/Dynodes are not synced, please wait until fully synced before mining!"));
        ui->pushSwitchMining->setText(tr("Disabled"));
        ui->pushSwitchMining->setEnabled(false);
    } 
    else if (dynodeSync.IsSynced() && dynodeSync.IsBlockchainSynced() && GUIUtil::GetHashRate() == 0) {
        ui->pushSwitchMining->setToolTip(tr("Click 'Start mining' to begin mining!"));
        ui->pushSwitchMining->setText(tr("Start mining"));
        ui->pushSwitchMining->setEnabled(true);
     }
     else {
        ui->pushSwitchMining->setToolTip(tr("Click 'Stop mining' to finish mining!"));
        ui->pushSwitchMining->setText(tr("Stop mining"));
        ui->pushSwitchMining->setEnabled(true);
    }
}

void MiningPage::StartMiner()
{
    int nThreads = (int)ui->sliderCores->value();
    GenerateDynamics(true, nThreads, Params(), *g_connman);
    updateUI();
}

void MiningPage::StopMiner()
{
    GenerateDynamics(false, 0, Params(), *g_connman);
    updateUI();
}

void MiningPage::changeNumberOfCores(int i)
{
    ui->labelNCores->setText(QString("%1").arg(i));
    if (i == 0) {
        StopMiner();
    }
    else if (i > 0 && GUIUtil::GetHashRate() > 0) {  
        StartMiner();
    }
}

void MiningPage::switchMining()
{
    int64_t hashRate = GUIUtil::GetHashRate();
    int nThreads = (int)ui->sliderCores->value();
    
    if (hashRate > 0) {
        ui->pushSwitchMining->setText(tr("Stopping"));
        StopMiner();
    }
    else if (nThreads == 0 && hashRate == 0){
        ui->sliderCores->setValue(1);
        ui->pushSwitchMining->setText(tr("Starting"));
        StartMiner();
    }
    else {
        ui->pushSwitchMining->setText(tr("Starting"));
        StartMiner();
    }
}

void MiningPage::timerEvent(QTimerEvent *)
{
    updateUI();
}

void MiningPage::showHashRate(int i)
{
    if (i == 0) {
        ui->minerHashRateWidget->StopHashMeter();
        showHashMeterControls(false);
    }
    else {
        ui->minerHashRateWidget->StartHashMeter();
        showHashMeterControls(true);
    }
}

void MiningPage::showHashMeterControls(bool show)
{
    if (show == false) {
        ui->sliderGraphSampleTime->setVisible(false);
        ui->labelGraphSampleSize->setVisible(false);
        ui->pushButtonClearData->setVisible(false);
    }
    else {
        ui->sliderGraphSampleTime->setVisible(true);
        ui->labelGraphSampleSize->setVisible(true);
        ui->pushButtonClearData->setVisible(true);
    }
}

void MiningPage::changeSampleTime(int i)
{
    if (i == 0) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::FIVE_MINUTES);
        ui->labelGraphSampleSize->setText(QString("5 minutes"));
    }
    else if (i == 1) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::TEN_MINUTES);
        ui->labelGraphSampleSize->setText(QString("10 minutes"));
    }
    else if (i == 2) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::THIRTY_MINUTES);
        ui->labelGraphSampleSize->setText(QString("30 minutes"));
    }
    else if (i == 3) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::ONE_HOUR);
        ui->labelGraphSampleSize->setText(QString("1 hour"));
    }
    else if (i == 4) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::EIGHT_HOURS);
        ui->labelGraphSampleSize->setText(QString("8 hours"));
    }
    else if (i == 5) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::TWELVE_HOURS);
        ui->labelGraphSampleSize->setText(QString("12 hours"));
    }
    else if (i == 6) {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::ONE_DAY);
        ui->labelGraphSampleSize->setText(QString("1 day"));
    }
    else {
        ui->minerHashRateWidget->UpdateSampleTime(HashRateGraphWidget::SampleTime::ONE_DAY);
        ui->labelGraphSampleSize->setText(QString("1 day"));
    }
}

void MiningPage::clearHashRateData()
{
    ui->minerHashRateWidget->clear();
}