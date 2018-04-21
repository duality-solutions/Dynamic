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
            ui->labelAddress->setText(QString("All mined coins will go to to %1").arg(Address.ToString().c_str()));
            hasMiningprivkey = true;
        }
    }

    ui->sliderCores->setMinimum(0);
    ui->sliderCores->setMaximum(nMaxUseThreads);
    ui->sliderCores->setValue(nMaxUseThreads);
    ui->labelNCores->setText(QString("%1").arg(nMaxUseThreads));

    ui->sliderCores->setToolTip(tr("Use the slider to select the amount of CPU threads to use"));
    ui->labelNethashrate->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelYourHashrate->setToolTip(tr("This shows the hashrate of your CPU whilst mining"));
    ui->labelNextBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));

    connect(ui->sliderCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfCores(int)));
    connect(ui->pushSwitchMining, SIGNAL(clicked()), this, SLOT(switchMining()));
    
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

    ui->labelNethashrate->setText(GUIUtil::FormatHashRate(NetworkHashrate));
    ui->labelYourHashrate->setText(GUIUtil::FormatHashRate(Hashrate));
    
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
        ui->pushSwitchMining->setToolTip(tr("Click 'Start' to begin mining!"));
        ui->pushSwitchMining->setText(tr("Start mining"));
        ui->pushSwitchMining->setEnabled(true);
        ui->ShowHashRateWidget->updateHashRateGraph();
     }
     else {
        ui->pushSwitchMining->setToolTip(tr("Click 'Stop' to finish mining!"));
        ui->pushSwitchMining->setText(tr("Stop mining"));
        ui->pushSwitchMining->setEnabled(true);
        ui->ShowHashRateWidget->updateHashRateGraph();
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