#include "miningpage.h"
#include "ui_miningpage.h"

#include "dynode-sync.h"
#include "miner.h"
#include "net.h"
#include "util.h"
#include "utiltime.h"
#include "validation.h"
#include "walletmodel.h"

#include <boost/thread.hpp>
#include <stdio.h>


static int MaxThreads() {
    int nThreads = boost::thread::hardware_concurrency();

    int nUseThreads = GetArg("-genproclimit", -1);
    if (nUseThreads < 0)
        nUseThreads = nThreads;

        return nUseThreads;
}

static int64_t GetHashRate() {

    if (GetTimeMillis() - nHPSTimerStart > 8000)
        return (int64_t)0;
    return (int64_t)dHashesPerSec;
}

static QString formatHashrate(qint64 n)
{
    if (n == 0)
        return "0 H/s";

    int i = (int)floor(log(n)/log(1000));
    float v = n*pow(1000.0f, -i);

    QString prefix = "";
    if (i >= 1 && i < 9)
        prefix = " kMGTPEZY"[i];

    return QString("%1 %2H/s").arg(v, 0, 'f', 2).arg(prefix);
}

static int64_t GetNetworkHashPS(int lookup, int height) {
    CBlockIndex *pb = chainActive.Tip();

    if (height >= 0 && height < chainActive.Height())
        pb = chainActive[height];

    if (pb == NULL || !pb->nHeight)
        return 0;

    // If lookup is -1, then use blocks since last difficulty change.
    if (lookup <= 0)
        lookup = pb->nHeight % Params().GetConsensus().DifficultyAdjustmentInterval() + 1;

    // If lookup is larger than chain, then set it to chain length.
    if (lookup > pb->nHeight)
        lookup = pb->nHeight;

    CBlockIndex *pb0 = pb;
    int64_t minTime = pb0->GetBlockTime();
    int64_t maxTime = minTime;
    for (int i = 0; i < lookup; i++) {
        pb0 = pb0->pprev;
        int64_t time = pb0->GetBlockTime();
        minTime = std::min(time, minTime);
        maxTime = std::max(time, maxTime);
    }

    // In case there's a situation where minTime == maxTime, we don't want a divide by zero exception.
    if (minTime == maxTime)
        return 0;

    arith_uint256 workDiff = pb->nChainWork - pb0->nChainWork;
    int64_t timeDiff = maxTime - minTime;

    return workDiff.getdouble() / timeDiff;
}

static QString formatTimeInterval(arith_uint256 time)
{
    enum  EUnit { YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, NUM_UNITS };

    const int SecondsPerUnit[NUM_UNITS] =
    {
        31556952, // average number of seconds in gregorian year
        31556952/12, // average number of seconds in gregorian month
        24*60*60, // number of seconds in a day
        60*60, // number of seconds in an hour
        60, // number of seconds in a minute
        1
    };

    const char* UnitNames[NUM_UNITS] =
    {
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second"
    };

    if (time > 0xFFFFFFFF)
    {
        time /= SecondsPerUnit[YEAR];
        return QString("%1 years").arg(time.ToString().c_str());
    }
    else
    {
        unsigned int t32 = (unsigned int)time.GetCompact();

        int Values[NUM_UNITS];
        for (int i = 0; i < NUM_UNITS; i++)
        {
            Values[i] = t32/SecondsPerUnit[i];
            t32 %= SecondsPerUnit[i];
        }

        int FirstNonZero = 0;
        while (FirstNonZero < NUM_UNITS && Values[FirstNonZero] == 0)
            FirstNonZero++;

        QString TimeStr;
        for (int i = FirstNonZero; i < std::min(FirstNonZero + 3, (int)NUM_UNITS); i++)
        {
            int Value = Values[i];
            TimeStr += QString("%1 %2%3 ").arg(Value).arg(UnitNames[i]).arg((Value == 1)? "" : "s"); // FIXME: this is English specific
        }
        return TimeStr;
    }
}

MiningPage::MiningPage(const PlatformStyle *platformStyle, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MiningPage),
    hasMiningprivkey(false)
{
    ui->setupUi(this);

    int nMaxUseThreads = MaxThreads();

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
    ui->sliderCores->setToolTip(tr("Use the slider to select the amount of CPU threads to use"));

    ui->labelNCores->setText(QString("%1").arg(nMaxUseThreads));

    connect(ui->sliderCores, SIGNAL(valueChanged(int)), this, SLOT(changeNumberOfCores(int)));
    connect(ui->pushSwitchMining, SIGNAL(clicked()), this, SLOT(switchMining()));

    updateUI();
    startTimer(487);
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
    qint64 NetworkHashrate = GetNetworkHashPS(120, -1);
    qint64 Hashrate = GetHashRate();

    ui->labelNethashrate->setText(formatHashrate(NetworkHashrate));
    ui->labelNethashrate->setToolTip(tr("This shows the overall hashrate of the Dynamic network"));
    ui->labelYourHashrate->setText(formatHashrate(Hashrate));
    ui->labelYourHashrate->setToolTip(tr("This shows the hashrate of your CPU whilst mining"));

    QString NextBlockTime;
    if (Hashrate == 0)
        NextBlockTime = QChar(L'∞');
    else
    {
        arith_uint256 Target;
        Target.SetCompact(chainActive.Tip()->nBits);
        arith_uint256 ExpectedTime = (arith_uint256(1) << 256)/(Target*Hashrate);
        NextBlockTime = formatTimeInterval(ExpectedTime);
    }
    
    ui->labelNextBlock->setText(NextBlockTime);
    ui->labelNextBlock->setToolTip(tr("This shows the average time between the blocks you have mined"));

    if (!dynodeSync.IsSynced() || !dynodeSync.IsBlockchainSynced()) {
        ui->pushSwitchMining->setToolTip(tr("Blockchain/Dynodes are not synced, please wait until fully synced before mining!"));
        ui->pushSwitchMining->setText(tr("Disabled"));
        ui->pushSwitchMining->setEnabled(false);
    } 
    else if (dynodeSync.IsSynced() && dynodeSync.IsBlockchainSynced() && GetHashRate() == 0) {
        ui->pushSwitchMining->setToolTip(tr("Click 'Start' to begin mining!"));
        ui->pushSwitchMining->setText(tr("Start mining"));
        ui->pushSwitchMining->setEnabled(true);
    }
    else {
        ui->pushSwitchMining->setToolTip(tr("Click 'Stop' to finish mining!"));
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
    else if (i > 0 && GetHashRate() > 0) {  
        StartMiner();
    }
}

void MiningPage::switchMining()
{
    int nThreads = (int)ui->sliderCores->value();
    if (GetHashRate() > 0) {
        ui->pushSwitchMining->setText(tr("Stopping"));
        StopMiner();
    }
    else if (nThreads == 0 && GetHashRate() == 0){
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
