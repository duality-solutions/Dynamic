// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "sandstormconfig.h"
#include "ui_sandstormconfig.h"

#include "darksilkunits.h"
#include "sandstorm.h"
#include "guiconstants.h"
#include "optionsmodel.h"
#include "walletmodel.h"

#include <QMessageBox>
#include <QPushButton>
#include <QKeyEvent>
#include <QSettings>

SandstormConfig::SandstormConfig(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SandstormConfig),
    model(0)
{
    ui->setupUi(this);

    connect(ui->buttonBasic, SIGNAL(clicked()), this, SLOT(clickBasic()));
    connect(ui->buttonHigh, SIGNAL(clicked()), this, SLOT(clickHigh()));
    connect(ui->buttonMax, SIGNAL(clicked()), this, SLOT(clickMax()));
}

SandstormConfig::~SandstormConfig()
{
    delete ui;
}

void SandstormConfig::setModel(WalletModel *model)
{
    this->model = model;
}

void SandstormConfig::clickBasic()
{
    configure(true, 1000, 2);

    QString strAmount(DarkSilkUnits::formatWithUnit(
        model->getOptionsModel()->getDisplayUnit(), 1000 * COIN));
    QMessageBox::information(this, tr("PrivateSend Configuration"),
        tr(
            "PrivateSend was successfully set to basic (%1 and 2 rounds). You can change this at any time by opening DarkSilk's configuration screen."
        ).arg(strAmount)
    );

    close();
}

void SandstormConfig::clickHigh()
{
    configure(true, 1000, 8);

    QString strAmount(DarkSilkUnits::formatWithUnit(
        model->getOptionsModel()->getDisplayUnit(), 1000 * COIN));
    QMessageBox::information(this, tr("PrivateSend Configuration"),
        tr(
            "PrivateSend was successfully set to high (%1 and 8 rounds). You can change this at any time by opening DarkSilk's configuration screen."
        ).arg(strAmount)
    );

    close();
}

void SandstormConfig::clickMax()
{
    configure(true, 1000, 16);

    QString strAmount(DarkSilkUnits::formatWithUnit(
        model->getOptionsModel()->getDisplayUnit(), 1000 * COIN));
    QMessageBox::information(this, tr("PrivateSend Configuration"),
        tr(
            "PrivateSend was successfully set to maximum (%1 and 16 rounds). You can change this at any time by opening DarkSilk's configuration screen."
        ).arg(strAmount)
    );

    close();
}

void SandstormConfig::configure(bool enabled, int coins, int rounds) {

    QSettings settings;

    settings.setValue("nPrivateSendRounds", rounds);
    settings.setValue("nPrivateSendAmount", coins);

    nPrivateSendRounds = rounds;
    nPrivateSendAmount = coins;
}
