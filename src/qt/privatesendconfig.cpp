// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "privatesendconfig.h"
#include "ui_privatesendconfig.h"

#include "dynamicunits.h"
#include "guiconstants.h"
#include "optionsmodel.h"
#include "walletmodel.h"

#include "privatesend.h"

#include <QKeyEvent>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>

PrivatesendConfig::PrivatesendConfig(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PrivatesendConfig),
    model(0)
{
    ui->setupUi(this);

    connect(ui->buttonBasic, SIGNAL(clicked()), this, SLOT(clickBasic()));
    connect(ui->buttonHigh, SIGNAL(clicked()), this, SLOT(clickHigh()));
    connect(ui->buttonMax, SIGNAL(clicked()), this, SLOT(clickMax()));
}

PrivatesendConfig::~PrivatesendConfig()
{
    delete ui;
}

void PrivatesendConfig::setModel(WalletModel *model)
{
    this->model = model;
}

void PrivatesendConfig::clickBasic()
{
    configure(true, 1000, 2);

    QString strAmount(DynamicUnits::formatWithUnit(
        model->getOptionsModel()->getDisplayUnit(), 1000 * COIN));
    QMessageBox::information(this, tr("PrivateSend Configuration"),
        tr(
            "PrivateSend was successfully set to basic (%1 and 2 rounds). You can change this at any time by opening Dynamic's configuration screen."
        ).arg(strAmount)
    );

    close();
}

void PrivatesendConfig::clickHigh()
{
    configure(true, 1000, 8);

    QString strAmount(DynamicUnits::formatWithUnit(
        model->getOptionsModel()->getDisplayUnit(), 1000 * COIN));
    QMessageBox::information(this, tr("PrivateSend Configuration"),
        tr(
            "PrivateSend was successfully set to high (%1 and 8 rounds). You can change this at any time by opening Dynamic's configuration screen."
        ).arg(strAmount)
    );

    close();
}

void PrivatesendConfig::clickMax()
{
    configure(true, 1000, 16);

    QString strAmount(DynamicUnits::formatWithUnit(
        model->getOptionsModel()->getDisplayUnit(), 1000 * COIN));
    QMessageBox::information(this, tr("PrivateSend Configuration"),
        tr(
            "PrivateSend was successfully set to maximum (%1 and 16 rounds). You can change this at any time by opening Dynamic's configuration screen."
        ).arg(strAmount)
    );

    close();
}

void PrivatesendConfig::configure(bool enabled, int coins, int rounds) {

    QSettings settings;

    settings.setValue("nPrivateSendRounds", rounds);
    settings.setValue("nPrivateSendAmount", coins);

    nPrivateSendRounds = rounds;
    nPrivateSendAmount = coins;
}
