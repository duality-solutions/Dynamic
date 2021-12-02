// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swapdialog.h"
#include "ui_swapdialog.h"

#include "guiutil.h"
#include "walletmodel.h"

#include <QUrl>
#include <QDialogButtonBox>

SwapDialog::SwapDialog(QWidget* parent) : QDialog(parent), ui(new Ui::SwapDialog)
{
    ui->setupUi(this);
#if QT_VERSION >= 0x040700
    ui->swapAddress->setPlaceholderText("Enter your swap address here");
#endif
    // ok button
    connect(ui->buttonBox, SIGNAL(clicked(QAbstractButton*)), this, SLOT(buttonBoxClicked(QAbstractButton*)));
}

SwapDialog::~SwapDialog()
{
    delete ui;
}

QString SwapDialog::getSwapAddress()
{
    return ui->swapAddress->text();
}

void SwapDialog::accept()
{
    QDialog::accept();
}

void SwapDialog::reject()
{
    QDialog::reject();
}

// ok button
void SwapDialog::buttonBoxClicked(QAbstractButton* button)
{
    if (ui->buttonBox->buttonRole(button) == QDialogButtonBox::AcceptRole)
        done(QDialog::Accepted); // closes the dialog
}