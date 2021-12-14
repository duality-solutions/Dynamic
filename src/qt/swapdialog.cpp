// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swapdialog.h"
#include "ui_swapdialog.h"

#include "askpassphrasedialog.h"
#include "guiutil.h"
#include "rpc/server.h"
#include "swap/ss58.h"
#include "utilmoneystr.h"
#include "wallet/wallet.h"
#include "walletmodel.h"

#include <univalue.h>

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
#ifdef ENABLE_WALLET
    if (pwalletMain) {
        walletLocked = pwalletMain->IsLocked();
        totalBalance = pwalletMain->GetBalance();
        lockedBalance = pwalletMain->LockedCoinsTotal();
        swapBalance = pwalletMain->SwapBalance();
        immatureBalance = totalBalance - swapBalance - lockedBalance;
        std::string label = "<label>Total Balance: " + FormatMoney(totalBalance) + "</label>";
        QString questionString = tr(label.c_str());
        questionString.append("<br />");
        label = "<label>Immuture Balance: " + FormatMoney(immatureBalance) + "</label>";
        questionString.append(tr(label.c_str()));
        questionString.append("<br />");
        label = "<label>Locked Balance: " + FormatMoney(lockedBalance) + "</label>";
        questionString.append(tr(label.c_str()));
        questionString.append("<br /><br />");
        label = "<label>Swap Balance: " + FormatMoney(swapBalance) + "</label>";
        questionString.append(tr(label.c_str()));
        ui->label_swapInfo->setText(QObject::tr(questionString.toStdString().c_str()));
    } else {
        walletLocked = true;
        totalBalance = 0;
        immatureBalance = 0;
        lockedBalance = 0;
        ui->label_swapInfo->setText(QObject::tr("Unable to access wallet. Please close this swap window."));
    }
#endif // ENABLE_WALLET
}

SwapDialog::~SwapDialog()
{
    delete ui;
}

#ifdef ENABLE_WALLET

void SwapDialog::setWalletModel(WalletModel* _walletModel)
{
    this->walletModel = _walletModel;
}

bool SwapDialog::swapDynamic()
{
    std::string errorMessage = "";
    UniValue result = SwapDynamic(ui->swapAddress->text().toStdString(), true, errorMessage);
    if (errorMessage != "") {
        QMessageBox::critical(0, "Swap Dynamic Error", QObject::tr(errorMessage.c_str()));
        return false;
    }
    return true;
}

#endif // ENABLE_WALLET

QString SwapDialog::getSwapAddress()
{
    return ui->swapAddress->text();
}

void SwapDialog::buttonBoxClicked(QAbstractButton* button)
{
#ifdef ENABLE_WALLET
    if (ui->buttonBox->buttonRole(button) == QDialogButtonBox::AcceptRole) {
        if (swapBalance <= 0) {
            QMessageBox::critical(this, QObject::tr("Invalid Amount"), QObject::tr("Swap amount must be greater than 0"));
            done(QDialog::Rejected); // closes the dialog
            return;
        }
        CSS58 swapAddress(getSwapAddress().toStdString());
        if (swapAddress.strError == "") {
            // show fee, amount of coins swapping and swap address
            QString questionString = tr("Are you sure you want to swap?"); 
            questionString.append("<br />");
            questionString.append("<hr /><span style='color:#aa0000;'>");
            questionString.append("</span> ");
            std::string label = "<label>Swap Address: " + getSwapAddress().toStdString() + "</label>";
            questionString.append(label.c_str());
            questionString.append("<br />");
            label = "<label>Swap Balance: " + FormatMoney(swapBalance) + "</label>";
            questionString.append(tr(label.c_str()));
            QMessageBox::StandardButton retval = 
            QMessageBox::question(this, tr("Confirm swap coins"), questionString, QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);

            if (retval != QMessageBox::Yes) {
                done(QDialog::Rejected); // closes the dialog
            } else {
                if (walletLocked) {
                    AskPassphraseDialog dlg(AskPassphraseDialog::Unlock, this);
                    dlg.setModel(walletModel);
                    dlg.exec();
                    walletLocked = pwalletMain->IsLocked();
                    if (walletLocked) {
                        done(QDialog::Rejected); // closes the dialog
                    } else {
                        if (swapDynamic()) {
                            pwalletMain->Lock();
                            done(QDialog::Accepted); // closes the dialog
                        } else {
                            done(QDialog::Rejected); // closes the dialog
                        }
                    }
                }
                else {
                    if (swapDynamic()) {
                        done(QDialog::Accepted); // closes the dialog
                    } else {
                        done(QDialog::Rejected); // closes the dialog
                    }
                }
            }
        } else { // SS58 error
            QMessageBox::critical(this, QObject::tr("Address validation failed"), QObject::tr(swapAddress.strError.c_str()));
        }
    }
    else {
        done(QDialog::Accepted); // closes the dialog
    }
#elif
    done(QDialog::Rejected); // closes the dialog
#endif // ENABLE_WALLET
}