// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "multisigaddressentry.h"
#include "ui_multisigaddressentry.h"

#include "addressbookpage.h"
#include "addresstablemodel.h"
#include "platformstyle.h"
#include "guiutil.h"
#include "walletmodel.h"

#include "base58.h"
#include "key.h"
#include "utilstrencodings.h"

#include <string>
#include <vector>

#include <QApplication>
#include <QClipboard>

MultisigAddressEntry::MultisigAddressEntry(const PlatformStyle *platformStyle, QWidget *parent) : QFrame(parent), ui(new Ui::MultisigAddressEntry), model(0)
{
    ui->setupUi(this);
    GUIUtil::setupAddressWidget(ui->walletaddress, this);
}

MultisigAddressEntry::~MultisigAddressEntry()
{
    delete ui;
}

void MultisigAddressEntry::setModel(WalletModel *model)
{
    this->model = model;
    clear();
}

void MultisigAddressEntry::clear()
{
    ui->walletaddress->clear();
    ui->label->clear();
    ui->walletaddress->setFocus();
}

bool MultisigAddressEntry::validate()
{
    return !ui->walletaddress->text().isEmpty();
}

QString MultisigAddressEntry::getWalletAddress()
{
    return ui->walletaddress->text();
}

void MultisigAddressEntry::setRemoveEnabled(bool enabled)
{
    ui->deleteButton->setEnabled(enabled);
}

void MultisigAddressEntry::on_pasteButton_clicked()
{
    ui->walletaddress->setText(QApplication::clipboard()->text());
}

void MultisigAddressEntry::on_deleteButton_clicked()
{
    Q_EMIT removeEntry(this);
}

void MultisigAddressEntry::on_addressBookButton_clicked()
{
    if(!model)
        return;
            
    AddressBookPage dlg(platformStyle, AddressBookPage::ForSelection, AddressBookPage::SendingTab, this);
    dlg.setModel(model->getAddressTableModel());
    if(dlg.exec())
    {
        ui->walletaddress->setText(dlg.getReturnValue());
    }
}

void MultisigAddressEntry::on_walletaddress_textChanged(const QString &address)
{
    updateLabel(address);
}

bool MultisigAddressEntry::updateLabel(const QString &address)
{
    if(!model)
        return false;

    // Fill in label from address book, if address has an associated label
    QString associatedLabel = model->getAddressTableModel()->labelForAddress(address);
    if(!associatedLabel.isEmpty())
    {
        ui->label->setText(associatedLabel);
        return true;
    }

    return false;
}
