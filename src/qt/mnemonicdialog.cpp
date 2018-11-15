// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "mnemonicdialog.h"
#include "ui_mnemonicdialog.h"

#include "rpcserver.h"
#include "rpcclient.h"
#include "wallet/mnemonic/mnemonic.h"
#include "wallet/db.h"
#include "wallet/wallet.h"
#include "init.h"
#include "util.h"
#include "guiutil.h"
#include "base58.h"

#include <QApplication>
#include <QMessageBox>
#include <QKeyEvent>
#include <QLineEdit>
#include <QPushButton>
#include <QListView>
#include <QFileDialog>
#include <QColor>
#include <QPalette>
#include <QSettings>
#include <QDesktopWidget>

#include <fstream>

MnemonicDialog::MnemonicDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MnemonicDialog)
{
    ui->setupUi(this);
//    QSettings settings;
//    if (!restoreGeometry(settings.value("MnemonicDialogGeometry").toByteArray())) {
//        // Restore failed (perhaps missing setting), center the window
//        move(QApplication::desktop()->availableGeometry().center() - frameGeometry().center());
//    }
    QString restyleSheet = "QPushButton{background-color:rgb(255,149,0);color: black;   border-radius: 2px;border-style: outset;border: 1px groove gray}""QPushButton:pressed{background-color:rgb(112, 170, 245);border-style: inset;border: 1px groove gray }";
    QString styleSheet = "QPushButton{background-color:rgb(27,173,248);color: black;   border-radius: 2px;border-style: outset;border: 1px groove gray}""QPushButton:pressed{background-color:rgb(112, 170, 245);border-style: inset; border: 1px groove gray}";

    ui->importMnemonic->setStyleSheet(styleSheet);
    ui->reimportMnemonic->setStyleSheet(restyleSheet);
    ui->importPrivatekey->setStyleSheet(styleSheet);
    ui->reimportPrivatekey->setStyleSheet(restyleSheet);
    ui->importWallet->setStyleSheet(styleSheet);
    ui->reimportWallet->setStyleSheet(restyleSheet);

    ui->textBrowser->setText("<p>"+tr("Tips: if the import process is interrupted(such as a power cut or accidental shutdown), please re-enter the recovery phrase or the private key and click the 'Reimport' button.")+"</p>");
}

MnemonicDialog::~MnemonicDialog()
{
    QSettings settings;
    settings.setValue("MnemonicDialogGeometry", saveGeometry());
    delete ui;
}


void MnemonicDialog::on_importPrivatekey_clicked()
{
    importPrivatekey(false);
}

void MnemonicDialog::on_importMnemonic_clicked()
{
    importMnemonic(false);
}
void MnemonicDialog::on_importWallet_clicked()
{
    importWallet(false);
}

void MnemonicDialog::on_fileButton_clicked()
{
    QString dataDir = GUIUtil::boostPathToQString(GetDataDir(false));
    QString dir = QDir::toNativeSeparators(QFileDialog::getOpenFileName(this, tr("Choose File"),dataDir));
    if(!dir.isEmpty())
        ui->fileEdit->setText(dir);        
}

void MnemonicDialog::on_reimportMnemonic_clicked()
{
    importMnemonic(true);
}

void MnemonicDialog::on_reimportWallet_clicked()
{
    importWallet(true);
}

void MnemonicDialog::on_reimportPrivatekey_clicked()
{
    importPrivatekey(true);
}

void MnemonicDialog::importMnemonic(bool forceRescan){
    QString mnemonicstr = ui->mnemonicEdit->toPlainText();
    mnemonicstr.replace(QString(" "),QString("-"));
    mnemonicstr.insert(0,QString("importmnemonic "));
    
    if(forceRescan)
        mnemonicstr.append(QString(" 0 100 true"));
    
    if(mnemonicstr.isEmpty())
    {
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("mnemonics is null"));
        return;
    }
    if(mnemonicstr.count(QRegExp("-")) < 11 || mnemonicstr.count(QRegExp("-")) >= 24){
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("input correct mnemonics"));
        return;
    }
    ui->mnemonicEdit->clear();
    Q_EMIT cmdToConsole(mnemonicstr);
}

void MnemonicDialog::importWallet(bool forceRescan){
    QString filepath = ui->fileEdit->text();
    ui->fileEdit->clear();
    if(filepath.isEmpty())
    {
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("file path is null"));
        return;
    }
    std::ifstream file;
    filepath.replace("\\", "/");
    file.open(filepath.toStdString(), std::ios::in | std::ios::ate);
    if (!file.is_open()){
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("Cannot open wallet dump file"));
        return;
    }
    file.close();
    filepath.insert(0,QString("importwallet "));
    if(forceRescan)
        filepath.append(QString(" true"));
        
    Q_EMIT cmdToConsole(filepath);
}

void MnemonicDialog::importPrivatekey(bool forceRescan){
    QString privatekeystr = ui->privatekeyEdit->text();
    ui->privatekeyEdit->clear();
    if(privatekeystr.isEmpty())
    {
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("privatekey is null"));
        return;
    }
    CDynamicSecret vchSecret;
    bool fGood = vchSecret.SetString(privatekeystr.toStdString());

    if (!fGood) {
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("Invalid private key encoding"));
        return;
    }
    privatekeystr.insert(0,QString("importprivkey "));
    if(forceRescan)
        privatekeystr.append(QString(" \"\" true"));
    
    Q_EMIT cmdToConsole(privatekeystr);
}