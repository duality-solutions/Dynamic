// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "mnemonicdialog.h"
#include "ui_mnemonicdialog.h"

#include "rpcserver.h"
#include "rpcclient.h"
#include "wallet/db.h"
#include "wallet/wallet.h"
#include "init.h"
#include "util.h"
#include "guiutil.h"
#include "base58.h"
#include "bip39.h"

#include <QApplication>
#include <QClipboard>
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
    ui->reimportMnemonic->setVisible(false);
    ui->importPrivatekey->setStyleSheet(styleSheet);
    ui->importWallet->setStyleSheet(styleSheet);
    ui->pushButtonCreateMnemonic_Generate->setStyleSheet(styleSheet);
    ui->pushButtonCreateMnemonic_Cancel->setStyleSheet(restyleSheet);
    ui->pushButtonImportMnemonicCancel->setStyleSheet(restyleSheet);
    ui->pushButtonCreateMnemonic_Validate->setStyleSheet(styleSheet);

    ui->pushButtonPrivatekeyCancel->setStyleSheet(restyleSheet);
    ui->pushButtonPrivateKeyFileCancel->setStyleSheet(restyleSheet);

    ui->textBrowser->setText("<p>"+tr("Tips: if the import process is interrupted(such as a power cut or accidental shutdown), please re-enter the recovery phrase or the private key and click the 'Import' button.")+"</p>");



    //initialize Language dropdowns
    std::vector<std::string> languageOptions = {
        "English", 
        "Chinese Simplified",
        "Chinese Traditional",
        "French", 
        //"German",
        "Italian",
        "Japanese",
        "Korean",
        //"Russian",
        "Spanish"
        //"Ukrainian",
        };

    for (auto & element : languageOptions) {
        ui->comboBoxImportMnemonic_Language->addItem(QObject::tr(element.c_str()));
        ui->comboBoxLanguage->addItem(QObject::tr(element.c_str()));

    }


    connect(ui->comboBoxImportMnemonic_Language, SIGNAL(currentIndexChanged(int)), this, SLOT(combobox1ItemChanged(int)));
    connect(ui->comboBoxLanguage, SIGNAL(currentIndexChanged(int)), this, SLOT(combobox2ItemChanged(int)));





}

MnemonicDialog::~MnemonicDialog()
{
    QSettings settings;
    settings.setValue("MnemonicDialogGeometry", saveGeometry());
    delete ui;
}


void MnemonicDialog::combobox1ItemChanged(int input)
{
    ui->comboBoxLanguage->setCurrentIndex(input);

}; //combobox1ItemChanged

void MnemonicDialog::combobox2ItemChanged(int input)
{
    ui->comboBoxImportMnemonic_Language->setCurrentIndex(input);

}; //combobox1ItemChanged

void MnemonicDialog::on_importPrivatekey_clicked()
{
    bool ForceRescan = true;
    
    importPrivatekey(ForceRescan);
}

void MnemonicDialog::on_importMnemonic_clicked()
{
    importMnemonic(false);
}

void MnemonicDialog::on_pushButtonImportMnemonicCancel_clicked()
{
    QDialog::reject(); //cancelled
}


void MnemonicDialog::on_pushButtonCreateMnemonic_Cancel_clicked()
{
    QDialog::reject(); //cancelled
}


void MnemonicDialog::on_pushButtonPrivatekeyCancel_clicked()
{
    QDialog::reject(); //cancelled
}

void MnemonicDialog::on_pushButtonPrivateKeyFileCancel_clicked()
{
    QDialog::reject(); //cancelled
}


void MnemonicDialog::on_toolButtonCreateMnemonic_Copy_clicked()
{
    GUIUtil::setClipboard(ui->textEditNewRecoveryPhrase->toPlainText());
}


void MnemonicDialog::on_toolButtonImportMnemonic_Paste_clicked()
{
    ui->mnemonicEdit->setText(QApplication::clipboard()->text());
}


void MnemonicDialog::on_toolButtonImportMnemonic_Clear_clicked()
{
    ui->mnemonicEdit->setText("");  
}



void MnemonicDialog::on_toolButtonCreateMnemonic_Clear_clicked()
{
    ui->textEditNewRecoveryPhrase->setText("");  
}


void MnemonicDialog::on_pushButtonCreateMnemonic_Validate_clicked()
{
    validateMnemonic();
}

void MnemonicDialog::on_pushButtonCreateMnemonic_Generate_clicked()
{
    createMnemonic();
}

void MnemonicDialog::on_importWallet_clicked()
{
    bool ForceRescan = ui->checkBoxPrivateKeyFileForceRescan->isChecked();

    importWallet(ForceRescan);
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


void MnemonicDialog::createMnemonic() {
    ui->textEditNewRecoveryPhrase->clear();
    
    int value = std::stoi(ui->comboBoxBytesOfEntropy->currentText().toStdString());
    QString languageValue = ui->comboBoxLanguage->currentText();
    languageValue.replace(QString(" "),QString(""));
    languageValue = languageValue.toLower();

    CMnemonic::Language selectLanguage = CMnemonic::Language::ENGLISH; //initialize default

    selectLanguage = CMnemonic::getLanguageEnumFromLabel(languageValue.toStdString());

    SecureString recoveryPhrase = CMnemonic::Generate(value,selectLanguage);

    
    ui->textEditNewRecoveryPhrase->setText(recoveryPhrase.c_str());


} //createMnemonic


void MnemonicDialog::validateMnemonic() {

    bool isValid = false;

    std::string mnemonicValue = ui->textEditNewRecoveryPhrase->toPlainText().toStdString();
    QString languageValue = ui->comboBoxLanguage->currentText();
    languageValue.replace(QString(" "),QString(""));
    languageValue = languageValue.toLower();

    CMnemonic::Language selectLanguage = CMnemonic::Language::ENGLISH; //initialize default

    selectLanguage = CMnemonic::getLanguageEnumFromLabel(languageValue.toStdString());


    isValid = CMnemonic::Check(mnemonicValue.c_str(),selectLanguage);

    if (isValid) {
        QMessageBox::information(this, "Validate Mnemonic", "Mnemonic is valid");
    } else {
        QMessageBox::critical(this, "Validate Mnemonic", "Mnemonic is invalid");
    }


} //createMnemonic

void MnemonicDialog::importMnemonic(bool forceRescan){
    QString mnemonicstr = ui->mnemonicEdit->toPlainText();
    QString RPCstr = (QString("importmnemonic "));
    QString languageValue = ui->comboBoxImportMnemonic_Language->currentText();
    QString passPhrase = ui->lineEditPassPhrase->text();
    languageValue.replace(QString(" "),QString(""));
    languageValue = languageValue.toLower();

    std::string outputmessage = "";
    CMnemonic::Language selectLanguage = CMnemonic::Language::ENGLISH; //initialize default
    bool isValid = false;

   selectLanguage = CMnemonic::getLanguageEnumFromLabel(languageValue.toStdString());

    isValid = CMnemonic::Check(mnemonicstr.toStdString().c_str(),selectLanguage);

    if(mnemonicstr.isEmpty())
    {
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("mnemonics is null"));
        return;
    }

    if  (mnemonicstr.count(QRegExp(" ")) < 11 || mnemonicstr.count(QRegExp(" ")) >= 24 || !isValid){
        QMessageBox::critical(this, "Error", QString("Error: ") + QString::fromStdString("input correct mnemonics"));
        return;
    }



    mnemonicstr.replace(QString(" "),QString("-"));
    RPCstr.append(mnemonicstr);

    languageValue.prepend(QString(" "));
    RPCstr.append(languageValue);
    if (passPhrase.length() > 0) RPCstr.append(QString(" \"") + passPhrase + QString("\""));

    try {
        Q_EMIT cmdToConsole(RPCstr);
    } catch (const std::exception& e) {
        outputmessage = e.what();
        QMessageBox::critical(0, "Import Mnemonic Error", QObject::tr(outputmessage.c_str()));
        return;
    }

    ui->mnemonicEdit->clear();
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