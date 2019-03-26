// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef MNEMONICDIALOG_H
#define MNEMONICDIALOG_H

#include <QDialog>

namespace Ui {
class MnemonicDialog;
}

class MnemonicDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MnemonicDialog(QWidget *parent = 0);
    ~MnemonicDialog();
    void importMnemonic(bool forceRescan);
    void importWallet(bool forceRescan);
    void importPrivatekey(bool forceRescan);
    void createMnemonic();
    void validateMnemonic();
    
Q_SIGNALS:
    void cmdToConsole(const QString &command);  

private Q_SLOTS:
    void on_importPrivatekey_clicked();
    //void on_reimportPrivatekey_clicked();
    void on_importMnemonic_clicked();
    void on_pushButtonImportMnemonicCancel_clicked();
    void on_pushButtonCreateMnemonic_Generate_clicked();
    void on_reimportMnemonic_clicked();
    void on_pushButtonCreateMnemonic_Cancel_clicked();
    void on_pushButtonCreateMnemonic_Validate_clicked();
    void on_toolButtonCreateMnemonic_Clear_clicked();
    void on_toolButtonCreateMnemonic_Copy_clicked();
    void on_toolButtonImportMnemonic_Paste_clicked();
    void on_toolButtonImportMnemonic_Clear_clicked();
    void on_importWallet_clicked();
    //void on_reimportWallet_clicked();
    void on_fileButton_clicked();
    void combobox1ItemChanged(int input);
    void combobox2ItemChanged(int input);

    void on_pushButtonPrivatekeyCancel_clicked();
    void on_pushButtonPrivateKeyFileCancel_clicked();


private:
    Ui::MnemonicDialog *ui;
};

#endif // MNEMONICDIALOG_H