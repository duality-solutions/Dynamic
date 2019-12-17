// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_ASKPASSPHRASEDIALOG_H
#define DYNAMIC_QT_ASKPASSPHRASEDIALOG_H

#include <QDialog>

class WalletModel;

namespace Ui
{
class AskPassphraseDialog;
}

/** Multifunctional dialog to ask for passphrases. Used for encryption, unlocking, and changing the passphrase.
 */
class AskPassphraseDialog : public QDialog
{
    Q_OBJECT

public:
    enum class Mode {
        Encrypt,      /**< Ask passphrase twice and encrypt */
        UnlockMixing, /**< Ask passphrase and unlock only for mixing */
        Unlock,       /**< Ask passphrase and unlock */
        ChangePass,   /**< Ask old passphrase + new passphrase twice */
        Decrypt       /**< Ask passphrase and decrypt wallet */
    };

    // Context from where / for what the passphrase dialog was called to set the status of the checkbox
    // Partly redundant to Mode above, but offers more flexibility for future enhancements
    enum class Context {
        Unlock_Menu,    /** Unlock wallet from menu     */
        Unlock_Full,    /** Wallet needs to be fully unlocked */
        Encrypt,        /** Encrypt unencrypted wallet */
        ToggleLock,     /** Toggle wallet lock state */
        ChangePass,     /** Change passphrase */
        Send_DYN,       /** Send PIV */
        Sign_Message,   /** Sign/verify message dialog */
    };

    explicit AskPassphraseDialog(Mode mode, QWidget* parent, WalletModel* model, Context context);
    ~AskPassphraseDialog();

    void accept();

private:
    Ui::AskPassphraseDialog* ui;
    Mode mode;
    WalletModel* model;
    Context context;
    bool fCapsLock;

private Q_SLOTS:
    void textChanged();
    void secureClearPassFields();

protected:
    bool event(QEvent* event);
    bool eventFilter(QObject* object, QEvent* event);
};

#endif // DYNAMIC_QT_ASKPASSPHRASEDIALOG_H
