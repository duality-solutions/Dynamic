// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "multisigdialog.h"

#include "ui_multisigdialog.h"
#include "addresstablemodel.h"
#include "base58.h"
#include "core_io.h"
#include "key.h"
#include "main.h"
#include "multisigaddressentry.h"
#include "multisiginputentry.h"
#include "rpcserver.h"
#include "script/script.h"
#include "script/sign.h"
#include "script/standard.h"
#include "sendcoinsentry.h"
#include "util.h"
#include "wallet/wallet.h"
#include "walletmodel.h"

#include <QClipboard>
#include <QWidget>
#include <QMessageBox>
#include <QScrollBar>
#include <vector>

MultisigDialog::MultisigDialog(QWidget *parent)
    : QDialog(parent), ui(new Ui::MultisigDialog), model(0)
{
    ui->setupUi(this);

#ifdef Q_WS_MAC // Icons on push buttons are very uncommon on Mac
    ui->addPubKeyButton->setIcon(QIcon());
    ui->clearButton->setIcon(QIcon());
    ui->addInputButton->setIcon(QIcon());
    ui->addOutputButton->setIcon(QIcon());
    ui->signTransactionButton->setIcon(QIcon());
    ui->sendTransactionButton->setIcon(QIcon());
#endif

    addPubKey();
    addPubKey();

    connect(ui->addPubKeyButton, SIGNAL(clicked()), this, SLOT(addPubKey()));
    connect(ui->clearButton, SIGNAL(clicked()), this, SLOT(clear()));

    addInput();
    addOutput();
    updateAmounts();

    connect(ui->addInputButton, SIGNAL(clicked()), this, SLOT(addInput()));
    connect(ui->addOutputButton, SIGNAL(clicked()), this, SLOT(addOutput()));

    ui->signTransactionButton->setEnabled(false);
    ui->sendTransactionButton->setEnabled(false);
}

MultisigDialog::~MultisigDialog()
{
    delete ui;
}

void MultisigDialog::setModel(WalletModel *model)
{
    this->model = model;

    for(int i = 0; i < ui->pubkeyEntries->count(); i++)
    {
        MultisigAddressEntry *entry = qobject_cast<MultisigAddressEntry *>(ui->pubkeyEntries->itemAt(i)->widget());
        if(entry)
            entry->setModel(model);
    }


    for(int i = 0; i < ui->inputs->count(); i++)
    {
        MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(i)->widget());
        if(entry)
            entry->setModel(model);
    }


    for(int i = 0; i < ui->outputs->count(); i++)
    {
        SendCoinsEntry *entry = qobject_cast<SendCoinsEntry *>(ui->outputs->itemAt(i)->widget());
        if(entry)
            entry->setModel(model);
    }
}

void MultisigDialog::updateRemoveEnabled()
{
    bool enabled = (ui->pubkeyEntries->count() > 2);

    for(int i = 0; i < ui->pubkeyEntries->count(); i++)
    {
        MultisigAddressEntry *entry = qobject_cast<MultisigAddressEntry *>(ui->pubkeyEntries->itemAt(i)->widget());
        if(entry)
            entry->setRemoveEnabled(enabled);
    }

    QString maxSigsStr;
    maxSigsStr.setNum(ui->pubkeyEntries->count());
    ui->maxSignaturesLabel->setText(QString("/ ") + maxSigsStr);


    enabled = (ui->inputs->count() > 1);
    for(int i = 0; i < ui->inputs->count(); i++)
    {
        MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(i)->widget());
        if(entry)
            entry->setRemoveEnabled(enabled);
    }


    enabled = (ui->outputs->count() > 1);
    for(int i = 0; i < ui->outputs->count(); i++)
    {
        SendCoinsEntry *entry = qobject_cast<SendCoinsEntry *>(ui->outputs->itemAt(i)->widget());
        if(entry)
            entry->setRemoveEnabled(enabled);
    }
}

void MultisigDialog::on_createAddressButton_clicked()
{
    ui->multisigAddress->clear();
    ui->redeemScript->clear();

    if(!model)
        return;

    CWallet *pwalletMain = model->getWallet();

    unsigned int countPublicKeyEntered = ui->pubkeyEntries->count();
    // Two or more public keys are required to create a multisig address
    if (countPublicKeyEntered < 2)
    {
        QMessageBox::critical(this, tr("Two or more public keys are required"), tr("(Current number of public keys entered is %1.").arg(countPublicKeyEntered));
        return;
    }

    std::vector<CPubKey> pubkeys;
    pubkeys.resize(countPublicKeyEntered);

    unsigned int required = ui->requiredSignatures->text().toUInt();
    if (required < 1)
    {
        // Multisignature address requires 1 or more signatures.
        QMessageBox::critical(this, tr("Multisig: Required Signatures Error!"), 

            tr("A multisignature address requires at least one key to redeem. Currently set to %1.").arg(required));
            return;
    }
    
    for(unsigned int i = 0; i < countPublicKeyEntered; i++)
    {
        MultisigAddressEntry *entry = qobject_cast<MultisigAddressEntry *>(ui->pubkeyEntries->itemAt(i)->widget());

    if(!entry->validate())
            return;

        std::string strPublicKeyEntered = entry->getWalletAddress().toUtf8().constData();
        CDarkSilkAddress address(strPublicKeyEntered);
        if (pwalletMain && address.IsValid())
        {
            CKeyID keyID;
            if (!address.GetKeyID(keyID))
                QMessageBox::critical(this, tr("Multisig: Invalid Public Key Entered!"), tr("%1  does not refer to a key").arg(strPublicKeyEntered.c_str()));
            CPubKey vchPubKey;
            if (!pwalletMain->GetPubKey(keyID, vchPubKey))
                QMessageBox::critical(this, tr("Multisig: Invalid Public Key Entered!"), tr("No full public key for address %1").arg(strPublicKeyEntered.c_str()));
            if (!vchPubKey.IsFullyValid())
                QMessageBox::critical(this, tr("Multisig: Invalid Public Key Entered!"), tr("Invalid public key: %1").arg(strPublicKeyEntered.c_str()));

            pubkeys[i] = vchPubKey;
        }
        else
        {
            if (IsHex(strPublicKeyEntered))
            {
                CPubKey vchPubKey(ParseHex(strPublicKeyEntered));
                if (!vchPubKey.IsFullyValid())
                    QMessageBox::critical(this, tr("Multisig: Invalid Public Key Entered!"), tr("Invalid public key: %1").arg(strPublicKeyEntered.c_str()));
                pubkeys[i] = vchPubKey;
            }
        }
     }

    if((required > pubkeys.size()))
    {
        QMessageBox::critical(this, tr("Multisig: Required Signatures Too High"), 
            tr("Required signatures(%1) can not be greater than the total amount of public key addresses(%2)").arg(required).arg(pubkeys.size()));

        return;
    }

    CScript script = GetScriptForMultisig(required, pubkeys);
    CScriptID scriptID = GetScriptID(script);
    CDarkSilkAddress multiSigAddress(scriptID);

    ui->multisigAddress->setText(multiSigAddress.ToString().c_str());
    ui->redeemScript->setText(HexStr(script.begin(), script.end()).c_str());
}

void MultisigDialog::on_copyMultisigAddressButton_clicked()
{
    QApplication::clipboard()->setText(ui->multisigAddress->text());
}

void MultisigDialog::on_copyRedeemScriptButton_clicked()
{
    QApplication::clipboard()->setText(ui->redeemScript->text());
}

void MultisigDialog::on_saveRedeemScriptButton_clicked()
{
    if(!model)
        return;

    CWallet *wallet = model->getWallet();
    std::string redeemScript = ui->redeemScript->text().toStdString();
    std::vector<unsigned char> scriptData(ParseHex(redeemScript));
    CScript script(scriptData.begin(), scriptData.end());
    CScriptID scriptID = GetScriptID(script);

    LOCK(wallet->cs_wallet);
    if(!wallet->HaveCScript(scriptID))
        wallet->AddCScript(script);
}

void MultisigDialog::on_saveMultisigAddressButton_clicked()
{
    if(!model)
        return;

    CWallet *pwalletMain = model->getWallet();
    std::string redeemScript = ui->redeemScript->text().toStdString();
    std::string address = ui->multisigAddress->text().toStdString();
    std::string label("multisig");

    if(!model->validateAddress(QString(address.c_str())))
        return;

    std::vector<unsigned char> scriptData(ParseHex(redeemScript));
    CScript script(scriptData.begin(), scriptData.end());
    CScriptID scriptID = GetScriptID(script);

    LOCK(pwalletMain->cs_wallet);
    if(!pwalletMain->HaveCScript(scriptID))
        pwalletMain->AddCScript(script);

    CDarkSilkAddress silkMultiSigAddress(address);
    if(!pwalletMain->mapAddressBook.count(silkMultiSigAddress.Get()))
        pwalletMain->SetAddressBook(silkMultiSigAddress.Get(), label, "multisig address");
}

void MultisigDialog::clear()
{
    while(ui->pubkeyEntries->count())
        delete ui->pubkeyEntries->takeAt(0)->widget();

    addPubKey();
    addPubKey();
    updateRemoveEnabled();
}

MultisigAddressEntry * MultisigDialog::addPubKey()
{
    MultisigAddressEntry *entry = new MultisigAddressEntry(this);

    entry->setModel(model);
    ui->pubkeyEntries->addWidget(entry);
    connect(entry, SIGNAL(removeEntry(MultisigAddressEntry *)), this, SLOT(removeEntry(MultisigAddressEntry *)));
    updateRemoveEnabled();
    entry->clear();
    ui->scrollAreaWidgetContents->resize(ui->scrollAreaWidgetContents->sizeHint());
    QScrollBar *bar = ui->scrollArea->verticalScrollBar();
    if(bar)
        bar->setSliderPosition(bar->maximum());

    return entry;
}

void MultisigDialog::removeEntry(MultisigAddressEntry *entry)
{
    delete entry;
    updateRemoveEnabled();
}

void MultisigDialog::on_createTransactionButton_clicked()
{
    CMutableTransaction transaction;

    // Get inputs
    for(int i = 0; i < ui->inputs->count(); i++)
    {
        MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(i)->widget());
        if(entry)
        {
            if(entry->validate())
            {
                CTxIn input = entry->getInput();
                transaction.vin.push_back(input);
            }
            else
                return;
        }
    }

    // Get outputs
   for(int i = 0; i < ui->outputs->count(); i++)
    {
        SendCoinsEntry *entry = qobject_cast<SendCoinsEntry *>(ui->outputs->itemAt(i)->widget());

        if(entry)
        {
            if(entry->validate())
            {
                SendCoinsRecipient recipient = entry->getValue();
                CDarkSilkAddress address(recipient.address.toStdString());
                CScript scriptPubKey = GetScriptForDestination(address.Get());
                CAmount amount = recipient.amount;
                CTxOut output(amount, scriptPubKey);
                transaction.vout.push_back(output);
            }
            else
                return;
        }
    }

    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    ss << transaction;
    ui->transaction->setText(HexStr(ss.begin(), ss.end()).c_str());
}

void MultisigDialog::on_transaction_textChanged()
{
    while(ui->inputs->count())
        delete ui->inputs->takeAt(0)->widget();
    
    while(ui->outputs->count())
        delete ui->outputs->takeAt(0)->widget();

    if(ui->transaction->text().size() > 0)
        ui->signTransactionButton->setEnabled(true);
    else
        ui->signTransactionButton->setEnabled(false);

    // Decode the raw transaction
    std::vector<unsigned char> txData(ParseHex(ui->transaction->text().toStdString()));
    CDataStream ss(txData, SER_NETWORK, PROTOCOL_VERSION);
    CTransaction tx;
    try
    {
        ss >> tx;
    }
    catch(std::exception &e)
    {
        return;
    }

    // Fill input list
    int index = -1;
    BOOST_FOREACH(const CTxIn& txin, tx.vin)
    {
        uint256 prevoutHash = txin.prevout.hash;
        addInput();
        index++;
        MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(index)->widget());
        if(entry)
        {
            entry->setTransactionId(QString(prevoutHash.GetHex().c_str()));
            entry->setTransactionOutputIndex(txin.prevout.n);
        }
    }

    // Fill output list
    index = -1;
    BOOST_FOREACH(const CTxOut& txout, tx.vout)
    {
        CScript scriptPubKey = txout.scriptPubKey;
        CTxDestination addr;
        ExtractDestination(scriptPubKey, addr);
        CDarkSilkAddress address(addr);
        SendCoinsRecipient recipient;
        recipient.address = QString(address.ToString().c_str());
        recipient.amount = txout.nValue;
        addOutput();
        index++;
        SendCoinsEntry *entry = qobject_cast<SendCoinsEntry *>(ui->outputs->itemAt(index)->widget());
        if(entry)
        {
            entry->setValue(recipient);
        }
    }

    updateRemoveEnabled();
}

void MultisigDialog::on_copyTransactionButton_clicked()
{
   QApplication::clipboard()->setText(ui->transaction->text());
}

void MultisigDialog::on_pasteTransactionButton_clicked()
{
    ui->transaction->setText(QApplication::clipboard()->text());
}

void MultisigDialog::on_signTransactionButton_clicked()
{
    //TODO (Amir) Sign multisig not working.
    ui->signedTransaction->clear();

    if(!model)
        return;

    CWallet *pwalletMain = model->getWallet();

    // Decode the raw transaction
    std::vector<unsigned char> txData(ParseHex(ui->transaction->text().toStdString()));
    CDataStream ssData(txData, SER_NETWORK, PROTOCOL_VERSION);
    std::vector<CMutableTransaction> txVariants;
    while (!ssData.empty()) {
        try {
            CMutableTransaction tx;
            ssData >> tx;
            txVariants.push_back(tx);
        }
        catch (const std::exception &) {
            QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("TX decodefailed"));
        }
    }

    if (txVariants.empty())
        QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("Missing transaction"));
    // mergedTx will end up with all the signatures; it
    // starts as a clone of the rawtx:
    CMutableTransaction mergedTx(txVariants[0]);
    // Fetch previous transactions (inputs):
    CCoinsView viewDummy;
    CCoinsViewCache view(&viewDummy);
    {
        AssertLockHeld(mempool.cs);
        CCoinsViewCache &viewChain = *pcoinsTip;
        CCoinsViewMemPool viewMempool(&viewChain, mempool);

        view.SetBackend(viewMempool); // temporarily switch cache backend to db+mempool view

        BOOST_FOREACH(const CTxIn& txin, mergedTx.vin) 
        {
            const uint256& prevHash = txin.prevout.hash;
            CCoins coins;
            view.AccessCoins(prevHash); // this is certainly allowed to fail
        }

        view.SetBackend(viewDummy); // switch back to avoid locking mempool for too long
    }

    EnsureWalletIsUnlocked();

    for(int i = 0; i < ui->inputs->count(); i++)
    {
        MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(i)->widget());
        if(entry)
       {
            QString redeemScriptStr = entry->getRedeemScript();
            if(redeemScriptStr.size() > 0)
            {
                std::vector<unsigned char> scriptData(ParseHex(redeemScriptStr.toStdString()));
                CScript redeemScript(scriptData.begin(), scriptData.end());
                pwalletMain->AddCScript(redeemScript);
            }
        }
    }

    // Sign what we can
    const CKeyStore& keystore = *pwalletMain;
    int nHashType = SIGHASH_ALL;
    bool fComplete = true;
    //bool fHashSingle = ((nHashType & ~SIGHASH_ANYONECANPAY) == SIGHASH_SINGLE);
    bool fHashSingle = true;
    for(unsigned int i = 0; i < mergedTx.vin.size(); i++)
    {
        CTxIn& txin = mergedTx.vin[i];
        const CCoins* coins = view.AccessCoins(txin.prevout.hash);
        if (coins == NULL || !coins->IsAvailable(txin.prevout.n)) 
        {
            QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("Input not found or already spent in coins"));            
            fComplete = false;
            continue;
        }

        const CScript& prevPubKey = coins->vout[txin.prevout.n].scriptPubKey;

        txin.scriptSig.clear();
        // Only sign SIGHASH_SINGLE if there's a corresponding output:
        if (!fHashSingle || (i < mergedTx.vout.size()))
        {
            if (!SignSignature(keystore, prevPubKey, mergedTx, i, nHashType))
            {
                CScriptID scriptID = GetScriptID(prevPubKey);
                CDarkSilkAddress multiSigPubKeyAddress(scriptID);
                QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("SignSignature failed for pub key = %1").arg(multiSigPubKeyAddress.ToString().c_str()));
                fComplete = false;
                break;
            }
        }
        // ... and merge in other signatures:
        BOOST_FOREACH(const CMutableTransaction& txv, txVariants) {
            txin.scriptSig = CombineSignatures(prevPubKey, mergedTx, i, txin.scriptSig, txv.vin[i].scriptSig);
        }

        if (!VerifyScript(txin.scriptSig, prevPubKey, MANDATORY_SCRIPT_VERIFY_FLAGS, MutableTransactionSignatureChecker(&mergedTx, i)))
        {
            QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("VerifyScript failed."));
            fComplete = false;
        }
    }

    ui->signedTransaction->setText(EncodeHexTx(mergedTx).c_str());

    if(fComplete)
    {
        ui->statusLabel->setText(tr("Transaction signature is complete"));
        ui->sendTransactionButton->setEnabled(true);
    }
    else
    {
        ui->statusLabel->setText(tr("Transaction is NOT completely signed"));
        ui->sendTransactionButton->setEnabled(true);
    }
}

void MultisigDialog::on_copySignedTransactionButton_clicked()
{
    QApplication::clipboard()->setText(ui->signedTransaction->text());
}

void MultisigDialog::on_sendTransactionButton_clicked()
{
    int64_t transactionSize = ui->signedTransaction->text().size() / 2;
    if(transactionSize == 0)
        return;

    // Check the fee
    CWallet *pwalletMain = model->getWallet();
    CAmount fee = (CAmount) (ui->fee->text().toDouble() * COIN);
    //TODO::Change nbytes of ui->signedTransaction
    size_t nbytes = 256; 
    CAmount minFee = (CAmount)(pwalletMain->minTxFee.GetFee(nbytes) * (1 + (int64_t)transactionSize / 1000));    if(fee < minFee)
    {
        QMessageBox::StandardButton ret = QMessageBox::question(this, tr("Confirm send transaction"), tr("The fee of the transaction (%1 DRKSLK) is smaller than the expected fee (%2 DRKSLK). Do you want to send the transaction anyway?").arg((double) fee / COIN).arg((double) minFee / COIN), QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
        if(ret != QMessageBox::Yes)
            return;
    }
    else if(fee > minFee)
    {
        QMessageBox::StandardButton ret = QMessageBox::question(this, tr("Confirm send transaction"), tr("The fee of the transaction (%1 DRKSLK) is bigger than the expected fee (%2 DRKSLK). Do you want to send the transaction anyway?").arg((double) fee / COIN).arg((double) minFee / COIN), QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
        if(ret != QMessageBox::Yes)
            return;
    }

    // Decode the raw transaction
    std::vector<unsigned char> txData(ParseHex(ui->signedTransaction->text().toStdString()));
    CDataStream ssData(txData, SER_NETWORK, PROTOCOL_VERSION);
    CTransaction tx;
    try
    {
        ssData >> tx;
    }
    catch(std::exception &e)
    {
        return;
    }
    uint256 txHash = tx.GetHash();

    // Check if the transaction is already in the blockchain
    CTransaction existingTx;
    uint256 blockHash = 0;
    if(GetTransaction(txHash, existingTx, blockHash))
    {
        if(blockHash != 0)
            return;
    }

    CValidationState state;
    // Send the transaction to the local node
    //   CTxDB txdb("r");
    if(!AcceptToMemoryPool(mempool, state, *this, fLimitFree, NULL, false, fRejectAbsurdFee))
        return;
    SyncWithWallets(tx, NULL);
    //(CInv(MSG_TX, txHash), tx);
    RelayTransaction(tx);
}

MultisigInputEntry * MultisigDialog::addInput()
{
    MultisigInputEntry *entry = new MultisigInputEntry(this);

    entry->setModel(model);
    ui->inputs->addWidget(entry);
    connect(entry, SIGNAL(removeEntry(MultisigInputEntry *)), this, SLOT(removeEntry(MultisigInputEntry *)));
    connect(entry, SIGNAL(updateAmount()), this, SLOT(updateAmounts()));
    updateRemoveEnabled();
    entry->clear();
    ui->scrollAreaWidgetContents_2->resize(ui->scrollAreaWidgetContents_2->sizeHint());
    QScrollBar *bar = ui->scrollArea_2->verticalScrollBar();
    if(bar)
        bar->setSliderPosition(bar->maximum());

    return entry;
}

void MultisigDialog::removeEntry(MultisigInputEntry *entry)
{
    delete entry;
    updateRemoveEnabled();
}

SendCoinsEntry * MultisigDialog::addOutput()
{
    SendCoinsEntry *entry = new SendCoinsEntry(this);

    entry->setModel(model);
    ui->outputs->addWidget(entry);
    connect(entry, SIGNAL(removeEntry(SendCoinsEntry *)), this, SLOT(removeEntry(SendCoinsEntry *)));
    connect(entry, SIGNAL(payAmountChanged()), this, SLOT(updateAmounts()));
    updateRemoveEnabled();
    entry->clear();
    ui->scrollAreaWidgetContents_3->resize(ui->scrollAreaWidgetContents_3->sizeHint());
    QScrollBar *bar = ui->scrollArea_3->verticalScrollBar();
    if(bar)
        bar->setSliderPosition(bar->maximum());

    return entry;
}

void MultisigDialog::removeEntry(SendCoinsEntry *entry)
{
    delete entry;
    updateRemoveEnabled();
}

void MultisigDialog::updateAmounts()
{
    // Update inputs amount
    CAmount inputsAmount = 0;
    for(int i = 0; i < ui->inputs->count(); i++)
    {
        MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(i)->widget());
        if(entry)
            inputsAmount += entry->getAmount();
    }
    QString inputsAmountStr;
    inputsAmountStr.sprintf("%.6f", (double) inputsAmount / COIN);
    ui->inputsAmount->setText(inputsAmountStr);

    // Update outputs amount
    CAmount outputsAmount = 0;
    for(int i = 0; i < ui->outputs->count(); i++)
    {
        SendCoinsEntry *entry = qobject_cast<SendCoinsEntry *>(ui->outputs->itemAt(i)->widget());
        if(entry)
            outputsAmount += entry->getValue().amount;
    }
    QString outputsAmountStr;
    outputsAmountStr.sprintf("%.6f", (double) outputsAmount / COIN);
    ui->outputsAmount->setText(outputsAmountStr);

    // Update Fee amount
    CAmount fee = inputsAmount - outputsAmount;
    QString feeStr;
    feeStr.sprintf("%.6f", (double) fee / COIN);
    ui->fee->setText(feeStr);
}