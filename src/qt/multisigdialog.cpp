#include <QClipboard>
#include <QWidget>
#include <QMessageBox>
#include <QScrollBar>
#include <vector>

#include "multisigdialog.h"
#include "ui_multisigdialog.h"
#include "addresstablemodel.h"
#include "base58.h"
#include "consensus/validation.h"
#include "key.h"
#include "main.h"
#include "multisigaddressentry.h"
#include "multisiginputentry.h"
#include "script/script.h"
#include "sendcoinsentry.h"
#include "util.h"
#include "wallet/wallet.h"
#include "walletmodel.h"
#include "script/standard.h"
#include "script/sign.h"
#include "rpcserver.h"
#include "policy/policy.h"
#include "txmempool.h"
#include "core_io.h"
#include "chainparamsbase.h"
#include "dns/dns.h"

#include "dbwrapper.h"

#include <QClipboard>
#include <QWidget>
#include <QMessageBox>
#include <QScrollBar>
#include <vector>

MultisigDialog::MultisigDialog(const PlatformStyle *platformStyle, QWidget *parent) :
    QDialog(parent),
	ui(new Ui::MultisigDialog), 
	model(0),
	platformStyle(platformStyle)
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
	fSetTxString = false;
}

MultisigDialog::~MultisigDialog()
{
    delete ui;
}


bool MultisigDialog::AdvertisePublicKeyForMultiSig(const std::string& address, const std::string& publickey)
{
    std::string strQuestion = "<p>Do you want to announce the public key for your " + address + " address?</p><p>You will be charged the minimun transaction fee.</p>";

    QMessageBox msgBox;
    msgBox.setWindowTitle("Advertise Public Key For MultiSig");
    msgBox.setText(strQuestion.c_str());
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if(msgBox.exec() == QMessageBox::Yes)
    {
        string strAddress = ""; 
        CNameVal name = nameValFromString("address:" + address);
        CNameVal value = nameValFromString(publickey);
        int nRentalDays = 35;
        
        NameTxReturn ret = name_operation(OP_NAME_MULTISIG, name, value, nRentalDays, address);
        if (!ret.ok)
        {
            QMessageBox::critical(this, tr("Multisig Dialog: Advertise PublicKey Error!"), tr("%1: %2").arg(ret.err_code).arg(ret.err_msg.c_str()));
            return false;
        }
    }
    return true;
}

CPubKey MultisigDialog::SearchForPubKeyByAddress(const std::string& address)
{
    CPubKey emptyPubKey;
    std::string strFindPubKey = MultiSigGetPubKeyFromAddress(address);
    if (strFindPubKey.size() > 0)
    {
        if (IsHex(strFindPubKey))
        {
            CPubKey vchPubKey(ParseHex(strFindPubKey));
            if (!vchPubKey.IsFullyValid())
            {
                QMessageBox::critical(this, tr("Multisig: SearchForPubKeyByAddress!"), tr("Invalid public key found in DDNS chain: %1").arg(strFindPubKey.c_str()));
                return emptyPubKey;
            }
            else
                return vchPubKey;
        }
    }
    return emptyPubKey;
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

    if (IsInitialBlockDownload())
    {
        QMessageBox::critical(this, tr("Multisig: Chain Download Incomplete!"), tr("Darksilk is downloading blocks..."));
        return;
    }

    unsigned int countPublicKeyEntered = ui->pubkeyEntries->count();
    if (countPublicKeyEntered > 20)
    {
        QMessageBox::critical(this, tr("Multisig: Less than 20 required."), tr("Number of addresses involved in the multisignature creation > 20.  Reduce the number!"));
        return;
    }
    // Two or more public keys are required to create a multisig address
    if (countPublicKeyEntered < 2)
    {
        QMessageBox::critical(this, tr("Multisig: 2 or more public keys required!"), tr("Current number of public keys entered is %1.").arg(countPublicKeyEntered));
        return;
    }

    CWallet *pwalletMain = model->getWallet();

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

    if((required > countPublicKeyEntered))
    {
        QMessageBox::critical(this, tr("Multisig: Required Signatures Too High"), 
            tr("Required signatures(%1) can not be greater than the total amount of public key addresses(%2)").arg(required).arg(countPublicKeyEntered));
        return;
    }

    unsigned int unknownAddressCount = 0;
    std::map<string, string> myAddressList;
    for(unsigned int i = 0; i < countPublicKeyEntered; i++)
    {
        MultisigAddressEntry *entry = qobject_cast<MultisigAddressEntry *>(ui->pubkeyEntries->itemAt(i)->widget());

        if(!entry->validate())
            return;

        std::string strAddressEntered = entry->getWalletAddress().toUtf8().constData();
        if (IsHex(strAddressEntered))
        {
            bool fError = false;
            CPubKey vchPubKey(ParseHex(strAddressEntered));
            if (!vchPubKey.IsFullyValid())
            {
                QMessageBox::critical(this, tr("Multisig: Invalid Public Key Entered!"), tr("Invalid public key: %1").arg(strAddressEntered.c_str()));
                fError = true;
            }
            if (fError)
                unknownAddressCount++;
            else
                pubkeys[i] = vchPubKey;
        }
        else
        {
            CDarkSilkAddress address(strAddressEntered);
            if (pwalletMain && address.IsValid())
            {
                bool fError = false;
                CKeyID keyID;
                if (!address.GetKeyID(keyID))
                {
                    QMessageBox::critical(this, tr("Multisig: Invalid Public Key Entered!"), tr("%1  does not refer to a key").arg(strAddressEntered.c_str()));
                    fError = true;
                }
                
                CPubKey vchPubKey;
                if ((fError == false) && (!pwalletMain->GetPubKey(keyID, vchPubKey)))
                    vchPubKey = SearchForPubKeyByAddress(strAddressEntered);

                if ((fError == false) && !vchPubKey.IsFullyValid())
                    fError = true;

                if (fError)
                {
                    unknownAddressCount = unknownAddressCount + 1;
                }
                else
                {
                    myAddressList.insert(pair<string, string>(strAddressEntered, HexStr(vchPubKey.begin(), vchPubKey.end())));
                    pubkeys[i] = vchPubKey;
                }
            }
        }
    }

    if (unknownAddressCount == 0)
    {
        CScript script = GetScriptForMultisig(required, pubkeys);
        CScriptID scriptID = GetScriptID(script);
        CDarkSilkAddress multiSigAddress(scriptID);
        ui->multisigAddress->setText(multiSigAddress.ToString().c_str());
        ui->redeemScript->setText(HexStr(script.begin(), script.end()).c_str());
    }
    else
    {
        for(std::map<string,string>::iterator iter = myAddressList.begin(); iter != myAddressList.end(); ++iter)
        {
            // TODO (Amir): If user agrees to advertise their public key using DDNS as a multisig name (fee = 0.001 SLK)
            AdvertisePublicKeyForMultiSig(iter->first, iter->second);
        }
    }
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

    CDarkSilkAddress dslkMultiSigAddress(address);
    if(!pwalletMain->mapAddressBook.count(dslkMultiSigAddress.Get()))
		pwalletMain->SetAddressBook(dslkMultiSigAddress.Get(), label, "multisig address");
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
    MultisigAddressEntry *entry = new MultisigAddressEntry(platformStyle, this);

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
    fSetTxString = true;
    ss << transaction;
    ui->transaction->setText(HexStr(ss.begin(), ss.end()).c_str());
    if(ui->transaction->text().size() > 0)
        ui->signTransactionButton->setEnabled(true);
    else
        ui->signTransactionButton->setEnabled(false);
    fSetTxString = false;
}

void MultisigDialog::on_transaction_textChanged()
{ 
	if (fSetTxString)		
	        return;
	        
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
  ui->signedTransaction->clear();

    if(!model)
        return;

    CWallet *pwalletMain = model->getWallet();

    std::string strSignatureHex = ui->transaction->text().toStdString();
    // Check input for valid hex string
    if (!IsHex(strSignatureHex))
        return;

    // Decode the raw transaction
    std::vector<unsigned char> txData(ParseHex(strSignatureHex));
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
    redeemScripts redeemScripts;
    CCoinsView viewDummy;
    CCoinsViewCache view(&viewDummy);
    {
        LOCK(mempool.cs);
        CCoinsViewCache &viewChain = *pcoinsTip;
        CCoinsViewMemPool viewMempool(&viewChain, mempool);
        view.SetBackend(viewMempool); // temporarily switch cache backend to db+mempool view
        unsigned int x = 0;
        BOOST_FOREACH(const CTxIn& txin, mergedTx.vin) {
            
            const uint256& prevHash = txin.prevout.hash;
            CCoins coins;
            view.AccessCoins(prevHash); // this is certainly allowed to fail
            MultisigInputEntry *entry = qobject_cast<MultisigInputEntry *>(ui->inputs->itemAt(x)->widget());
            QString redeemScriptStr = entry->getRedeemScript();
            if(redeemScriptStr.size() > 0)
            {
                std::vector<unsigned char> scriptData(ParseHex(redeemScriptStr.toStdString()));
                CScript redeemScript(scriptData.begin(), scriptData.end());
                pwalletMain->AddCScript(redeemScript);
                redeemScripts.push_back(redeemScript);
            }
            else
            {
                CScript emptyScript;
                redeemScripts.push_back(emptyScript);
            }
        }

        view.SetBackend(viewDummy); // switch back to avoid locking mempool for too long
    }
    
	EnsureWalletIsUnlocked();
	
    // Sign what we can
    const CKeyStore& keystore = *pwalletMain;
    int nHashType = SIGHASH_ALL;
    bool fComplete = true;

    //bool fHashSingle = ((nHashType & ~SIGHASH_ANYONECANPAY) == SIGHASH_SINGLE);
    bool fHashSingle = true;
    for(unsigned int i = 0; i < mergedTx.vin.size(); i++)
    {
        CScript redeemScript = redeemScripts[i];
        CTxIn& txin = mergedTx.vin[i];
        const CCoins* coins = view.AccessCoins(txin.prevout.hash);
        if (coins == NULL || !coins->IsAvailable(txin.prevout.n)) {
            QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("Input not found or already spent in coins"));
            fComplete = false;
            continue;
        }

        const CScript& prevPubKey = coins->vout[txin.prevout.n].scriptPubKey;

        txin.scriptSig.clear();

        // Only sign SIGHASH_SINGLE if there's a corresponding output:
        if (!fHashSingle || (i < mergedTx.vout.size()))
        {
            SignSignature(keystore, prevPubKey, mergedTx, i, nHashType);
        }
        // ... and merge in other signatures:
        unsigned int sigCount = 0;
        BOOST_FOREACH(const CMutableTransaction& txv, txVariants) {
            txin.scriptSig = CombineSignatures(prevPubKey, mergedTx, i, txin.scriptSig, txv.vin[i].scriptSig);
            sigCount = sigCount + 1;
        }

        ScriptError scriptError = SCRIPT_ERR_OK;
        if (!VerifyScript(txin.scriptSig, prevPubKey, STANDARD_SCRIPT_VERIFY_FLAGS, MutableTransactionSignatureChecker(&mergedTx, i)), scriptError)
        {
            if (scriptError != 0)
            {
                QMessageBox::critical(this, tr("Multisig: Sign Button failed!"), tr("VerifyScript failed."));
                fComplete = false;
            }
        }
        unsigned int requiredSigs = redeemScript.GetSigOpCount(redeemScript) - 1;
        if (requiredSigs > sigCount)
        {
            QMessageBox::critical(this, tr("Multisig: More signatures needed!"), tr("sigCount=%1, requiredSigs = %2").arg(sigCount).arg(requiredSigs));
            fComplete = false;
        }
    }

    ui->signedTransaction->setText(EncodeHexTx(mergedTx).c_str());
    ui->sendTransactionButton->setEnabled(fComplete);
    if(fComplete)
    {
        ui->statusLabel->setText(tr("Transaction signature is complete"));
    }
    else
    {
        ui->statusLabel->setText(tr("Transaction is NOT completely signed"));
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
    int64_t fee = (int64_t) (ui->fee->text().toDouble() * COIN);
    int64_t minFee = DEFAULT_TRANSACTION_MINFEE * (1 + (int64_t) transactionSize / 1000);
    if(fee < minFee)
    {
        QMessageBox::StandardButton ret = QMessageBox::question(this, tr("Confirm send transaction"), tr("The fee of the transaction (%1 DSLK) is smaller than the expected fee (%2 DSLK). Do you want to send the transaction anyway?").arg((double) fee / COIN).arg((double) minFee / COIN), QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
        if(ret != QMessageBox::Yes)
            return;
    }
    else if(fee > minFee)
    {
        QMessageBox::StandardButton ret = QMessageBox::question(this, tr("Confirm send transaction"), tr("The fee of the transaction (%1 DSLK) is bigger than the expected fee (%2 DSLK). Do you want to send the transaction anyway?").arg((double) fee / COIN).arg((double) minFee / COIN), QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
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
uint256 blockHash = uint256S("0");
    if(GetTransaction(txHash, existingTx, Params().GetConsensus(), blockHash))
    {
        if(blockHash != 0)
            return;
    }

    CMerkleTx cmt;
    // Send the transaction to the local node
    //   CTxDB txdb("r");
    if(!cmt.AcceptToMemoryPool(false))
    return;
    SyncWithWallets(tx, NULL);
    //(CInv(MSG_TX, txHash), tx);
    RelayTransaction(tx);
}

MultisigInputEntry * MultisigDialog::addInput()
{
    MultisigInputEntry *entry = new MultisigInputEntry(platformStyle, this);

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
    SendCoinsEntry *entry = new SendCoinsEntry(platformStyle, this);

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
    int64_t inputsAmount = 0;
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
    int64_t outputsAmount = 0;
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
    int64_t fee = inputsAmount - outputsAmount;
    QString feeStr;
    feeStr.sprintf("%.6f", (double) fee / COIN);
    ui->fee->setText(feeStr);
}
