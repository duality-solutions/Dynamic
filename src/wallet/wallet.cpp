// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2017-2019 The Particl Core developers
// Copyright (c) 2014 The ShadowCoin developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "wallet/wallet.h"

#include "assets/assets.h"
#include "base58.h"
#include "bdap/bdap.h"
#include "bdap/domainentrydb.h"
#include "bdap/linkingdb.h"
#include "bdap/linkstorage.h"
#include "bdap/stealth.h"
#include "bdap/utils.h"
#include "chain.h"
#include "checkpoints.h"
#include "consensus/consensus.h"
#include "consensus/validation.h"
#include "core_io.h"
#include "dynode-sync.h"
#include "fluid/fluid.h"
#include "fluid/fluiddb.h"
#include "fs.h"
#include "governance.h"
#include "init.h"
#include "instantsend.h"
#include "keepass.h"
#include "key.h"
#include "keystore.h"
#include "net.h"
#include "policy/fees.h"
#include "policy/policy.h"
#include "pos/kernel.h"
#include "pos/stakeinput.h"
#include "primitives/block.h"
#include "primitives/transaction.h"
#include "privatesend-client.h"
#include "rpc/protocol.h"
#include "script/script.h"
#include "script/sign.h"
#include "spork.h"
#include "timedata.h"
#include "txmempool.h"
#include "ui_interface.h"
#include "util.h"
#include "utilmoneystr.h"
#include "validation.h" // For CheckTransaction
#include "wallet/coincontrol.h"
#include "wallet/fees.h"

#include <assert.h>

#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

CWallet* pwalletMain = nullptr;

/** Transaction fee set by the user */
CFeeRate payTxFee(DEFAULT_TRANSACTION_FEE);
unsigned int nTxConfirmTarget = DEFAULT_TX_CONFIRM_TARGET;
bool bSpendZeroConfChange = DEFAULT_SPEND_ZEROCONF_CHANGE;
bool fSendFreeTransactions = DEFAULT_SEND_FREE_TRANSACTIONS;
bool fWalletUnlockMixStakeOnly = WALLET_UNLOCKED_FOR_MIXING_STAKING_ONLY;
bool fWalletRbf = DEFAULT_WALLET_RBF;

const char* DEFAULT_WALLET_DAT = "wallet.dat";
const char* DEFAULT_WALLET_DAT_MNEMONIC = "wallet_mnemonic.dat";

/** 
 * Fees smaller than this (in satoshis) are considered zero fee (for transaction creation)
 * Override with -mintxfee
 */
CFeeRate CWallet::minTxFee = CFeeRate(DEFAULT_TRANSACTION_MINFEE);
/**
 * If fee estimation does not have enough data to provide estimates, use this fee instead.
 * Has no effect if not using fee estimation
 * Override with -fallbackfee
 */
CFeeRate CWallet::fallbackFee = CFeeRate(DEFAULT_FALLBACK_FEE);

CFeeRate CWallet::m_discard_rate = CFeeRate(DEFAULT_DISCARD_FEE);

const uint256 CMerkleTx::ABANDON_HASH(uint256S("0000000000000000000000000000000000000000000000000000000000000001"));

/** @defgroup mapWallet
 *
 * @{
 */

struct CompareValueOnly
{
    bool operator()(const CInputCoin& t1, 
                    const CInputCoin& t2) const
    {
        return t1.txout.nValue < t2.txout.nValue;
    }
};

struct CompareAssetValueOnly
{
    bool operator()(const std::pair<CInputCoin, CAmount>& t1,
                    const std::pair<CInputCoin, CAmount>& t2) const
    {
        return t1.second < t2.second;
    }
};

std::string COutput::ToString() const
{
    return strprintf("COutput(%s, %d, %d) [%s]", tx->GetHash().ToString(), i, nDepth, FormatMoney(tx->tx->vout[i].nValue));
}

int COutput::Priority() const
{
    for (const auto& d : CPrivateSend::GetStandardDenominations()) {
        // large denoms have lower value
        if(tx->tx->vout[i].nValue == d) return (float)COIN / d * 10000;
    }
    if(tx->tx->vout[i].nValue < 1*COIN) return 20000;

    //nondenom return largest first
    return -(tx->tx->vout[i].nValue/COIN);
}

const CWalletTx* CWallet::GetWalletTx(const uint256& hash) const
{
    LOCK(cs_wallet);
    std::map<uint256, CWalletTx>::const_iterator it = mapWallet.find(hash);
    if (it == mapWallet.end())
        return nullptr;
    return &(it->second);
}

void CWallet::GenerateEdandStealthKey(CKey& keyIn)
{
    CKeyEd25519 secretEdRet;
    DeriveEd25519ChildKey(keyIn,secretEdRet); //Derive Ed25519 key
    DeriveChildStealthKey(keyIn); //Derive stealth key
}

CPubKey CWallet::GenerateNewKey(uint32_t nAccountIndex, bool fInternal)
{
    AssertLockHeld(cs_wallet);                                 // mapKeyMetadata
    bool fCompressed = CanSupportFeature(FEATURE_COMPRPUBKEY); // default to compressed public keys if we want 0.6.0 wallets

    CKey secret;

    // Create new metadata
    int64_t nCreationTime = GetTime();
    CKeyMetadata metadata(nCreationTime);

    CPubKey pubkey;
    // use HD key derivation if HD was enabled during wallet creation
    if (IsHDEnabled()) {
        DeriveNewChildKey(metadata, secret, nAccountIndex, fInternal);
        pubkey = secret.GetPubKey();
    } else {
        secret.MakeNewKey(fCompressed);

        // Compressed public keys were introduced in version 0.6.0
        if (fCompressed)
            SetMinVersion(FEATURE_COMPRPUBKEY);

        pubkey = secret.GetPubKey();
        assert(secret.VerifyPubKey(pubkey));

        // Create new metadata
        mapKeyMetadata[pubkey.GetID()] = metadata;
        UpdateTimeFirstKey(nCreationTime);

        if (!AddKeyPubKey(secret, pubkey))
            throw std::runtime_error(std::string(__func__) + ": AddKey failed");
    }
    return pubkey;
}

std::vector<unsigned char> CWallet::GenerateNewEdKey(uint32_t nAccountIndex, bool fInternal, const CKey& seedIn)
{
    bool fCompressed = CanSupportFeature(FEATURE_COMPRPUBKEY); // default to compressed public keys if we want 0.6.0 wallets
    CKey secretRet;
    secretRet.MakeNewKey(fCompressed);

    // Create new metadata
    int64_t nCreationTime = GetTime();
    CKeyMetadata metadata(nCreationTime);

    CKeyEd25519 secretEdRet;

    if (!seedIn.IsValid())
    {
        DeriveEd25519ChildKey(secretRet,secretEdRet);
    }
    else
    {
        DeriveEd25519ChildKey(seedIn,secretEdRet);
    };

    // Create new metadata
    mapKeyMetadata[secretEdRet.GetID()] = metadata;
    UpdateTimeFirstKey(nCreationTime);

    return secretEdRet.GetPubKey();
} //GenerateNewEdKey

std::array<char, 32> CWallet::ConvertSecureVector32ToArray(const std::vector<unsigned char, secure_allocator<unsigned char> >& vIn)
{
    std::array<char, 32> arrReturn;
    for(unsigned int i = 0; i < 32; i++) {
         arrReturn[i] = (char)vIn[i];
    }
    return arrReturn;
}

void CWallet::DeriveEd25519ChildKey(const CKey& seed, CKeyEd25519& secretEdRet)
{
    std::array<char, 32> edSeed = ConvertSecureVector32ToArray(seed.getKeyData());
    CKeyEd25519 childKey(edSeed);

    AddDHTKey(childKey, childKey.GetPubKey()); //TODO

    secretEdRet = childKey; //return Ed25519 key

} //DeriveEd25519ChildKey

bool CWallet::DeriveChildStealthKey(const CKey& key)
{
    CKey spendKey, scanKey;
    if (!key.DeriveChildKey(spendKey))
        return false;

    if (!spendKey.DeriveChildKey(scanKey))
        return false;

    if (!AddKeyPubKey(spendKey, spendKey.GetPubKey()))
        return false;

    if (!AddKeyPubKey(scanKey, scanKey.GetPubKey()))
        return false;

    CStealthAddress sxAddr(scanKey, spendKey);
    return AddStealthAddress(sxAddr);
}

void CWallet::DeriveNewChildKey(const CKeyMetadata& metadata, CKey& secretRet, uint32_t nAccountIndex, bool fInternal)
{
    CHDChain hdChainTmp;

    if (!GetHDChain(hdChainTmp)) {
        throw std::runtime_error(std::string(__func__) + ": GetHDChain failed");
    }

    if (!DecryptHDChain(hdChainTmp))
        throw std::runtime_error(std::string(__func__) + ": DecryptHDChainSeed failed");
    // make sure seed matches this chain
    if (hdChainTmp.GetID() != hdChainTmp.GetSeedHash())
        throw std::runtime_error(std::string(__func__) + ": Wrong HD chain!");

    CHDAccount acc;
    if (!hdChainTmp.GetAccount(nAccountIndex, acc))
        throw std::runtime_error(std::string(__func__) + ": Wrong HD account!");

    // derive child key at next index, skip keys already known to the wallet
    CExtKey childKey;
    uint32_t nChildIndex = fInternal ? acc.nInternalChainCounter : acc.nExternalChainCounter;
    do {
        hdChainTmp.DeriveChildExtKey(nAccountIndex, fInternal, nChildIndex, childKey);
        // increment childkey index
        nChildIndex++;
    } while (HaveKey(childKey.key.GetPubKey().GetID()));
    secretRet = childKey.key;

    CKeyEd25519 secretEdRet;
    DeriveEd25519ChildKey(secretRet,secretEdRet); //Derive Ed25519 key
    DeriveChildStealthKey(secretRet);

    CPubKey pubkey = secretRet.GetPubKey();
    assert(secretRet.VerifyPubKey(pubkey));

    // store metadata
    mapKeyMetadata[pubkey.GetID()] = metadata;
    UpdateTimeFirstKey(metadata.nCreateTime);

    // update the chain model in the database
    CHDChain hdChainCurrent;
    GetHDChain(hdChainCurrent);

    if (fInternal) {
        acc.nInternalChainCounter = nChildIndex;
    } else {
        acc.nExternalChainCounter = nChildIndex;
    }

    if (!hdChainCurrent.SetAccount(nAccountIndex, acc))
        throw std::runtime_error(std::string(__func__) + ": SetAccount failed");

    if (IsCrypted()) {
        if (!SetCryptedHDChain(hdChainCurrent, false))
            throw std::runtime_error(std::string(__func__) + ": SetCryptedHDChain failed");
    } else {
        if (!SetHDChain(hdChainCurrent, false))
            throw std::runtime_error(std::string(__func__) + ": SetHDChain failed");
    }

    if (!AddHDPubKey(childKey.Neuter(), fInternal))
        throw std::runtime_error(std::string(__func__) + ": AddHDPubKey failed");
}

void CWallet::DeriveNewChildKeyBIP44BychainChildKey(CExtKey& chainChildKey, CKey& secret, bool internal, uint32_t* nInternalChainCounter, uint32_t* nExternalChainCounter)
{
    CExtKey childKey;              //key at m/0'/0'/<n>'
     // derive child key at next index, skip keys already known to the wallet
    
    // always derive hardened keys
    // childIndex | BIP32_HARDENED_KEY_LIMIT = derive childIndex in hardened child-index-range
    // example: 1 | BIP32_HARDENED_KEY_LIMIT == 0x80000001 == 2147483649
    if (internal) {
        chainChildKey.Derive(childKey, *nInternalChainCounter);
        //metadata.hdKeypath = "m/44'/0'/0'/1/" + std::to_string(hdChain.nInternalChainCounter);
        (*nInternalChainCounter)++;
    }
    else {
        chainChildKey.Derive(childKey, *nExternalChainCounter);
        //metadata.hdKeypath = "m/44'/0'/0'/0/" + std::to_string(hdChain.nExternalChainCounter);
        (*nExternalChainCounter)++;
    }
    secret = childKey.key;
}

bool CWallet::GetPubKey(const CKeyID& address, CPubKey& vchPubKeyOut) const
{
    LOCK(cs_wallet);
    std::map<CKeyID, CHDPubKey>::const_iterator mi = mapHdPubKeys.find(address);
    if (mi != mapHdPubKeys.end()) {
        const CHDPubKey& hdPubKey = (*mi).second;
        vchPubKeyOut = hdPubKey.extPubKey.pubkey;
        return true;
    } else
        return CCryptoKeyStore::GetPubKey(address, vchPubKeyOut);
}

bool CWallet::GetKey(const CKeyID& address, CKey& keyOut) const
{
    LOCK(cs_wallet);
    std::map<CKeyID, CHDPubKey>::const_iterator mi = mapHdPubKeys.find(address);
    if (mi != mapHdPubKeys.end()) {
        // if the key has been found in mapHdPubKeys, derive it on the fly
        const CHDPubKey& hdPubKey = (*mi).second;
        CHDChain hdChainCurrent;
        if (!GetHDChain(hdChainCurrent))
            throw std::runtime_error(std::string(__func__) + ": GetHDChain failed");
        if (!DecryptHDChain(hdChainCurrent))
            throw std::runtime_error(std::string(__func__) + ": DecryptHDChainSeed failed");
        // make sure seed matches this chain
        if (hdChainCurrent.GetID() != hdChainCurrent.GetSeedHash())
            throw std::runtime_error(std::string(__func__) + ": Wrong HD chain!");

        CExtKey extkey;
        hdChainCurrent.DeriveChildExtKey(hdPubKey.nAccountIndex, hdPubKey.nChangeIndex != 0, hdPubKey.extPubKey.nChild, extkey);
        keyOut = extkey.key;

        return true;
    } else {
        return CCryptoKeyStore::GetKey(address, keyOut);
    }
}

bool CWallet::GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const
{
    LOCK(cs_wallet);
    return CCryptoKeyStore::GetDHTKey(address, keyOut);
}

bool CWallet::HaveDHTKey(const CKeyID &address) const
{
    LOCK(cs_wallet);
    if (mapHdPubKeys.count(address) > 0)
        return true;
    return CCryptoKeyStore::HaveDHTKey(address);
}

bool CWallet::HaveKey(const CKeyID &address) const
{
    LOCK(cs_wallet);
    if (mapHdPubKeys.count(address) > 0)
        return true;
    return CCryptoKeyStore::HaveKey(address);
}

bool CWallet::LoadHDPubKey(const CHDPubKey& hdPubKey)
{
    AssertLockHeld(cs_wallet);

    mapHdPubKeys[hdPubKey.extPubKey.pubkey.GetID()] = hdPubKey;
    return true;
}

bool CWallet::AddHDPubKey(const CExtPubKey& extPubKey, bool fInternal)
{
    AssertLockHeld(cs_wallet);

    CHDChain hdChainCurrent;
    GetHDChain(hdChainCurrent);

    CHDPubKey hdPubKey;
    hdPubKey.extPubKey = extPubKey;
    hdPubKey.hdchainID = hdChainCurrent.GetID();
    hdPubKey.nChangeIndex = fInternal ? 1 : 0;
    mapHdPubKeys[extPubKey.pubkey.GetID()] = hdPubKey;

    // check if we need to remove from watch-only
    CScript script;
    script = GetScriptForDestination(extPubKey.pubkey.GetID());
    if (HaveWatchOnly(script))
        RemoveWatchOnly(script);
    script = GetScriptForRawPubKey(extPubKey.pubkey);
    if (HaveWatchOnly(script))
        RemoveWatchOnly(script);

    if (!fFileBacked)
        return true;

    return CWalletDB(strWalletFile).WriteHDPubKey(hdPubKey, mapKeyMetadata[extPubKey.pubkey.GetID()]);
}

bool CWallet::AddKeyPubKey(const CKey& secret, const CPubKey& pubkey)
{
    AssertLockHeld(cs_wallet); // mapKeyMetadata
    if (!CCryptoKeyStore::AddKeyPubKey(secret, pubkey))
        return false;

    // check if we need to remove from watch-only
    CScript script;
    script = GetScriptForDestination(pubkey.GetID());
    if (HaveWatchOnly(script))
        RemoveWatchOnly(script);
    script = GetScriptForRawPubKey(pubkey);
    if (HaveWatchOnly(script))
        RemoveWatchOnly(script);

    if (!fFileBacked)
        return true;
    if (!IsCrypted()) {
        return CWalletDB(strWalletFile).WriteKey(pubkey, secret.GetPrivKey(), mapKeyMetadata[pubkey.GetID()]);
    }
    return true;
}

bool CWallet::AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& pubkey)
{
    AssertLockHeld(cs_wallet); // mapKeyMetadata
    if (!CCryptoKeyStore::AddDHTKey(key, pubkey)) {
        return false;
    }

    if (!fFileBacked)
        return true;

    if (!IsCrypted()) {
        CKeyID keyID(Hash160(pubkey.begin(), pubkey.end()));
        return CWalletDB(strWalletFile).WriteDHTKey(key, pubkey, mapKeyMetadata[keyID]);
    }
    LogPrint("dht", "CWallet::AddDHTKey \npubkey = %s, \nprivkey = %s, \nprivseed = %s\n", 
                    key.GetPubKeyString(), key.GetPrivKeyString(), key.GetPrivSeedString());
    return true;
}

bool CWallet::AddCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret)
{
    if (!CCryptoKeyStore::AddCryptedDHTKey(vchPubKey, vchCryptedSecret)) {
        LogPrint("dht", "CWallet::AddCryptedDHTKey AddCryptedDHTKey failed.\n");
        return false;
    }
    if (!fFileBacked)
        return true;
    {
        LOCK(cs_wallet);
        CKeyID keyID(Hash160(vchPubKey.begin(), vchPubKey.end()));
        if (pwalletdbEncryption) {
            return pwalletdbEncryption->WriteCryptedDHTKey(vchPubKey, vchCryptedSecret, mapKeyMetadata[keyID]);
        }
        else {
            return CWalletDB(strWalletFile).WriteCryptedDHTKey(vchPubKey, vchCryptedSecret, mapKeyMetadata[keyID]);
        }
    }
    return false;
}

bool CWallet::AddCryptedKey(const CPubKey& vchPubKey,
    const std::vector<unsigned char>& vchCryptedSecret)

{
    if (!CCryptoKeyStore::AddCryptedKey(vchPubKey, vchCryptedSecret))
        return false;
    if (!fFileBacked)
        return true;
    {
        LOCK(cs_wallet);
        if (pwalletdbEncryption)
            return pwalletdbEncryption->WriteCryptedKey(vchPubKey,
                vchCryptedSecret,
                mapKeyMetadata[vchPubKey.GetID()]);
        else
            return CWalletDB(strWalletFile).WriteCryptedKey(vchPubKey, vchCryptedSecret, mapKeyMetadata[vchPubKey.GetID()]);
    }
    return false;
}

bool CWallet::LoadKeyMetadata(const CTxDestination& keyID, const CKeyMetadata& meta)
{
    AssertLockHeld(cs_wallet); // mapKeyMetadata
    UpdateTimeFirstKey(meta.nCreateTime);
    mapKeyMetadata[keyID] = meta;
    return true;
}

bool CWallet::LoadCryptedKey(const CPubKey& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret)
{
    return CCryptoKeyStore::AddCryptedKey(vchPubKey, vchCryptedSecret);
}

bool CWallet::LoadCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret)
{
    return CCryptoKeyStore::AddCryptedDHTKey(vchPubKey, vchCryptedSecret);
}

void CWallet::UpdateTimeFirstKey(int64_t nCreateTime)
{
    AssertLockHeld(cs_wallet);
    if (nCreateTime <= 1) {
        // Cannot determine birthday information, so set the wallet birthday to
        // the beginning of time.
        nTimeFirstKey = 1;
    } else if (!nTimeFirstKey || nCreateTime < nTimeFirstKey) {
        nTimeFirstKey = nCreateTime;
    }
}

bool CWallet::AddCScript(const CScript& redeemScript)
{
    if (!CCryptoKeyStore::AddCScript(redeemScript))
        return false;
    if (!fFileBacked)
        return true;
    return CWalletDB(strWalletFile).WriteCScript(Hash160(redeemScript), redeemScript);
}

bool CWallet::LoadCScript(const CScript& redeemScript)
{
    /* A sanity check was added in pull #3843 to avoid adding redeemScripts
     * that never can be redeemed. However, old wallets may still contain
     * these. Do not add them to the wallet and warn. */
    if (redeemScript.size() > MAX_SCRIPT_ELEMENT_SIZE) {
        std::string strAddr = CDynamicAddress(CScriptID(redeemScript)).ToString();
        LogPrintf("%s: Warning: This wallet contains a redeemScript of size %i which exceeds maximum size %i thus can never be redeemed. Do not use address %s.\n",
            __func__, redeemScript.size(), MAX_SCRIPT_ELEMENT_SIZE, strAddr);
        return true;
    }

    return CCryptoKeyStore::AddCScript(redeemScript);
}

bool CWallet::AddWatchOnly(const CScript& dest)
{
    if (!CCryptoKeyStore::AddWatchOnly(dest))
        return false;
    const CKeyMetadata& meta = mapKeyMetadata[CScriptID(dest)];
    UpdateTimeFirstKey(meta.nCreateTime);
    NotifyWatchonlyChanged(true);
    if (!fFileBacked)
        return true;
    return CWalletDB(strWalletFile).WriteWatchOnly(dest, meta);
}

bool CWallet::AddWatchOnly(const CScript& dest, int64_t nCreateTime)
{
    mapKeyMetadata[CScriptID(dest)].nCreateTime = nCreateTime;
    return AddWatchOnly(dest);
}

bool CWallet::RemoveWatchOnly(const CScript& dest)
{
    AssertLockHeld(cs_wallet);
    if (!CCryptoKeyStore::RemoveWatchOnly(dest))
        return false;
    if (!HaveWatchOnly())
        NotifyWatchonlyChanged(false);
    if (fFileBacked)
        if (!CWalletDB(strWalletFile).EraseWatchOnly(dest))
            return false;

    return true;
}

bool CWallet::LoadWatchOnly(const CScript& dest)
{
    return CCryptoKeyStore::AddWatchOnly(dest);
}

bool CWallet::Unlock(const SecureString& strWalletPassphrase, bool fForMixingOnly)
{
    SecureString strWalletPassphraseFinal;

    if (!IsLocked()) // was already fully unlocked, not only for mixing
        return true;
    // Verify KeePassIntegration
    if (strWalletPassphrase == "keepass" && gArgs.GetBoolArg("-keepass", false)) {
        try {
            strWalletPassphraseFinal = keePassInt.retrievePassphrase();
        } catch (const std::exception& e) {
            LogPrintf("CWallet::Unlock could not retrieve passphrase from KeePass: Error: %s\n", e.what());
            return false;
        }
    } else {
        strWalletPassphraseFinal = strWalletPassphrase;
    }

    CCrypter crypter;
    CKeyingMaterial vMasterKey;

    {
        LOCK(cs_wallet);
        for (const MasterKeyMap::value_type& pMasterKey : mapMasterKeys) {
            if (!crypter.SetKeyFromPassphrase(strWalletPassphraseFinal, pMasterKey.second.vchSalt, pMasterKey.second.nDeriveIterations, pMasterKey.second.nDerivationMethod))
                return false;
            if (!crypter.Decrypt(pMasterKey.second.vchCryptedKey, vMasterKey))
                continue; // try another master key
            if (CCryptoKeyStore::Unlock(vMasterKey, fForMixingOnly)) {
                if (fNeedToUpgradeWallet) {
                    if (SyncEdKeyPool()) {
                        SetMinVersion(FEATURE_HD);
                        LogPrintf("%s - Upgraded wallet\n", __func__);
                    }
                }
                if (nWalletBackups == -2) {
                    TopUpKeyPoolCombo();
                    LogPrintf("Keypool replenished, re-initializing automatic backups.\n");
                    nWalletBackups = gArgs.GetArg("-createwalletbackups", 10);
                }
                if (!fForMixingOnly) {
                    RunProcessStealthQueue(); // Process stealth transactions in queue
                    ProcessLinkQueue(); // Process encrypted links in queue
                }
                if (!fNeedToUpgradeWallet) return true;
            }
        }
    }

    if (fNeedToUpgradeWallet) {
        {
            LOCK2(cs_main, pwalletMain->cs_wallet);
            fNeedToUpgradeWallet = false;
            fNeedToUpdateKeyPools = true;
            fNeedToUpdateLinks = true;
            LogPrintf("%s - Rescanning transasctions\n", __func__);
            pwalletMain->UpdateTimeFirstKey(1);
            pwalletMain->ScanForWalletTransactions(chainActive.Genesis(), true);
            LogPrintf("%s - Rescanning transasctions DONE\n", __func__);        
        }
        return true;
    }

    return false;
}

bool CWallet::ChangeWalletPassphrase(const SecureString& strOldWalletPassphrase, const SecureString& strNewWalletPassphrase)
{
    bool fWasLocked = IsLocked(true);
    bool bUseKeePass = false;

    SecureString strOldWalletPassphraseFinal;

    // Verify KeePassIntegration
    if (strOldWalletPassphrase == "keepass" && gArgs.GetBoolArg("-keepass", false)) {
        bUseKeePass = true;
        try {
            strOldWalletPassphraseFinal = keePassInt.retrievePassphrase();
        } catch (const std::exception& e) {
            LogPrintf("CWallet::ChangeWalletPassphrase -- could not retrieve passphrase from KeePass: Error: %s\n", e.what());
            return false;
        }
    } else {
        strOldWalletPassphraseFinal = strOldWalletPassphrase;
    }

    {
        LOCK(cs_wallet);
        Lock();

        CCrypter crypter;
        CKeyingMaterial vMasterKey;
        for (MasterKeyMap::value_type& pMasterKey : mapMasterKeys) {
            if (!crypter.SetKeyFromPassphrase(strOldWalletPassphraseFinal, pMasterKey.second.vchSalt, pMasterKey.second.nDeriveIterations, pMasterKey.second.nDerivationMethod))
                return false;
            if (!crypter.Decrypt(pMasterKey.second.vchCryptedKey, vMasterKey))
                return false;
            if (CCryptoKeyStore::Unlock(vMasterKey)) {
                int64_t nStartTime = GetTimeMillis();
                crypter.SetKeyFromPassphrase(strNewWalletPassphrase, pMasterKey.second.vchSalt, pMasterKey.second.nDeriveIterations, pMasterKey.second.nDerivationMethod);
                pMasterKey.second.nDeriveIterations = pMasterKey.second.nDeriveIterations * (100 / ((double)(GetTimeMillis() - nStartTime)));

                nStartTime = GetTimeMillis();
                crypter.SetKeyFromPassphrase(strNewWalletPassphrase, pMasterKey.second.vchSalt, pMasterKey.second.nDeriveIterations, pMasterKey.second.nDerivationMethod);
                pMasterKey.second.nDeriveIterations = (pMasterKey.second.nDeriveIterations + pMasterKey.second.nDeriveIterations * 100 / ((double)(GetTimeMillis() - nStartTime))) / 2;

                if (pMasterKey.second.nDeriveIterations < 25000)
                    pMasterKey.second.nDeriveIterations = 25000;

                LogPrintf("Wallet passphrase changed to an nDeriveIterations of %i\n", pMasterKey.second.nDeriveIterations);

                if (!crypter.SetKeyFromPassphrase(strNewWalletPassphrase, pMasterKey.second.vchSalt, pMasterKey.second.nDeriveIterations, pMasterKey.second.nDerivationMethod))
                    return false;
                if (!crypter.Encrypt(vMasterKey, pMasterKey.second.vchCryptedKey))
                    return false;
                CWalletDB(strWalletFile).WriteMasterKey(pMasterKey.first, pMasterKey.second);
                if (fWasLocked)
                    Lock();

                // Update KeePass if necessary
                if (bUseKeePass) {
                    LogPrintf("CWallet::ChangeWalletPassphrase -- Updating KeePass with new passphrase");
                    try {
                        keePassInt.updatePassphrase(strNewWalletPassphrase);
                    } catch (const std::exception& e) {
                        LogPrintf("CWallet::ChangeWalletPassphrase -- could not update passphrase in KeePass: Error: %s\n", e.what());
                        return false;
                    }
                }

                return true;
            }
        }
    }

    return false;
}

void CWallet::SetBestChain(const CBlockLocator& loc)
{
    CWalletDB walletdb(strWalletFile);
    walletdb.WriteBestBlock(loc);
}

bool CWallet::SetMinVersion(enum WalletFeature nVersion, CWalletDB* pwalletdbIn, bool fExplicit)
{
    LOCK(cs_wallet); // nWalletVersion
    if (nWalletVersion >= nVersion)
        return true;

    // when doing an explicit upgrade, if we pass the max version permitted, upgrade all the way
    if (fExplicit && nVersion > nWalletMaxVersion)
        nVersion = FEATURE_LATEST;

    nWalletVersion = nVersion;

    if (nVersion > nWalletMaxVersion)
        nWalletMaxVersion = nVersion;

    if (fFileBacked) {
        CWalletDB* pwalletdb = pwalletdbIn ? pwalletdbIn : new CWalletDB(strWalletFile);
        if (nWalletVersion > 40000)
            pwalletdb->WriteMinVersion(nWalletVersion);
        if (!pwalletdbIn)
            delete pwalletdb;
    }

    return true;
}

bool CWallet::SetMaxVersion(int nVersion)
{
    LOCK(cs_wallet); // nWalletVersion, nWalletMaxVersion
    // cannot downgrade below current version
    if (nWalletVersion > nVersion)
        return false;

    nWalletMaxVersion = nVersion;

    return true;
}

std::set<uint256> CWallet::GetConflicts(const uint256& txid) const
{
    std::set<uint256> result;
    AssertLockHeld(cs_wallet);

    std::map<uint256, CWalletTx>::const_iterator it = mapWallet.find(txid);
    if (it == mapWallet.end())
        return result;
    const CWalletTx& wtx = it->second;

    std::pair<TxSpends::const_iterator, TxSpends::const_iterator> range;

    for (const CTxIn& txin : wtx.tx->vin) {
        if (mapTxSpends.count(txin.prevout) <= 1)
            continue; // No conflict if zero or one spends
        range = mapTxSpends.equal_range(txin.prevout);
        for (TxSpends::const_iterator _it = range.first; _it != range.second; ++_it)
            result.insert(_it->second);
    }
    return result;
}

bool CWallet::HasWalletSpend(const uint256& txid) const
{
    AssertLockHeld(cs_wallet);
    auto iter = mapTxSpends.lower_bound(COutPoint(txid, 0));
    return (iter != mapTxSpends.end() && iter->first.hash == txid);
}

void CWallet::Flush(bool shutdown)
{
    bitdb.Flush(shutdown);
}

bool CWallet::Verify()
{
    if (gArgs.GetBoolArg("-disablewallet", DEFAULT_DISABLE_WALLET))
        return true;

    LogPrintf("Using BerkeleyDB version %s\n", DbEnv::version(0, 0, 0));
    std::string walletFile = gArgs.GetArg("-wallet", DEFAULT_WALLET_DAT);

    LogPrintf("Using wallet %s\n", walletFile);
    uiInterface.InitMessage(_("Verifying wallet..."));

    // Wallet file must be a plain filename without a directory
    if (walletFile != boost::filesystem::basename(walletFile) + boost::filesystem::extension(walletFile))
        return InitError(strprintf(_("Wallet %s resides outside data directory %s"), walletFile, GetDataDir().string()));

    if (!bitdb.Open(GetDataDir())) {
        // try moving the database env out of the way
        boost::filesystem::path pathDatabase = GetDataDir() / "database";
        boost::filesystem::path pathDatabaseBak = GetDataDir() / strprintf("database.%d.bak", GetTime());
        try {
            boost::filesystem::rename(pathDatabase, pathDatabaseBak);
            LogPrintf("Moved old %s to %s. Retrying.\n", pathDatabase.string(), pathDatabaseBak.string());
        } catch (const boost::filesystem::filesystem_error&) {
            // failure is ok (well, not really, but it's not worse than what we started with)
        }

        // try again
        if (!bitdb.Open(GetDataDir())) {
            // if it still fails, it probably means we can't even create the database env
            return InitError(strprintf(_("Error initializing wallet database environment %s!"), GetDataDir()));
        }
    }

    if (gArgs.GetBoolArg("-salvagewallet", false)) {
        // Recover readable keypairs:
        if (!CWalletDB::Recover(bitdb, walletFile, true))
            return false;
    }

    if (boost::filesystem::exists(GetDataDir() / walletFile)) {
        CDBEnv::VerifyResult r = bitdb.Verify(walletFile, CWalletDB::Recover);
        if (r == CDBEnv::RECOVER_OK) {
            InitWarning(strprintf(_("Warning: Wallet file corrupt, data salvaged!"
                                    " Original %s saved as %s in %s; if"
                                    " your balance or transactions are incorrect you should"
                                    " restore from a backup."),
                walletFile, "wallet.{timestamp}.bak", GetDataDir()));
        }
        if (r == CDBEnv::RECOVER_FAIL)
            return InitError(strprintf(_("%s corrupt, salvage failed"), walletFile));
    }

    return true;
}


void CWallet::SyncMetaData(std::pair<TxSpends::iterator, TxSpends::iterator> range)
{
    // We want all the wallet transactions in range to have the same metadata as
    // the oldest (smallest nOrderPos).
    // So: find smallest nOrderPos:

    int nMinOrderPos = std::numeric_limits<int>::max();
    const CWalletTx* copyFrom = nullptr;
    for (TxSpends::iterator it = range.first; it != range.second; ++it) {
        const uint256& hash = it->second;
        int n = mapWallet[hash].nOrderPos;
        if (n < nMinOrderPos) {
            nMinOrderPos = n;
            copyFrom = &mapWallet[hash];
        }
    }
    // Now copy data from copyFrom to rest:
    for (TxSpends::iterator it = range.first; it != range.second; ++it) {
        const uint256& hash = it->second;
        CWalletTx* copyTo = &mapWallet[hash];
        if (copyFrom == copyTo)
            continue;
        if (!copyFrom->IsEquivalentTo(*copyTo))
            continue;
        copyTo->mapValue = copyFrom->mapValue;
        copyTo->vOrderForm = copyFrom->vOrderForm;
        // fTimeReceivedIsTxTime not copied on purpose
        // nTimeReceived not copied on purpose
        copyTo->nTimeSmart = copyFrom->nTimeSmart;
        copyTo->fFromMe = copyFrom->fFromMe;
        copyTo->strFromAccount = copyFrom->strFromAccount;
        // nOrderPos not copied on purpose
        // cached members not copied on purpose
    }
}

/**
 * Outpoint is spent if any non-conflicted transaction
 * spends it:
 */
bool CWallet::IsSpent(const uint256& hash, unsigned int n) const
{
    const COutPoint outpoint(hash, n);
    std::pair<TxSpends::const_iterator, TxSpends::const_iterator> range;
    range = mapTxSpends.equal_range(outpoint);

    for (TxSpends::const_iterator it = range.first; it != range.second; ++it) {
        const uint256& wtxid = it->second;
        std::map<uint256, CWalletTx>::const_iterator mit = mapWallet.find(wtxid);
        if (mit != mapWallet.end()) {
            int depth = mit->second.GetDepthInMainChain();
            if (depth > 0 || (depth == 0 && !mit->second.isAbandoned()))
                return true; // Spent
        }
    }
    return false;
}

void CWallet::AddToSpends(const COutPoint& outpoint, const uint256& wtxid)
{
    mapTxSpends.insert(std::make_pair(outpoint, wtxid));
    setWalletUTXO.erase(outpoint);

    std::pair<TxSpends::iterator, TxSpends::iterator> range;
    range = mapTxSpends.equal_range(outpoint);
    SyncMetaData(range);
}


void CWallet::AddToSpends(const uint256& wtxid)
{
    assert(mapWallet.count(wtxid));
    CWalletTx& thisTx = mapWallet[wtxid];
    if (thisTx.IsCoinBase()) // Coinbases don't spend anything!
        return;

    for (const CTxIn& txin : thisTx.tx->vin)
        AddToSpends(txin.prevout, wtxid);
}

bool CWallet::EncryptWallet(const SecureString& strWalletPassphrase)
{
    if (IsCrypted())
        return false;

    CKeyingMaterial vMasterKey;

    vMasterKey.resize(WALLET_CRYPTO_KEY_SIZE);
    GetStrongRandBytes(&vMasterKey[0], WALLET_CRYPTO_KEY_SIZE);

    CMasterKey kMasterKey;

    kMasterKey.vchSalt.resize(WALLET_CRYPTO_SALT_SIZE);
    GetStrongRandBytes(&kMasterKey.vchSalt[0], WALLET_CRYPTO_SALT_SIZE);

    CCrypter crypter;
    int64_t nStartTime = GetTimeMillis();
    crypter.SetKeyFromPassphrase(strWalletPassphrase, kMasterKey.vchSalt, 25000, kMasterKey.nDerivationMethod);
    kMasterKey.nDeriveIterations = 2500000 / ((double)(GetTimeMillis() - nStartTime));

    nStartTime = GetTimeMillis();
    crypter.SetKeyFromPassphrase(strWalletPassphrase, kMasterKey.vchSalt, kMasterKey.nDeriveIterations, kMasterKey.nDerivationMethod);
    kMasterKey.nDeriveIterations = (kMasterKey.nDeriveIterations + kMasterKey.nDeriveIterations * 100 / ((double)(GetTimeMillis() - nStartTime))) / 2;

    if (kMasterKey.nDeriveIterations < 25000)
        kMasterKey.nDeriveIterations = 25000;

    LogPrintf("Encrypting Wallet with an nDeriveIterations of %i\n", kMasterKey.nDeriveIterations);

    if (!crypter.SetKeyFromPassphrase(strWalletPassphrase, kMasterKey.vchSalt, kMasterKey.nDeriveIterations, kMasterKey.nDerivationMethod))
        return false;
    if (!crypter.Encrypt(vMasterKey, kMasterKey.vchCryptedKey))
        return false;

    {
        LOCK(cs_wallet);
        mapMasterKeys[++nMasterKeyMaxID] = kMasterKey;
        if (fFileBacked) {
            assert(!pwalletdbEncryption);
            pwalletdbEncryption = new CWalletDB(strWalletFile);
            if (!pwalletdbEncryption->TxnBegin()) {
                delete pwalletdbEncryption;
                pwalletdbEncryption = nullptr;
                return false;
            }
            pwalletdbEncryption->WriteMasterKey(nMasterKeyMaxID, kMasterKey);
        }

        // must get current HD chain before EncryptKeys
        CHDChain hdChainCurrent;
        GetHDChain(hdChainCurrent);

        if (!EncryptKeys(vMasterKey)) {
            if (fFileBacked) {
                pwalletdbEncryption->TxnAbort();
                delete pwalletdbEncryption;
            }
            // We now probably have half of our keys encrypted in memory, and half not...
            // die and let the user reload the unencrypted wallet.
            assert(false);
        }

        if (!hdChainCurrent.IsNull()) {
            assert(EncryptHDChain(vMasterKey));

            CHDChain hdChainCrypted;
            assert(GetHDChain(hdChainCrypted));

            DBG(
                printf("EncryptWallet -- current seed: '%s'\n", HexStr(hdChainCurrent.GetSeed()).c_str());
                printf("EncryptWallet -- crypted seed: '%s'\n", HexStr(hdChainCrypted.GetSeed()).c_str()););

            // ids should match, seed hashes should not
            assert(hdChainCurrent.GetID() == hdChainCrypted.GetID());
            assert(hdChainCurrent.GetSeedHash() != hdChainCrypted.GetSeedHash());

            assert(SetCryptedHDChain(hdChainCrypted, false));
        }

        // Encryption was introduced in version 0.4.0
        SetMinVersion(FEATURE_WALLETCRYPT, pwalletdbEncryption, true);

        if (fFileBacked) {
            if (!pwalletdbEncryption->TxnCommit()) {
                delete pwalletdbEncryption;
                // We now have keys encrypted in memory, but not on disk...
                // die to avoid confusion and let the user reload the unencrypted wallet.
                assert(false);
            }

            delete pwalletdbEncryption;
            pwalletdbEncryption = nullptr;
        }

        Lock();
        Unlock(strWalletPassphrase);

        // if we are not using HD, generate new keypool
        if (IsHDEnabled()) {
            TopUpKeyPoolCombo();
        } else {
            NewKeyPool();
            NewEdKeyPool();
        }

        Lock();

        // Need to completely rewrite the wallet file; if we don't, bdb might keep
        // bits of the unencrypted private key in slack space in the database file.
        CDB::Rewrite(strWalletFile);

        // Update KeePass if necessary
        if (gArgs.GetBoolArg("-keepass", false)) {
            LogPrintf("CWallet::EncryptWallet -- Updating KeePass with new passphrase");
            try {
                keePassInt.updatePassphrase(strWalletPassphrase);
            } catch (const std::exception& e) {
                LogPrintf("CWallet::EncryptWallet -- could not update passphrase in KeePass: Error: %s\n", e.what());
            }
        }
    }
    NotifyStatusChanged(this);

    return true;
}

DBErrors CWallet::ReorderTransactions()
{
    LOCK(cs_wallet);
    CWalletDB walletdb(strWalletFile);

    // Old wallets didn't have any defined order for transactions
    // Probably a bad idea to change the output of this

    // First: get all CWalletTx and CAccountingEntry into a sorted-by-time multimap.
    typedef std::pair<CWalletTx*, CAccountingEntry*> TxPair;
    typedef std::multimap<int64_t, TxPair> TxItems;
    TxItems txByTime;

    for (std::map<uint256, CWalletTx>::iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
        CWalletTx* wtx = &((*it).second);
        txByTime.insert(make_pair(wtx->nTimeReceived, TxPair(wtx, (CAccountingEntry*)0)));
    }
    std::list<CAccountingEntry> acentries;
    walletdb.ListAccountCreditDebit("", acentries);
    for (CAccountingEntry& entry : acentries) {
        txByTime.insert(make_pair(entry.nTime, TxPair((CWalletTx*)0, &entry)));
    }

    nOrderPosNext = 0;
    std::vector<int64_t> nOrderPosOffsets;
    for (TxItems::iterator it = txByTime.begin(); it != txByTime.end(); ++it) {
        CWalletTx* const pwtx = (*it).second.first;
        CAccountingEntry* const pacentry = (*it).second.second;
        int64_t& nOrderPos = (pwtx != 0) ? pwtx->nOrderPos : pacentry->nOrderPos;

        if (nOrderPos == -1) {
            nOrderPos = nOrderPosNext++;
            nOrderPosOffsets.push_back(nOrderPos);

            if (pwtx) {
                if (!walletdb.WriteTx(*pwtx))
                    return DB_LOAD_FAIL;
            } else if (!walletdb.WriteAccountingEntry(pacentry->nEntryNo, *pacentry))
                return DB_LOAD_FAIL;
        } else {
            int64_t nOrderPosOff = 0;
            for (const int64_t& nOffsetStart : nOrderPosOffsets) {
                if (nOrderPos >= nOffsetStart)
                    ++nOrderPosOff;
            }
            nOrderPos += nOrderPosOff;
            nOrderPosNext = std::max(nOrderPosNext, nOrderPos + 1);

            if (!nOrderPosOff)
                continue;

            // Since we're changing the order, write it back
            if (pwtx) {
                if (!walletdb.WriteTx(*pwtx))
                    return DB_LOAD_FAIL;
            } else if (!walletdb.WriteAccountingEntry(pacentry->nEntryNo, *pacentry))
                return DB_LOAD_FAIL;
        }
    }
    walletdb.WriteOrderPosNext(nOrderPosNext);

    return DB_LOAD_OK;
}

int64_t CWallet::IncOrderPosNext(CWalletDB* pwalletdb)
{
    AssertLockHeld(cs_wallet); // nOrderPosNext
    int64_t nRet = nOrderPosNext++;
    if (pwalletdb) {
        pwalletdb->WriteOrderPosNext(nOrderPosNext);
    } else {
        CWalletDB(strWalletFile).WriteOrderPosNext(nOrderPosNext);
    }
    return nRet;
}

bool CWallet::AccountMove(std::string strFrom, std::string strTo, CAmount nAmount, std::string strComment)
{
    CWalletDB walletdb(strWalletFile);
    if (!walletdb.TxnBegin())
        return false;

    int64_t nNow = GetAdjustedTime();

    // Debit
    CAccountingEntry debit;
    debit.nOrderPos = IncOrderPosNext(&walletdb);
    debit.strAccount = strFrom;
    debit.nCreditDebit = -nAmount;
    debit.nTime = nNow;
    debit.strOtherAccount = strTo;
    debit.strComment = strComment;
    AddAccountingEntry(debit, &walletdb);

    // Credit
    CAccountingEntry credit;
    credit.nOrderPos = IncOrderPosNext(&walletdb);
    credit.strAccount = strTo;
    credit.nCreditDebit = nAmount;
    credit.nTime = nNow;
    credit.strOtherAccount = strFrom;
    credit.strComment = strComment;
    AddAccountingEntry(credit, &walletdb);

    if (!walletdb.TxnCommit())
        return false;

    return true;
}

bool CWallet::GetAccountPubkey(CPubKey& pubKey, std::string strAccount, bool bForceNew)
{
    CWalletDB walletdb(strWalletFile);

    CAccount account;
    walletdb.ReadAccount(strAccount, account);

    if (!bForceNew) {
        if (!account.vchPubKey.IsValid())
            bForceNew = true;
        else {
            // Check if the current key has been used
            CScript scriptPubKey = GetScriptForDestination(account.vchPubKey.GetID());
            for (std::map<uint256, CWalletTx>::iterator it = mapWallet.begin();
                 it != mapWallet.end() && account.vchPubKey.IsValid();
                 ++it)
                for (const CTxOut& txout : (*it).second.tx->vout)
                    if (txout.scriptPubKey == scriptPubKey) {
                        bForceNew = true;
                        break;
                    }
        }
    }

    // Generate a new key
    if (bForceNew) {
        std::vector<unsigned char> newEdKey;
        if (!GetKeysFromPool(account.vchPubKey, newEdKey, false))
            return false;

        SetAddressBook(account.vchPubKey.GetID(), strAccount, "receive");
        walletdb.WriteAccount(strAccount, account);
    }

    pubKey = account.vchPubKey;

    return true;
}

void CWallet::MarkDirty()
{
    {
        LOCK(cs_wallet);
        for (std::pair<const uint256, CWalletTx>& item : mapWallet)
            item.second.MarkDirty();
    }

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;
}

bool CWallet::MarkReplaced(const uint256& originalHash, const uint256& newHash)
{
    LOCK(cs_wallet);

    auto mi = mapWallet.find(originalHash);

    // There is a bug if MarkReplaced is not called on an existing wallet transaction.
    assert(mi != mapWallet.end());

    CWalletTx& wtx = (*mi).second;

    // Ensure for now that we're not overwriting data
    assert(wtx.mapValue.count("replaced_by_txid") == 0);

    wtx.mapValue["replaced_by_txid"] = newHash.ToString();

    CWalletDB walletdb(strWalletFile, "r+");

    bool success = true;
    if (!walletdb.WriteTx(wtx)) {
        LogPrintf("%s: Updating walletdb tx %s failed", __func__, wtx.GetHash().ToString());
        success = false;
    }

    NotifyTransactionChanged(this, originalHash, CT_UPDATED);

    return success;
}

bool CWallet::AddToWallet(const CWalletTx& wtxIn, bool fFlushOnClose)
{
    LOCK(cs_wallet);

    CWalletDB walletdb(strWalletFile, "r+", fFlushOnClose);

    uint256 hash = wtxIn.GetHash();

    // Inserts only if not already there, returns tx inserted or tx found
    std::pair<std::map<uint256, CWalletTx>::iterator, bool> ret = mapWallet.insert(std::make_pair(hash, wtxIn));
    CWalletTx& wtx = (*ret.first).second;
    wtx.BindWallet(this);
    bool fInsertedNew = ret.second;
    if (fInsertedNew) {
        wtx.nTimeReceived = GetAdjustedTime();
        wtx.nOrderPos = IncOrderPosNext(&walletdb);
        wtxOrdered.insert(std::make_pair(wtx.nOrderPos, TxPair(&wtx, (CAccountingEntry*)0)));

        wtx.nTimeSmart = wtx.nTimeReceived;
        if (!wtxIn.hashUnset()) {
            if (mapBlockIndex.count(wtxIn.hashBlock)) {
                int64_t latestNow = wtx.nTimeReceived;
                int64_t latestEntry = 0;
                {
                    // Tolerate times up to the last timestamp in the wallet not more than 5 minutes into the future
                    int64_t latestTolerated = latestNow + 300;
                    const TxItems& txOrdered = wtxOrdered;
                    for (TxItems::const_reverse_iterator it = txOrdered.rbegin(); it != txOrdered.rend(); ++it) {
                        CWalletTx* const pwtx = (*it).second.first;
                        if (pwtx == &wtx)
                            continue;
                        CAccountingEntry* const pacentry = (*it).second.second;
                        int64_t nSmartTime;
                        if (pwtx) {
                            nSmartTime = pwtx->nTimeSmart;
                            if (!nSmartTime)
                                nSmartTime = pwtx->nTimeReceived;
                        } else
                            nSmartTime = pacentry->nTime;
                        if (nSmartTime <= latestTolerated) {
                            latestEntry = nSmartTime;
                            if (nSmartTime > latestNow)
                                latestNow = nSmartTime;
                            break;
                        }
                    }
                }

                int64_t blocktime = mapBlockIndex[wtxIn.hashBlock]->GetBlockTime();
                wtx.nTimeSmart = std::max(latestEntry, std::min(blocktime, latestNow));
            } else
                LogPrintf("AddToWallet(): found %s in block %s not in index\n",
                    wtxIn.GetHash().ToString(),
                    wtxIn.hashBlock.ToString());
        }
        AddToSpends(hash);
        for (unsigned int i = 0; i < wtx.tx->vout.size(); ++i) {
            if (IsMine(wtx.tx->vout[i]) && !IsSpent(hash, i)) {
                setWalletUTXO.insert(COutPoint(hash, i));
            }
        }
    }

    bool fUpdated = false;
    if (!fInsertedNew) {
        // Merge
        if (!wtxIn.hashUnset() && wtxIn.hashBlock != wtx.hashBlock) {
            wtx.hashBlock = wtxIn.hashBlock;
            fUpdated = true;
        }
        // If no longer abandoned, update
        if (wtxIn.hashBlock.IsNull() && wtx.isAbandoned()) {
            wtx.hashBlock = wtxIn.hashBlock;
            fUpdated = true;
        }
        if (wtxIn.nIndex != -1 && (wtxIn.nIndex != wtx.nIndex)) {
            wtx.nIndex = wtxIn.nIndex;
            fUpdated = true;
        }
        if (wtxIn.fFromMe && wtxIn.fFromMe != wtx.fFromMe) {
            wtx.fFromMe = wtxIn.fFromMe;
            fUpdated = true;
        }
    }

    //// debug print
    LogPrint("wallet", "AddToWallet %s  %s%s\n", wtxIn.GetHash().ToString(), (fInsertedNew ? "new" : ""), (fUpdated ? "update" : ""));

    // Write to disk
    if (fInsertedNew || fUpdated)
        if (!walletdb.WriteTx(wtx))
            return false;

    // Break debit/credit balance caches:
    wtx.MarkDirty();

    // Notify UI of new or updated transaction
    NotifyTransactionChanged(this, hash, fInsertedNew ? CT_NEW : CT_UPDATED);

    // notify an external script when a wallet transaction comes in or is updated
    std::string strCmd = gArgs.GetArg("-walletnotify", "");

    if (!strCmd.empty()) {
        boost::replace_all(strCmd, "%s", wtxIn.GetHash().GetHex());
        boost::thread t(runCommand, strCmd); // thread runs free
    }

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;

    return true;
}

bool CWallet::LoadToWallet(const CWalletTx& wtxIn)
{
    uint256 hash = wtxIn.GetHash();

    mapWallet[hash] = wtxIn;
    CWalletTx& wtx = mapWallet[hash];
    wtx.BindWallet(this);
    wtxOrdered.insert(make_pair(wtx.nOrderPos, TxPair(&wtx, (CAccountingEntry*)0)));
    AddToSpends(hash);
    for (const CTxIn& txin : wtx.tx->vin) {
        if (mapWallet.count(txin.prevout.hash)) {
            CWalletTx& prevtx = mapWallet[txin.prevout.hash];
            if (prevtx.nIndex == -1 && !prevtx.hashUnset()) {
                MarkConflicted(prevtx.hashBlock, wtx.GetHash());
            }
        }
    }

    return true;
}

/**
 * Add a transaction to the wallet, or update it.  pIndex and posInBlock should
 * be set when the transaction was known to be included in a block.  When
 * posInBlock = SYNC_TRANSACTION_NOT_IN_BLOCK (-1) , then wallet state is not
 * updated in AddToWallet, but notifications happen and cached balances are
 * marked dirty.
 * If fUpdate is true, existing transactions will be updated.
 * TODO: One exception to this is that the abandoned state is cleared under the
 * assumption that any further notification of a transaction that was considered
 * abandoned is an indication that it is not safe to be considered abandoned.
 * Abandoned state should probably be more carefuly tracked via different
 * posInBlock signals or by checking mempool presence when necessary.
 */
bool CWallet::AddToWalletIfInvolvingMe(const CTransaction& tx, const CBlockIndex* pIndex, int posInBlock, bool fUpdate)
{
    {
        AssertLockHeld(cs_wallet);

        if (posInBlock != -1) {
            for (const CTxIn& txin : tx.vin) {
                std::pair<TxSpends::const_iterator, TxSpends::const_iterator> range = mapTxSpends.equal_range(txin.prevout);
                while (range.first != range.second) {
                    if (range.first->second != tx.GetHash()) {
                        LogPrintf("Transaction %s (in block %s) conflicts with wallet transaction %s (both spend %s:%i)\n", tx.GetHash().ToString(), pIndex->GetBlockHash().ToString(), range.first->second.ToString(), range.first->first.hash.ToString(), range.first->first.n);
                        MarkConflicted(pIndex->GetBlockHash(), range.first->second);
                    }
                    range.first++;
                }
            }
        }

        bool fExisted = mapWallet.count(tx.GetHash()) != 0;
        if (fExisted && !fUpdate)
            return false;

        bool fIsMyStealth = false;
        if (fStealthTx || dynodeSync.IsBlockchainSynced()) {
            // Check if stealth address belongs to this wallet
            fIsMyStealth = ScanForStealthOwnedOutputs(tx);
        }

        if (fExisted || IsMine(tx) || IsRelevantToMe(tx) || fIsMyStealth) {
            const CTransactionRef ptx = MakeTransactionRef(tx);
            if (tx.nVersion == BDAP_TX_VERSION) {
                uint256 linkID;
                CScript bdapOpScript;
                int op1, op2;
                std::vector<std::vector<unsigned char>> vvchOpParameters;
                if (GetBDAPOpScript(ptx, bdapOpScript, vvchOpParameters, op1, op2)) {
                    std::string strOpType = GetBDAPOpTypeString(op1, op2);
                    UpdateKeyPoolsFromTransactions(strOpType,vvchOpParameters);
                    if (GetOpCodeType(strOpType) == "link" && vvchOpParameters.size() > 1) {
                        uint64_t nExpireTime = 0;
                        uint64_t nHeight = pIndex ? (uint64_t)pIndex->nHeight : (uint64_t)chainActive.Height();
                        // get block time from height.
                        std::vector<unsigned char> vchLinkPubKey = vvchOpParameters[0];
                        std::vector<unsigned char> vchSharedPubKey = vvchOpParameters[1];
                        CWalletDB walletdb(strWalletFile);
                        if (strOpType == "bdap_new_link_request") {
                            int nOut;
                            std::vector<unsigned char> vchData, vchHash;
                            if (GetBDAPData(ptx, vchData, vchHash, nOut)) {
                                CLinkStorage link(vchData, vchLinkPubKey, vchSharedPubKey, (uint8_t)BDAP::LinkType::RequestType, nHeight, nExpireTime, GetTime(), tx.GetHash());
                                if (walletdb.WriteLink(link)) {
                                    LogPrint("bdap", "%s -- WriteLinkRequest nHeight = %llu, txid = %s\n", __func__, nHeight, tx.GetHash().ToString());
                                }
                            }
                        }
                        else if (strOpType == "bdap_new_link_accept") {
                            int nOut;
                            std::vector<unsigned char> vchData, vchHash;
                            if (GetBDAPData(ptx, vchData, vchHash, nOut)) {
                                CLinkStorage link(vchData, vchLinkPubKey, vchSharedPubKey, (uint8_t)BDAP::LinkType::AcceptType, nHeight, nExpireTime, GetTime(), tx.GetHash());
                                if (walletdb.WriteLink(link)) {
                                    LogPrint("bdap", "%s -- WriteLinkAccept nHeight = %llu, txid = %s\n", __func__, nHeight, tx.GetHash().ToString());
                                }
                            }
                        }
                        else if (strOpType == "bdap_delete_link_request" ||  strOpType == "bdap_delete_link_accept") {
                            // TODO (BDAP): Implement link delete
                            /*
                            if (strOpType == "bdap_delete_link_request") {
                                if (walletdb.EraseLink(vchLinkPubKey, vchSharedPubKey)) {
                                    LogPrintf("%s -- ErasePendingLink nHeight = %llu, txid = %s\n", __func__, nHeight, tx.GetHash().ToString());
                                }
                            }
                            else if (strOpType == "bdap_delete_link_accept") {
                                if (walletdb.EraseLink(vchLinkPubKey, vchSharedPubKey)) {
                                    LogPrintf("%s -- EraseCompletedLink nHeight = %llu, txid = %s\n", __func__, nHeight, tx.GetHash().ToString());
                                }
                            }
                            */
                        }
                    }
                }
            } else {
                TopUpKeyPoolCombo(0, true);
                for (const CTxOut& txout : tx.vout) {
                    CScript scriptPubKey = txout.scriptPubKey;
                    CTxDestination dest;
                    if (!ExtractDestination(scriptPubKey, dest))
                        continue;

                    CDynamicAddress address(dest);
                    CKeyID keyID;
                    if (!address.GetKeyID(keyID))
                        continue;

                    CPubKey retrievePubKey;
                    if (GetPubKey(keyID, retrievePubKey)) {
                        if (ReserveKeyForTransactions(retrievePubKey)) {
                            SetAddressBook(dest, "", "");
                            fNeedToRescanTransactions = true;
                        }
                    }
                }
            }

            CWalletTx wtx(this, ptx);
                
            // Get merkle branch if transaction was found in a block
            if (posInBlock != -1)
                wtx.SetMerkleBranch(pIndex, posInBlock);

            return AddToWallet(wtx, false);
        }
    }
    return false;
}

bool CWallet::AbandonTransaction(const uint256& hashTx)
{
    LOCK2(cs_main, cs_wallet);

    // Do not flush the wallet here for performance reasons
    CWalletDB walletdb(strWalletFile, "r+", false);

    std::set<uint256> todo;
    std::set<uint256> done;

    // Can't mark abandoned if confirmed or in mempool
    assert(mapWallet.count(hashTx));
    CWalletTx& origtx = mapWallet[hashTx];
    if (origtx.GetDepthInMainChain() > 0 || origtx.InMempool() || origtx.IsLockedByInstantSend()) {
        return false;
    }

    todo.insert(hashTx);

    while (!todo.empty()) {
        uint256 now = *todo.begin();
        todo.erase(now);
        done.insert(now);
        assert(mapWallet.count(now));
        CWalletTx& wtx = mapWallet[now];
        int currentconfirm = wtx.GetDepthInMainChain();
        // If the orig tx was not in block, none of its spends can be
        assert(currentconfirm <= 0);
        // if (currentconfirm < 0) {Tx and spends are already conflicted, no need to abandon}
        if (currentconfirm == 0 && !wtx.isAbandoned()) {
            // If the orig tx was not in block/mempool, none of its spends can be in mempool
            assert(!wtx.InMempool());
            wtx.nIndex = -1;
            wtx.setAbandoned();
            wtx.MarkDirty();
            walletdb.WriteTx(wtx);
            NotifyTransactionChanged(this, wtx.GetHash(), CT_UPDATED);
            // Iterate over all its outputs, and mark transactions in the wallet that spend them abandoned too
            TxSpends::const_iterator iter = mapTxSpends.lower_bound(COutPoint(hashTx, 0));
            while (iter != mapTxSpends.end() && iter->first.hash == now) {
                if (!done.count(iter->second)) {
                    todo.insert(iter->second);
                }
                iter++;
            }
            // If a transaction changes 'conflicted' state, that changes the balance
            // available of the outputs it spends. So force those to be recomputed
            for (const CTxIn& txin : wtx.tx->vin) {
                if (mapWallet.count(txin.prevout.hash))
                    mapWallet[txin.prevout.hash].MarkDirty();
            }
        }
    }

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;

    return true;
}

void CWallet::MarkConflicted(const uint256& hashBlock, const uint256& hashTx)
{
    LOCK2(cs_main, cs_wallet);

    int conflictconfirms = 0;
    if (mapBlockIndex.count(hashBlock)) {
        CBlockIndex* pindex = mapBlockIndex[hashBlock];
        if (chainActive.Contains(pindex)) {
            conflictconfirms = -(chainActive.Height() - pindex->nHeight + 1);
        }
    }
    // If number of conflict confirms cannot be determined, this means
    // that the block is still unknown or not yet part of the main chain,
    // for example when loading the wallet during a reindex. Do nothing in that
    // case.
    if (conflictconfirms >= 0)
        return;

    // Do not flush the wallet here for performance reasons
    CWalletDB walletdb(strWalletFile, "r+", false);

    std::set<uint256> todo;
    std::set<uint256> done;

    todo.insert(hashTx);

    while (!todo.empty()) {
        uint256 now = *todo.begin();
        todo.erase(now);
        done.insert(now);
        assert(mapWallet.count(now));
        CWalletTx& wtx = mapWallet[now];
        int currentconfirm = wtx.GetDepthInMainChain();
        if (conflictconfirms < currentconfirm) {
            // Block is 'more conflicted' than current confirm; update.
            // Mark transaction as conflicted with this block.
            wtx.nIndex = -1;
            wtx.hashBlock = hashBlock;
            wtx.MarkDirty();
            walletdb.WriteTx(wtx);
            // Iterate over all its outputs, and mark transactions in the wallet that spend them conflicted too
            TxSpends::const_iterator iter = mapTxSpends.lower_bound(COutPoint(now, 0));
            while (iter != mapTxSpends.end() && iter->first.hash == now) {
                if (!done.count(iter->second)) {
                    todo.insert(iter->second);
                }
                iter++;
            }
            // If a transaction changes 'conflicted' state, that changes the balance
            // available of the outputs it spends. So force those to be recomputed
            for (const CTxIn& txin : wtx.tx->vin) {
                if (mapWallet.count(txin.prevout.hash))
                    mapWallet[txin.prevout.hash].MarkDirty();
            }
        }
    }

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;
}

void CWallet::SyncTransaction(const CTransaction& tx, const CBlockIndex* pindex, int posInBlock)
{
    LOCK2(cs_main, cs_wallet);

    if (!AddToWalletIfInvolvingMe(tx, pindex, posInBlock, true))
        return; // Not one of ours

    // If a transaction changes 'conflicted' state, that changes the balance
    // available of the outputs it spends. So force those to be
    // recomputed, also:
    for (const CTxIn& txin : tx.vin) {
        if (mapWallet.count(txin.prevout.hash))
            mapWallet[txin.prevout.hash].MarkDirty();
    }

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;

    if (fNeedToUpdateKeyPools) {
        TopUpKeyPoolCombo(0,true);
        fNeedToUpdateKeyPools = false;
    }

    if (fNeedToUpdateLinks) {
        ProcessLinkQueue();
        fNeedToUpdateLinks = false;
    }
}

void CWallet::TransactionAddedToMempool(const CTransactionRef& ptx) {
    LOCK2(cs_main, cs_wallet);
    const CTransaction& tx = *ptx;
    SyncTransaction(tx);
}

void CWallet::BlockConnected(const std::shared_ptr<const CBlock>& pblock, const CBlockIndex *pindex, const std::vector<CTransactionRef>& vtxConflicted) {
    LOCK2(cs_main, cs_wallet);
    // TODO: Temporarily ensure that mempool removals are notified before
    // connected transactions.  This shouldn't matter, but the abandoned
    // state of transactions in our wallet is currently cleared when we
    // receive another notification and there is a race condition where
    // notification of a connected conflict might cause an outside process
    // to abandon a transaction and then have it inadvertently cleared by
    // the notification that the conflicted transaction was evicted.

    for (const CTransactionRef& ptx : vtxConflicted) {
        const CTransaction& tx = *ptx;
        SyncTransaction(tx);
    }
    for (size_t i = 0; i < pblock->vtx.size(); i++) {
        const CTransaction& tx = *pblock->vtx[i];
        SyncTransaction(tx, pindex, i);
    }
}

void CWallet::BlockDisconnected(const std::shared_ptr<const CBlock>& pblock) {
    LOCK2(cs_main, cs_wallet);

    for (const CTransactionRef& ptx : pblock->vtx) {
        const CTransaction& tx = *ptx;
        SyncTransaction(tx);
    }
}

isminetype CWallet::IsMine(const CTxIn& txin) const
{
    {
        LOCK(cs_wallet);
        std::map<uint256, CWalletTx>::const_iterator mi = mapWallet.find(txin.prevout.hash);
        if (mi != mapWallet.end()) {
            const CWalletTx& prev = (*mi).second;
            if (txin.prevout.n < prev.tx->vout.size())
                return IsMine(prev.tx->vout[txin.prevout.n]);
        }
    }
    return ISMINE_NO;
}

CAmount CWallet::GetDebit(const CTxIn &txin, const isminefilter& filter) const {
    CAssetOutputEntry assetData;
    return GetDebit(txin, filter, assetData);
}

// Note that this function doesn't distinguish between a 0-valued input,
// and a not-"is mine" (according to the filter) input.
CAmount CWallet::GetDebit(const CTxIn &txin, const isminefilter& filter, CAssetOutputEntry& assetData) const
{
    {
        LOCK(cs_wallet);
        std::map<uint256, CWalletTx>::const_iterator mi = mapWallet.find(txin.prevout.hash);
        if (mi != mapWallet.end())
        {
            const CWalletTx& prev = (*mi).second;
            if (txin.prevout.n < prev.tx->vout.size())
                if (IsMine(prev.tx->vout[txin.prevout.n]) & filter) {
                    // if asset get that assets data from the scriptPubKey
                    if (prev.tx->vout[txin.prevout.n].scriptPubKey.IsAssetScript())
                        GetAssetData(prev.tx->vout[txin.prevout.n].scriptPubKey, assetData);

                    return prev.tx->vout[txin.prevout.n].nValue;
                }
        }
    }
    return 0;
}

// Recursively determine the rounds of a given input (How deep is the PrivateSend chain for a given input)
int CWallet::GetRealOutpointPrivateSendRounds(const COutPoint& outpoint, int nRounds) const
{
    static std::map<uint256, CMutableTransaction> mDenomWtxes;

    if (nRounds >= MAX_PRIVATESEND_ROUNDS) {
        // there can only be MAX_PRIVATESEND_ROUNDS rounds max
        return MAX_PRIVATESEND_ROUNDS - 1;
    }

    uint256 hash = outpoint.hash;
    unsigned int nout = outpoint.n;

    const CWalletTx* wtx = GetWalletTx(hash);
    if (wtx != nullptr) {
        std::map<uint256, CMutableTransaction>::const_iterator mdwi = mDenomWtxes.find(hash);
        if (mdwi == mDenomWtxes.end()) {
            // not known yet, let's add it
            LogPrint("privatesend", "GetRealOutpointPrivateSendRounds INSERTING %s\n", hash.ToString());
            mDenomWtxes[hash] = CMutableTransaction(*wtx);
        } else if (mDenomWtxes[hash].vout[nout].nRounds != -10) {
            // found and it's not an initial value, just return it
            return mDenomWtxes[hash].vout[nout].nRounds;
        }


        // bounds check
        if (nout >= wtx->tx->vout.size()) {
            // should never actually hit this
            LogPrint("privatesend", "GetRealOutpointPrivateSendRounds UPDATED   %s %3d %3d\n", hash.ToString(), nout, -4);
            return -4;
        }

        if (CPrivateSend::IsCollateralAmount(wtx->tx->vout[nout].nValue)) {
            mDenomWtxes[hash].vout[nout].nRounds = -3;
            LogPrint("privatesend", "GetRealOutpointPrivateSendRounds UPDATED   %s %3d %3d\n", hash.ToString(), nout, mDenomWtxes[hash].vout[nout].nRounds);
            return mDenomWtxes[hash].vout[nout].nRounds;
        }

        //make sure the final output is non-denominate
        if (!CPrivateSend::IsDenominatedAmount(wtx->tx->vout[nout].nValue)) { //NOT DENOM
            mDenomWtxes[hash].vout[nout].nRounds = -2;
            LogPrint("privatesend", "GetRealOutpointPrivateSendRounds UPDATED   %s %3d %3d\n", hash.ToString(), nout, mDenomWtxes[hash].vout[nout].nRounds);
            return mDenomWtxes[hash].vout[nout].nRounds;
        }

        bool fAllDenoms = true;
        for (const auto& out : wtx->tx->vout) {
            fAllDenoms = fAllDenoms && CPrivateSend::IsDenominatedAmount(out.nValue);
        }

        // this one is denominated but there is another non-denominated output found in the same tx
        if (!fAllDenoms) {
            mDenomWtxes[hash].vout[nout].nRounds = 0;
            LogPrint("privatesend", "GetRealOutpointPrivateSendRounds UPDATED   %s %3d %3d\n", hash.ToString(), nout, mDenomWtxes[hash].vout[nout].nRounds);
            return mDenomWtxes[hash].vout[nout].nRounds;
        }

        int nShortest = -10; // an initial value, should be no way to get this by calculations
        bool fDenomFound = false;
        // only denoms here so let's look up
        for (const auto& txinNext : wtx->tx->vin) {
            if (IsMine(txinNext)) {
                int n = GetRealOutpointPrivateSendRounds(txinNext.prevout, nRounds + 1);
                // denom found, find the shortest chain or initially assign nShortest with the first found value
                if (n >= 0 && (n < nShortest || nShortest == -10)) {
                    nShortest = n;
                    fDenomFound = true;
                }
            }
        }
        mDenomWtxes[hash].vout[nout].nRounds = fDenomFound ? (nShortest >= MAX_PRIVATESEND_ROUNDS - 1 ? MAX_PRIVATESEND_ROUNDS : nShortest + 1) // good, we a +1 to the shortest one but only MAX_PRIVATESEND_ROUNDS rounds max allowed
                                                             :
                                                             0; // too bad, we are the fist one in that chain
        LogPrint("privatesend", "GetRealOutpointPrivateSendRounds UPDATED   %s %3d %3d\n", hash.ToString(), nout, mDenomWtxes[hash].vout[nout].nRounds);
        return mDenomWtxes[hash].vout[nout].nRounds;
    }

    return nRounds - 1;
}

// respect current settings
int CWallet::GetOutpointPrivateSendRounds(const COutPoint& outpoint) const
{
    LOCK(cs_wallet);
    int realPrivateSendRounds = GetRealOutpointPrivateSendRounds(outpoint);
    return realPrivateSendRounds > privateSendClient.nPrivateSendRounds ? privateSendClient.nPrivateSendRounds : realPrivateSendRounds;
}

bool CWallet::IsDenominated(const COutPoint& outpoint) const
{
    LOCK(cs_wallet);

    std::map<uint256, CWalletTx>::const_iterator mi = mapWallet.find(outpoint.hash);
    if (mi != mapWallet.end()) {
        const CWalletTx& prev = (*mi).second;
        if (outpoint.n < prev.tx->vout.size()) {
            return CPrivateSend::IsDenominatedAmount(prev.tx->vout[outpoint.n].nValue);
        }
    }

    return false;
}

isminetype CWallet::IsMine(const CTxOut& txout) const
{
    return ::IsMine(*this, txout.scriptPubKey);
}

CAmount CWallet::GetCredit(const CTxOut& txout, const isminefilter& filter) const
{
    if (!MoneyRange(txout.nValue))
        throw std::runtime_error("CWallet::GetCredit(): value out of range");
    return ((IsMine(txout) & filter) ? txout.nValue : 0);
}

bool CWallet::IsChange(const CTxOut& txout) const
{
    // TODO: fix handling of 'change' outputs. The assumption is that any
    // payment to a script that is ours, but is not in the address book
    // is change. That assumption is likely to break when we implement multisignature
    // wallets that return change back into a multi-signature-protected address;
    // a better way of identifying which outputs are 'the send' and which are
    // 'the change' will need to be implemented (maybe extend CWalletTx to remember
    // which output, if any, was change).
    if (::IsMine(*this, txout.scriptPubKey)) {
        CTxDestination address;
        if (!ExtractDestination(txout.scriptPubKey, address))
            return true;

        LOCK(cs_wallet);
        if (!mapAddressBook.count(address))
            return true;
    }
    return false;
}

CAmount CWallet::GetChange(const CTxOut& txout) const
{
    if (!MoneyRange(txout.nValue))
        throw std::runtime_error("CWallet::GetChange(): value out of range");
    return (IsChange(txout) ? txout.nValue : 0);
}

void CWallet::GenerateNewHDChain()
{
    CHDChain newHdChain;

    std::string strSeed = gArgs.GetArg("-hdseed", "not hex");

    if (gArgs.IsArgSet("-hdseed") && IsHex(strSeed)) {
        std::vector<unsigned char> vchSeed = ParseHex(strSeed);
        if (!newHdChain.SetSeed(SecureVector(vchSeed.begin(), vchSeed.end()), true))
            throw std::runtime_error(std::string(__func__) + ": SetSeed failed");
    } else {
        if (gArgs.IsArgSet("-hdseed") && !IsHex(strSeed))
            LogPrintf("CWallet::GenerateNewHDChain -- Incorrect seed, generating random one instead\n");

        // NOTE: empty mnemonic means "generate a new one for me"
        std::string strMnemonic = gArgs.GetArg("-mnemonic", "");
        // NOTE: default mnemonic passphrase is an empty string
        std::string strMnemonicPassphrase = gArgs.GetArg("-mnemonicpassphrase", "");

        SecureVector vchMnemonic(strMnemonic.begin(), strMnemonic.end());
        SecureVector vchMnemonicPassphrase(strMnemonicPassphrase.begin(), strMnemonicPassphrase.end());

        if (!newHdChain.SetMnemonic(vchMnemonic, vchMnemonicPassphrase, true))
            throw std::runtime_error(std::string(__func__) + ": SetMnemonic failed");
    }
    newHdChain.Debug(__func__);

    if (!SetHDChain(newHdChain, false))
        throw std::runtime_error(std::string(__func__) + ": SetHDChain failed");

    // clean up
    gArgs.ForceRemoveArg("-hdseed");
    gArgs.ForceRemoveArg("-mnemonic");
    gArgs.ForceRemoveArg("-mnemonicpassphrase");
}

bool CWallet::SetHDChain(const CHDChain& chain, bool memonly)
{
    LOCK(cs_wallet);

    if (!CCryptoKeyStore::SetHDChain(chain))
        return false;

    if (!memonly && !CWalletDB(strWalletFile).WriteHDChain(chain))
        throw std::runtime_error(std::string(__func__) + ": WriteHDChain failed");

    return true;
}

bool CWallet::SetCryptedHDChain(const CHDChain& chain, bool memonly)
{
    LOCK(cs_wallet);

    if (!CCryptoKeyStore::SetCryptedHDChain(chain))
        return false;

    if (!memonly) {
        if (!fFileBacked)
            return false;
        if (pwalletdbEncryption) {
            if (!pwalletdbEncryption->WriteCryptedHDChain(chain))
                throw std::runtime_error(std::string(__func__) + ": WriteCryptedHDChain failed");
        } else {
            if (!CWalletDB(strWalletFile).WriteCryptedHDChain(chain))
                throw std::runtime_error(std::string(__func__) + ": WriteCryptedHDChain failed");
        }
    }

    return true;
}

bool CWallet::GetDecryptedHDChain(CHDChain& hdChainRet)
{
    LOCK(cs_wallet);

    CHDChain hdChainTmp;
    if (!GetHDChain(hdChainTmp)) {
        return false;
    }

    if (!DecryptHDChain(hdChainTmp))
        return false;

    // make sure seed matches this chain
    if (hdChainTmp.GetID() != hdChainTmp.GetSeedHash())
        return false;

    hdChainRet = hdChainTmp;

    return true;
}

bool CWallet::GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const
{
    if (IsCrypted())
        return CCryptoKeyStore::GetDHTPubKeys(vvchDHTPubKeys);

    return CBasicKeyStore::GetDHTPubKeys(vvchDHTPubKeys);
}

bool CWallet::WriteLinkMessageInfo(const uint256& subjectID, const std::vector<unsigned char>& vchPubKey)
{
    CWalletDB walletdb(strWalletFile);
    return walletdb.WriteLinkMessageInfo(subjectID, vchPubKey);
}

bool CWallet::EraseLinkMessageInfo(const uint256& subjectID)
{
    CWalletDB walletdb(strWalletFile);
    return walletdb.EraseLinkMessageInfo(subjectID);
}

bool CWallet::IsHDEnabled()
{
    CHDChain hdChainCurrent;
    return GetHDChain(hdChainCurrent);
}

bool CWallet::IsMine(const CTransaction& tx) const
{
    for (const CTxOut& txout : tx.vout)
        if (IsMine(txout))
            return true;
    return false;
}

bool CWallet::IsRelevantToMe(const CTransaction& tx) const
{
    return (GetDebit(tx, ISMINE_ALL) > 0);
}

bool CWallet::IsFromMe(const CTransaction& tx, const isminefilter& filter) const
{
    for (const CTxIn& txin : tx.vin)
    {
        LOCK(cs_wallet);
        auto mi = mapWallet.find(txin.prevout.hash);
        if (mi == mapWallet.end())
            return false;

        const CWalletTx& prev = (*mi).second;
        if (txin.prevout.n < prev.tx->vout.size())
            if (!(IsMine(prev.tx->vout[txin.prevout.n]) & filter))
                return false;
    }

    return true;
}

CAmount CWallet::GetDebit(const CTransaction& tx, const isminefilter& filter) const
{
    CAmount nDebit = 0;
    for (const CTxIn& txin : tx.vin) {
        nDebit += GetDebit(txin, filter);
        if (!MoneyRange(nDebit))
            throw std::runtime_error("CWallet::GetDebit(): value out of range");
    }
    return nDebit;
}

bool CWallet::IsAllFromMe(const CTransaction& tx, const isminefilter& filter) const
{
    LOCK(cs_wallet);

    for (const CTxIn& txin : tx.vin) {
        auto mi = mapWallet.find(txin.prevout.hash);
        if (mi == mapWallet.end())
            return false; // any unknown inputs can't be from us

        const CWalletTx& prev = (*mi).second;

        if (txin.prevout.n >= prev.tx->vout.size())
            return false; // invalid input!

        if (!(IsMine(prev.tx->vout[txin.prevout.n]) & filter))
            return false;
    }
    return true;
}

CAmount CWallet::GetCredit(const CTransaction& tx, const isminefilter& filter) const
{
    CAmount nCredit = 0;
    for (const CTxOut& txout : tx.vout) {
        nCredit += GetCredit(txout, filter);
        if (!MoneyRange(nCredit))
            throw std::runtime_error("CWallet::GetCredit(): value out of range");
    }
    return nCredit;
}

CAmount CWallet::GetChange(const CTransaction& tx) const
{
    CAmount nChange = 0;
    for (const CTxOut& txout : tx.vout) {
        nChange += GetChange(txout);
        if (!MoneyRange(nChange))
            throw std::runtime_error("CWallet::GetChange(): value out of range");
    }
    return nChange;
}

int64_t CWalletTx::GetTxTime() const
{
    int64_t n = nTimeSmart;
    return n ? n : nTimeReceived;
}

int CWalletTx::GetRequestCount() const
{
    // Returns -1 if it wasn't being tracked
    int nRequests = -1;
    {
        LOCK(pwallet->cs_wallet);
        if (IsCoinBase()) {
            // Generated block
            if (!hashUnset()) {
                std::map<uint256, int>::const_iterator _mi = pwallet->mapRequestCount.find(hashBlock);
                if (_mi != pwallet->mapRequestCount.end())
                    nRequests = (*_mi).second;
            }
        } else {
            // Did anyone request this transaction?
            std::map<uint256, int>::const_iterator mi = pwallet->mapRequestCount.find(GetHash());
            if (mi != pwallet->mapRequestCount.end()) {
                nRequests = (*mi).second;

                // How about the block it's in?
                if (nRequests == 0 && !hashUnset()) {
                    std::map<uint256, int>::const_iterator mi = pwallet->mapRequestCount.find(hashBlock);
                    if (mi != pwallet->mapRequestCount.end())
                        nRequests = (*mi).second;
                    else
                        nRequests = 1; // If it's in someone else's block it must have got out
                }
            }
        }
    }
    return nRequests;
}

void CWalletTx::GetAmounts(std::list<COutputEntry>& listReceived,
                           std::list<COutputEntry>& listSent, CAmount& nFee, std::string& strSentAccount, const isminefilter& filter) const {

    std::list<CAssetOutputEntry> assetsReceived;
    std::list<CAssetOutputEntry> assetsSent;
    GetAmounts(listReceived, listSent, nFee, strSentAccount, filter, assetsReceived, assetsSent);
}

void CWalletTx::GetAmounts(std::list<COutputEntry>& listReceived,
                           std::list<COutputEntry>& listSent, CAmount& nFee, std::string& strSentAccount, const isminefilter& filter, std::list<CAssetOutputEntry>& assetsReceived, std::list<CAssetOutputEntry>& assetsSent) const
{
    nFee = 0;
    listReceived.clear();
    listSent.clear();
    strSentAccount = strFromAccount;

    // Compute fee:
    CAmount nDebit = GetDebit(filter);
    CAmount nCredit = 0;
    bool fFromMe = IsFromMe(filter);
    if (fFromMe) // means we signed/sent this transaction and all inputs are from us
    {
        CAmount nValueOut = tx->GetValueOut();
        nFee = nDebit - nValueOut;
    }

    // treat coinstake as a single "recieve" entry
    if (IsCoinStake())
    {
        for (unsigned int i = 0; i < tx->vout.size(); ++i)
        {
            const CTxOut& txout = tx->vout[i];
            isminetype fIsMine = pwallet->IsMine(txout);

            // get my vout with positive output
            if (!(fIsMine & filter) || txout.nValue <= 0)
                        continue;

            // get address
            CTxDestination address = CNoDestination();
            ExtractDestination(txout.scriptPubKey, address);

            // nfee is negative for coinstake generation, because we are gaining money from it
            COutputEntry output = {address, -nFee, (int)i};
            listReceived.push_back(output);
            nFee = 0;
            return;
        }

        // if we reach here there is probably a mistake
        COutputEntry output = {CNoDestination(), 0, 0};
        listReceived.push_back(output);
        return;
    }

    // Sent/received.
    for (unsigned int i = 0; i < tx->vout.size(); ++i) {
        const CTxOut& txout = tx->vout[i];
        isminetype fIsMine = pwallet->IsMine(txout);
        // Only need to handle txouts if AT LEAST one of these is true:
        //   1) they debit from us (sent)
        //   2) the output is to us (received)
        if (fFromMe) {
            // Don't report 'change' txouts
            if (pwallet->IsChange(txout))
                continue;
        }

        if (nDebit == 0 && !(fIsMine & filter))
            continue;

        // In either case, we need to get the destination address
        CTxDestination address;

        if (!ExtractDestination(txout.scriptPubKey, address) && !txout.scriptPubKey.IsUnspendable()) {
            if (!IsCoinStake() && !IsCoinBase()) {
                LogPrintf("CWalletTx::GetAmounts: Unknown transaction type found, txid %s\n", this->GetHash().ToString());
            }
            address = CNoDestination();
        }

        COutputEntry output = {address, txout.nValue, (int)i};

        /** ASSET START */
        // In either case, we need to get the destination address
        if (!txout.scriptPubKey.IsAssetScript()) {
            COutputEntry output = {address, txout.nValue, (int) i};

            // If we are debited by the transaction, add the output as a "sent" entry
            if (fFromMe)
                listSent.push_back(output);

            // If we are receiving the output, add it as a "received" entry
            if (fIsMine & filter) {
                nCredit += txout.nValue;
                listReceived.push_back(output);
            }
        }
        if (AreAssetsDeployed()) {
            if (txout.scriptPubKey.IsAssetScript()) {
                CAssetOutputEntry assetoutput;
                assetoutput.vout = i;
                GetAssetData(txout.scriptPubKey, assetoutput);

                // The only asset type we send is transfer_asset. We need to skip all other types for the sent category
                if (nDebit > 0 && assetoutput.type == TX_TRANSFER_ASSET)
                    assetsSent.emplace_back(assetoutput);

                if (fIsMine & filter)
                    assetsReceived.emplace_back(assetoutput);
            }
        }
        /** ASSET END */
    }

    if (!fFromMe && nDebit > 0) {
        if (nCredit == nDebit) {
            for(const auto& output: listReceived)
                listSent.push_back(output);
        } else {
            COutputEntry output = {CNoDestination(), nDebit, -1};
            listSent.push_back(output);
        }
    }
}

void CWalletTx::GetAccountAmounts(const std::string& strAccount, CAmount& nReceived, CAmount& nSent, CAmount& nFee, const isminefilter& filter) const
{
    nReceived = nSent = nFee = 0;

    CAmount allFee;
    std::string strSentAccount;
    std::list<COutputEntry> listReceived;
    std::list<COutputEntry> listSent;
    GetAmounts(listReceived, listSent, allFee, strSentAccount, filter);

    if (strAccount == strSentAccount) {
        for (const COutputEntry& s : listSent)
            nSent += s.amount;
        nFee = allFee;
    }
    {
        LOCK(pwallet->cs_wallet);
        for (const COutputEntry& r : listReceived) {
            if (pwallet->mapAddressBook.count(r.destination)) {
                std::map<CTxDestination, CAddressBookData>::const_iterator mi = pwallet->mapAddressBook.find(r.destination);
                if (mi != pwallet->mapAddressBook.end() && (*mi).second.name == strAccount)
                    nReceived += r.amount;
            } else if (strAccount.empty()) {
                nReceived += r.amount;
            }
        }
    }
}

/**
 * Scan the block chain (starting in pindexStart) for transactions
 * from or to us. If fUpdate is true, found transactions that already
 * exist in the wallet will be updated.
 */
CBlockIndex* CWallet::ScanForWalletTransactions(CBlockIndex* pindexStart, bool fUpdate)
{
    CBlockIndex* ret = nullptr;
    int64_t nNow = GetTime();
    const CChainParams& chainParams = Params();

    CBlockIndex* pindex = pindexStart;
    {
        LOCK2(cs_main, cs_wallet);

        // no need to read and scan block, if block was created before
        // our wallet birthday (as adjusted for block time variability)
        while (pindex && nTimeFirstKey && (pindex->GetBlockTime() < (nTimeFirstKey - 7200)))
            pindex = chainActive.Next(pindex);

        ShowProgress(_("Rescanning..."), 0); // show rescan progress in GUI as dialog or on splashscreen, if -rescan on startup
        double dProgressStart = GuessVerificationProgress(chainParams.TxData(), pindex);
        double dProgressTip = GuessVerificationProgress(chainParams.TxData(), chainActive.Tip());
        while (pindex) {
            if (pindex->nHeight % 100 == 0 && dProgressTip - dProgressStart > 0.0)
                ShowProgress(_("Rescanning..."), std::max(1, std::min(99, (int)((GuessVerificationProgress(chainParams.TxData(), pindex) - dProgressStart) / (dProgressTip - dProgressStart) * 100))));
            if (GetTime() >= nNow + 60) {
                nNow = GetTime();
                LogPrintf("Still rescanning. At block %d. Progress=%f\n", pindex->nHeight, GuessVerificationProgress(chainParams.TxData(), pindex));
            }

            CBlock block;
            if (ReadBlockFromDisk(block, pindex, Params().GetConsensus())) {
                for (size_t posInBlock = 0; posInBlock < block.vtx.size(); ++posInBlock) {
                    AddToWalletIfInvolvingMe(*block.vtx[posInBlock], pindex, posInBlock, fUpdate);

                    if (SaveRescanIndex) {
                        rescan_index = pindex;    
                        SaveRescanIndex = false;
                    }

                    if (fNeedToUpdateKeyPools) {
                        TopUpKeyPoolCombo(0,true);
                        fNeedToUpdateKeyPools = false;
                    }

                    if (fNeedToUpdateLinks) {
                        ProcessLinkQueue();
                        fNeedToUpdateLinks = false;
                    }
                }
                if (!ret) {
                    ret = pindex;
                }
            } else {
                ret = nullptr;
            }
            pindex = chainActive.Next(pindex);


        }

        ShowProgress(_("Rescanning..."), 100); // hide progress dialog in GUI
    }
    return ret;
}

void CWallet::ReacceptWalletTransactions()
{
    // If transactions aren't being broadcasted, don't let them into local mempool either
    if (!fBroadcastTransactions)
        return;
    LOCK2(cs_main, cs_wallet);
    std::map<int64_t, CWalletTx*> mapSorted;

    // Sort pending wallet transactions based on their initial wallet insertion order
    for (std::pair<const uint256, CWalletTx>& item : mapWallet) {
        const uint256& wtxid = item.first;
        CWalletTx& wtx = item.second;
        assert(wtx.GetHash() == wtxid);

        int nDepth = wtx.GetDepthInMainChain();

        if (!wtx.IsCoinBase() && !wtx.IsCoinStake() && (nDepth == 0 && !wtx.IsLockedByInstantSend() && !wtx.isAbandoned())) {
            mapSorted.insert(std::make_pair(wtx.nOrderPos, &wtx));
        }
    }

    // Try to add wallet transactions to memory pool
    for (std::pair<const int64_t, CWalletTx*>& item : mapSorted) {
        CWalletTx& wtx = *(item.second);

        LOCK(mempool.cs);
        CValidationState state;
        wtx.AcceptToMemoryPool(maxTxFee, state);
    }
}

bool CWalletTx::RelayWalletTransaction(CConnman* connman, const std::string& strCommand)
{
    assert(pwallet->GetBroadcastTransactions());
    if ((!IsCoinBase() || IsCoinStake()) && !isAbandoned() && GetDepthInMainChain() == 0) {
        CValidationState state;
        /* GetDepthInMainChain already catches known conflicts. */
        if (InMempool() || AcceptToMemoryPool(maxTxFee, state)) {
            uint256 hash = GetHash();
            LogPrintf("Relaying wtx %s\n", hash.ToString());

            if ((strCommand == NetMsgType::TXLOCKREQUEST) ||
                ((CTxLockRequest(*this).IsSimple()) && CInstantSend::CanAutoLock())) {
                if (instantsend.ProcessTxLockRequest((CTxLockRequest) * this, connman)) {
                    instantsend.AcceptLockRequest((CTxLockRequest) * this);
                } else {
                    instantsend.RejectLockRequest((CTxLockRequest) * this);
                }
            }
            if (connman) {
                connman->RelayTransaction((CTransaction) * this);
                return true;
            }
        }
    }
    return false;
}

std::set<uint256> CWalletTx::GetConflicts() const
{
    std::set<uint256> result;
    if (pwallet != nullptr) {
        uint256 myHash = GetHash();
        result = pwallet->GetConflicts(myHash);
        result.erase(myHash);
    }
    return result;
}

CAmount CWalletTx::GetDebit(const isminefilter& filter) const
{
    if (tx->vin.empty())
        return 0;

    CAmount debit = 0;
    if (filter & ISMINE_SPENDABLE) {
        if (fDebitCached)
            debit += nDebitCached;
        else {
            nDebitCached = pwallet->GetDebit(*this, ISMINE_SPENDABLE);
            fDebitCached = true;
            debit += nDebitCached;
        }
    }
    if (filter & ISMINE_WATCH_ONLY) {
        if (fWatchDebitCached)
            debit += nWatchDebitCached;
        else {
            nWatchDebitCached = pwallet->GetDebit(*this, ISMINE_WATCH_ONLY);
            fWatchDebitCached = true;
            debit += nWatchDebitCached;
        }
    }
    return debit;
}

bool CWalletTx::IsFromMe(const isminefilter& filter) const
{
    if (fFromMeCached)
        return fFromMeCachedValue;

    fFromMeCachedValue = pwallet->IsFromMe(*this, filter);
    fFromMeCached = true;
    return fFromMeCachedValue;
}

CAmount CWalletTx::GetCredit(const isminefilter& filter) const
{
    // Must wait until coinbase is safely deep enough in the chain before valuing it
    if (IsCoinBase() && GetBlocksToMaturity() > 0)
        return 0;

    CAmount credit = 0;
    if (filter & ISMINE_SPENDABLE) {
        // GetBalance can assume transactions in mapWallet won't change
        if (fCreditCached)
            credit += nCreditCached;
        else {
            nCreditCached = pwallet->GetCredit(*this, ISMINE_SPENDABLE);
            fCreditCached = true;
            credit += nCreditCached;
        }
    }
    if (filter & ISMINE_WATCH_ONLY) {
        if (fWatchCreditCached)
            credit += nWatchCreditCached;
        else {
            nWatchCreditCached = pwallet->GetCredit(*this, ISMINE_WATCH_ONLY);
            fWatchCreditCached = true;
            credit += nWatchCreditCached;
        }
    }
    return credit;
}

CAmount CWalletTx::GetImmatureCredit(bool fUseCache) const
{
    if ((IsCoinBase() || IsCoinStake()) && GetBlocksToMaturity() > 0 && IsInMainChain()) {
        if (fUseCache && fImmatureCreditCached)
            return nImmatureCreditCached;
        nImmatureCreditCached = pwallet->GetCredit(*this, ISMINE_SPENDABLE);
        fImmatureCreditCached = true;
        return nImmatureCreditCached;
    }

    return 0;
}

CAmount CWalletTx::GetAvailableCredit(bool fUseCache) const
{
    if (pwallet == 0)
        return 0;

    // Must wait until coinbase is safely deep enough in the chain before valuing it
    if ((IsCoinBase() || IsCoinStake()) && GetBlocksToMaturity() > 0)
        return 0;

    if (fUseCache && fAvailableCreditCached)
        return nAvailableCreditCached;

    CAmount nCredit = 0;
    uint256 hashTx = GetHash();
    for (unsigned int i = 0; i < tx->vout.size(); i++) {
        if (!pwallet->IsSpent(hashTx, i)) {
            const CTxOut& txout = tx->vout[i];
            if (txout.IsBDAP())
                continue;

            nCredit += pwallet->GetCredit(txout, ISMINE_SPENDABLE);
            if (!MoneyRange(nCredit))
                throw std::runtime_error("CWalletTx::GetAvailableCredit() : value out of range");
        }
    }

    nAvailableCreditCached = nCredit;
    fAvailableCreditCached = true;
    return nCredit;
}

CAmount CWalletTx::GetImmatureWatchOnlyCredit(const bool& fUseCache) const
{
    if (IsCoinBase() && GetBlocksToMaturity() > 0 && IsInMainChain()) {
        if (fUseCache && fImmatureWatchCreditCached)
            return nImmatureWatchCreditCached;
        nImmatureWatchCreditCached = pwallet->GetCredit(*this, ISMINE_WATCH_ONLY);
        fImmatureWatchCreditCached = true;
        return nImmatureWatchCreditCached;
    }

    return 0;
}

CAmount CWalletTx::GetAvailableWatchOnlyCredit(const bool& fUseCache) const
{
    if (pwallet == 0)
        return 0;

    // Must wait until coinbase is safely deep enough in the chain before valuing it
    if ((IsCoinBase() || IsCoinStake()) && GetBlocksToMaturity() > 0)
        return 0;

    if (fUseCache && fAvailableWatchCreditCached)
        return nAvailableWatchCreditCached;

    CAmount nCredit = 0;
    for (unsigned int i = 0; i < tx->vout.size(); i++) {
        if (!pwallet->IsSpent(GetHash(), i)) {
            const CTxOut& txout = tx->vout[i];
            nCredit += pwallet->GetCredit(txout, ISMINE_WATCH_ONLY);
            if (!MoneyRange(nCredit))
                throw std::runtime_error("CWalletTx::GetAvailableCredit() : value out of range");
        }
    }

    nAvailableWatchCreditCached = nCredit;
    fAvailableWatchCreditCached = true;
    return nCredit;
}

CAmount CWalletTx::GetAnonymizedCredit(bool fUseCache) const
{
    if (pwallet == 0)
        return 0;

    // Must wait until coinbase is safely deep enough in the chain before valuing it
    if (IsCoinBase() && GetBlocksToMaturity() > 0)
        return 0;

    if (fUseCache && fAnonymizedCreditCached)
        return nAnonymizedCreditCached;

    CAmount nCredit = 0;
    uint256 hashTx = GetHash();
    for (unsigned int i = 0; i < tx->vout.size(); i++) {
        const CTxOut& txout = tx->vout[i];
        const COutPoint outpoint = COutPoint(hashTx, i);

        if (pwallet->IsSpent(hashTx, i) || !pwallet->IsDenominated(outpoint))
            continue;

        const int nRounds = pwallet->GetOutpointPrivateSendRounds(outpoint);
        if (nRounds >= privateSendClient.nPrivateSendRounds) {
            nCredit += pwallet->GetCredit(txout, ISMINE_SPENDABLE);
            if (!MoneyRange(nCredit))
                throw std::runtime_error("CWalletTx::GetAnonymizedCredit() : value out of range");
        }
    }

    nAnonymizedCreditCached = nCredit;
    fAnonymizedCreditCached = true;
    return nCredit;
}

CAmount CWalletTx::GetDenominatedCredit(bool unconfirmed, bool fUseCache) const
{
    if (pwallet == 0)
        return 0;

    // Must wait until coinbase is safely deep enough in the chain before valuing it
    if (IsCoinBase() && GetBlocksToMaturity() > 0)
        return 0;

    int nDepth = GetDepthInMainChain();
    if (nDepth < 0)
        return 0;

    bool isUnconfirmed = IsTrusted() && nDepth == 0;
    if (unconfirmed != isUnconfirmed)
        return 0;

    if (fUseCache) {
        if (unconfirmed && fDenomUnconfCreditCached)
            return nDenomUnconfCreditCached;
        else if (!unconfirmed && fDenomConfCreditCached)
            return nDenomConfCreditCached;
    }

    CAmount nCredit = 0;
    uint256 hashTx = GetHash();
    for (unsigned int i = 0; i < tx->vout.size(); i++) {
        const CTxOut& txout = tx->vout[i];

        if (pwallet->IsSpent(hashTx, i) || !CPrivateSend::IsDenominatedAmount(tx->vout[i].nValue))
            continue;

        nCredit += pwallet->GetCredit(txout, ISMINE_SPENDABLE);
        if (!MoneyRange(nCredit))
            throw std::runtime_error("CWalletTx::GetDenominatedCredit() : value out of range");
    }

    if (unconfirmed) {
        nDenomUnconfCreditCached = nCredit;
        fDenomUnconfCreditCached = true;
    } else {
        nDenomConfCreditCached = nCredit;
        fDenomConfCreditCached = true;
    }
    return nCredit;
}

CAmount CWalletTx::GetChange() const
{
    if (fChangeCached)
        return nChangeCached;
    nChangeCached = pwallet->GetChange(*this);
    fChangeCached = true;
    return nChangeCached;
}

bool CWalletTx::InMempool() const
{
    LOCK(mempool.cs);
    if (mempool.exists(GetHash())) {
        return true;
    }
    return false;
}

bool CWalletTx::IsTrusted() const
{
    // Quick answer in most cases
    if (!CheckFinalTx(*this))
        return false;
    int nDepth = GetDepthInMainChain();
    if (nDepth >= 1)
        return true;
    if (nDepth < 0)
        return false;
    if (IsLockedByInstantSend())
        return true;
    if (!bSpendZeroConfChange || !IsFromMe(ISMINE_ALL)) // using wtx's cached debit
        return false;

    // Don't trust unconfirmed transactions from us unless they are in the mempool.
    if (!InMempool())
        return false;

    // Trusted if all inputs are from us and are in the mempool:
    for (const CTxIn& txin : tx->vin) {
        // Transactions not sent by us: not trusted
        const CWalletTx* parent = pwallet->GetWalletTx(txin.prevout.hash);
        if (parent == nullptr)
            return false;
        const CTxOut& parentOut = parent->tx->vout[txin.prevout.n];
        if (pwallet->IsMine(parentOut) != ISMINE_SPENDABLE)
            return false;
    }
    return true;
}

bool CWalletTx::IsEquivalentTo(const CWalletTx& _tx) const
{
    CMutableTransaction tx1 = *this->tx;
    CMutableTransaction tx2 = *_tx.tx;
    for (unsigned int i = 0; i < tx1.vin.size(); i++)
        tx1.vin[i].scriptSig = CScript();
    for (unsigned int i = 0; i < tx2.vin.size(); i++)
        tx2.vin[i].scriptSig = CScript();
    return CTransaction(tx1) == CTransaction(tx2);
}

bool CWalletTx::IsBDAP() const
{
    return tx->nVersion == BDAP_TX_VERSION ? true : false;
}

std::vector<uint256> CWallet::ResendWalletTransactionsBefore(int64_t nTime, CConnman* connman)
{
    std::vector<uint256> result;

    LOCK(cs_wallet);
    // Sort them in chronological order
    std::multimap<unsigned int, CWalletTx*> mapSorted;
    for (std::pair<const uint256, CWalletTx>& item : mapWallet) {
        CWalletTx& wtx = item.second;
        // Don't rebroadcast if newer than nTime:
        if (wtx.nTimeReceived > nTime)
            continue;
        mapSorted.insert(std::make_pair(wtx.nTimeReceived, &wtx));
    }
    for (std::pair<const unsigned int, CWalletTx*>& item: mapSorted) {
        CWalletTx& wtx = *item.second;
        if (wtx.RelayWalletTransaction(connman))
            result.push_back(wtx.GetHash());
    }
    return result;
}

void CWallet::ResendWalletTransactions(int64_t nBestBlockTime, CConnman* connman)
{
    // Do this infrequently and randomly to avoid giving away
    // that these are our transactions.
    if (GetTime() < nNextResend || !fBroadcastTransactions)
        return;
    bool fFirst = (nNextResend == 0);
    nNextResend = GetTime() + GetRand(30 * 60);
    if (fFirst)
        return;

    // Only do it if there's been a new block since last time
    if (nBestBlockTime < nLastResend)
        return;
    nLastResend = GetTime();

    // Rebroadcast unconfirmed txes older than 5 minutes before the last
    // block was found:
    std::vector<uint256> relayed = ResendWalletTransactionsBefore(nBestBlockTime - 5 * 60, connman);
    if (!relayed.empty())
        LogPrintf("%s: rebroadcast %u unconfirmed transactions\n", __func__, relayed.size());
}

/** @} */ // end of mapWallet


/** @defgroup Actions
 *
 * @{
 */


CAmount CWallet::GetBalance() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            if (pcoin->IsTrusted())
                nTotal += pcoin->GetAvailableCredit();
        }
    }

    return nTotal;
}

CAmount CWallet::GetTotal() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it)
        {
            const CWalletTx* pcoin = &(*it).second;
            if (pcoin->IsTrusted())
                nTotal += pcoin->GetAvailableCredit();
        }
    }

    return nTotal;
}

// ppcoin: total coins staked (non-spendable until maturity)
CAmount CWallet::GetStake() const
{
    CAmount nTotal = 0;
    LOCK2(cs_main, cs_wallet);
    for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it)
    {
        const CWalletTx* pcoin = &(*it).second;
        if (pcoin->IsCoinStake() && pcoin->GetBlocksToMaturity() > 0 && pcoin->GetDepthInMainChain() > 0)
            nTotal += CWallet::GetCredit(*pcoin, ISMINE_ALL);
    }
    return nTotal;
}

CAmount CWallet::GetAnonymizableBalance(bool fSkipDenominated, bool fSkipUnconfirmed) const
{
    if (fLiteMode)
        return 0;

    std::vector<CompactTallyItem> vecTally;
    if (!SelectCoinsGroupedByAddresses(vecTally, fSkipDenominated, true, fSkipUnconfirmed))
        return 0;

    CAmount nTotal = 0;

    const CAmount nSmallestDenom = CPrivateSend::GetSmallestDenomination();
    const CAmount nMixingCollateral = CPrivateSend::GetCollateralAmount();
    for (CompactTallyItem& item : vecTally) {
        bool fIsDenominated = CPrivateSend::IsDenominatedAmount(item.nAmount);
        if (fSkipDenominated && fIsDenominated)
            continue;
        // assume that the fee to create denoms should be mixing collateral at max
        if (item.nAmount >= nSmallestDenom + (fIsDenominated ? 0 : nMixingCollateral))
            nTotal += item.nAmount;
    }

    return nTotal;
}

CAmount CWallet::GetAnonymizedBalance() const
{
    if (fLiteMode)
        return 0;

    CAmount nTotal = 0;

    LOCK2(cs_main, cs_wallet);

    std::set<uint256> setWalletTxesCounted;
    for (auto& outpoint : setWalletUTXO) {
        if (setWalletTxesCounted.find(outpoint.hash) != setWalletTxesCounted.end())
            continue;
        setWalletTxesCounted.insert(outpoint.hash);

        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.find(outpoint.hash); it != mapWallet.end() && it->first == outpoint.hash; ++it) {
            if (it->second.IsTrusted())
                nTotal += it->second.GetAnonymizedCredit();
        }
    }

    return nTotal;
}

// Note: calculated including unconfirmed,
// that's ok as long as we use it for informational purposes only
float CWallet::GetAverageAnonymizedRounds() const
{
    if (fLiteMode)
        return 0;

    int nTotal = 0;
    int nCount = 0;

    LOCK2(cs_main, cs_wallet);
    for (auto& outpoint : setWalletUTXO) {
        if (!IsDenominated(outpoint))
            continue;

        nTotal += GetOutpointPrivateSendRounds(outpoint);
        nCount++;
    }

    if (nCount == 0)
        return 0;

    return (float)nTotal / nCount;
}

// Note: calculated including unconfirmed,
// that's ok as long as we use it for informational purposes only
CAmount CWallet::GetNormalizedAnonymizedBalance() const
{
    if (fLiteMode)
        return 0;

    CAmount nTotal = 0;

    LOCK2(cs_main, cs_wallet);
    for (auto& outpoint : setWalletUTXO) {
        std::map<uint256, CWalletTx>::const_iterator it = mapWallet.find(outpoint.hash);
        if (it == mapWallet.end())
            continue;
        if (!IsDenominated(outpoint))
            continue;
        if (it->second.GetDepthInMainChain() < 0)
            continue;

        int nRounds = GetOutpointPrivateSendRounds(outpoint);
        nTotal += it->second.tx->vout[outpoint.n].nValue * nRounds / privateSendClient.nPrivateSendRounds;
    }

    return nTotal;
}

CAmount CWallet::GetNeedsToBeAnonymizedBalance(CAmount nMinBalance) const
{
    if (fLiteMode)
        return 0;

    CAmount nAnonymizedBalance = GetAnonymizedBalance();
    CAmount nNeedsToAnonymizeBalance = privateSendClient.nPrivateSendAmount * COIN - nAnonymizedBalance;

    // try to overshoot target PS balance up to nMinBalance
    nNeedsToAnonymizeBalance += nMinBalance;

    CAmount nAnonymizableBalance = GetAnonymizableBalance();

    // anonymizable balance is way too small
    if (nAnonymizableBalance < nMinBalance)
        return 0;

    // not enough funds to anonymze amount we want, try the max we can
    if (nNeedsToAnonymizeBalance > nAnonymizableBalance)
        nNeedsToAnonymizeBalance = nAnonymizableBalance;

    // we should never exceed the pool max
    if (nNeedsToAnonymizeBalance > CPrivateSend::GetMaxPoolAmount())
        nNeedsToAnonymizeBalance = CPrivateSend::GetMaxPoolAmount();

    return nNeedsToAnonymizeBalance;
}

CAmount CWallet::GetDenominatedBalance(bool unconfirmed) const
{
    if (fLiteMode)
        return 0;

    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;

            nTotal += pcoin->GetDenominatedCredit(unconfirmed);
        }
    }

    return nTotal;
}

CAmount CWallet::GetUnconfirmedBalance() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            if (!pcoin->IsTrusted() && pcoin->GetDepthInMainChain() == 0 && !pcoin->IsLockedByInstantSend() && pcoin->InMempool())
                nTotal += pcoin->GetAvailableCredit();
        }
    }
    return nTotal;
}

CAmount CWallet::GetImmatureBalance() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            nTotal += pcoin->GetImmatureCredit();
        }
    }
    return nTotal;
}

CAmount CWallet::GetWatchOnlyBalance() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            if (pcoin->IsTrusted())
                nTotal += pcoin->GetAvailableWatchOnlyCredit();
        }
    }

    return nTotal;
}

CAmount CWallet::GetWatchOnlyStake() const
{
    CAmount nTotal = 0;
    LOCK2(cs_main, cs_wallet);
    for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it)
    {
        const CWalletTx* pcoin = &(*it).second;
        if (pcoin->IsCoinStake() && pcoin->GetBlocksToMaturity() > 0 && pcoin->GetDepthInMainChain() > 0)
            nTotal += CWallet::GetCredit(*pcoin, ISMINE_WATCH_ONLY);
    }
    return nTotal;
}

CAmount CWallet::GetUnconfirmedWatchOnlyBalance() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            if (!pcoin->IsTrusted() && pcoin->GetDepthInMainChain() == 0 && !pcoin->IsLockedByInstantSend() && pcoin->InMempool())
                nTotal += pcoin->GetAvailableWatchOnlyCredit();
        }
    }
    return nTotal;
}

CAmount CWallet::GetImmatureWatchOnlyBalance() const
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            nTotal += pcoin->GetImmatureWatchOnlyCredit();
        }
    }
    return nTotal;
}

void CWallet::GetBDAPCoins(std::vector<COutput>& vCoins, const CScript& prevScriptPubKey) const
{
    CDynamicAddress prevAddress = GetScriptAddress(prevScriptPubKey);
    //LogPrintf("GetBDAPCoins prevAddress =  %s\n", prevAddress.ToString());
    vCoins.clear();
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const uint256& wtxid = it->first;
            const CWalletTx* pcoin = &(*it).second;

            if (!CheckFinalTx(*pcoin))
                continue;

            if (!pcoin->IsTrusted())
                continue;

            if (pcoin->IsCoinBase() && pcoin->GetBlocksToMaturity() > 0)
                continue;

            int nDepth = pcoin->GetDepthInMainChain();

            // We should not consider coins which aren't at least in our mempool
            // It's possible for these to be conflicted via ancestors which we may never be able to detect
            if (nDepth == 0 && !pcoin->InMempool())
                continue;

            for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++) {
                isminetype mine = IsMine(pcoin->tx->vout[i]);
                if (!(IsSpent(wtxid, i)) && mine != ISMINE_NO && (!IsLockedCoin((*it).first, i)) && (pcoin->tx->vout[i].nValue > 0)) {
                    CDynamicAddress address = GetScriptAddress(pcoin->tx->vout[i].scriptPubKey);
                    //LogPrintf("GetBDAPCoins address =  %s\n", address.ToString());
                    if (prevAddress == address) {
                        bool safeTx = pcoin->IsTrusted();
                        vCoins.push_back(COutput(pcoin, i, nDepth, true, (mine & (ISMINE_SPENDABLE | ISMINE_WATCH_SOLVABLE)) != ISMINE_NO, safeTx));
                    }
                }
            }
        }
    }
}

CAmount CWallet::GetBDAPDynamicAmount() const
{
    std::vector<std::pair<CTxOut, COutPoint>> vCredits;
    pwalletMain->AvailableBDAPCredits(vCredits);
    CAmount nTotalCredits = 0;
    for (const std::pair<CTxOut, COutPoint>& credit : vCredits) {
        int opCode1 = -1; int opCode2 = -1;
        std::vector<std::vector<unsigned char>> vvch;
        credit.first.GetBDAPOpCodes(opCode1, opCode2, vvch);
        std::string strOpType = GetBDAPOpTypeString(opCode1, opCode2);
        const CDynamicAddress address = GetScriptAddress(credit.first.scriptPubKey);
        if (strOpType == "bdap_move_asset") {
            if (vvch.size() > 1) {
                std::string strMoveDestination = stringFromVch(vvch[1]);
                if (strMoveDestination == "BDAP") {
                    nTotalCredits += credit.first.nValue;
                }
            }
        }
    }
    return nTotalCredits;
}

/** ASSET START */

void CWallet::AvailableCoins(std::vector<COutput> &vCoins, bool fOnlySafe, const CCoinControl *coinControl, const CAmount &nMinimumAmount, const CAmount &nMaximumAmount, const CAmount &nMinimumSumAmount, const uint64_t &nMaximumCount, const int &nMinDepth, const int &nMaxDepth, bool fOnlyConfirmed, bool fIncludeZeroValue, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fUseBDAP) const
{
    std::map<std::string, std::vector<COutput> > mapAssetCoins;
    AvailableCoinsAll(vCoins, mapAssetCoins, true, false, fOnlySafe, coinControl, nMinimumAmount, nMaximumAmount, nMinimumSumAmount, nMaximumCount, nMinDepth, nMaxDepth, fOnlyConfirmed, fIncludeZeroValue, nCoinType, fUseInstantSend, fUseBDAP);
}

void CWallet::AvailableAssets(std::map<std::string, std::vector<COutput> > &mapAssetCoins, bool fOnlySafe,
                              const CCoinControl* coinControl, const CAmount &nMinimumAmount,
                              const CAmount &nMaximumAmount, const CAmount &nMinimumSumAmount,
                              const uint64_t &nMaximumCount, const int &nMinDepth, const int &nMaxDepth,
                              bool fOnlyConfirmed, bool fIncludeZeroValue, AvailableCoinsType nCoinType, 
                              bool fUseInstantSend, bool fUseBDAP) const
{
    if (!AreAssetsDeployed())
        return;

    std::vector<COutput> vCoins;

    AvailableCoinsAll(vCoins, mapAssetCoins, false, true, fOnlySafe, coinControl, nMinimumAmount, nMaximumAmount, nMinimumSumAmount, nMaximumCount, nMinDepth, nMaxDepth, fOnlyConfirmed, fIncludeZeroValue, nCoinType, fUseInstantSend, fUseBDAP);
}

void CWallet::AvailableCoinsWithAssets(std::vector<COutput> &vCoins, std::map<std::string, std::vector<COutput> > &mapAssetCoins,
                              bool fOnlySafe, const CCoinControl* coinControl, const CAmount &nMinimumAmount,
                              const CAmount &nMaximumAmount, const CAmount &nMinimumSumAmount,
                              const uint64_t &nMaximumCount, const int &nMinDepth, const int &nMaxDepth,
                              bool fOnlyConfirmed, bool fIncludeZeroValue, AvailableCoinsType nCoinType, 
                              bool fUseInstantSend, bool fUseBDAP) const
{
    AvailableCoinsAll(vCoins, mapAssetCoins, true, AreAssetsDeployed(), fOnlySafe, coinControl, nMinimumAmount, nMaximumAmount, nMinimumSumAmount, nMaximumCount, nMinDepth, nMaxDepth, fOnlyConfirmed, fIncludeZeroValue, nCoinType, fUseInstantSend, fUseBDAP);
}

void CWallet::AvailableCoinsAll(std::vector<COutput>& vCoins, std::map<std::string, std::vector<COutput> >& mapAssetCoins, bool fGetDYN, bool fGetAssets, bool fOnlySafe, const CCoinControl* coinControl, const CAmount& nMinimumAmount, const CAmount& nMaximumAmount, const CAmount& nMinimumSumAmount, const uint64_t& nMaximumCount, const int& nMinDepth, const int& nMaxDepth, bool fOnlyConfirmed, bool fIncludeZeroValue, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fUseBDAP) const
{
    vCoins.clear();

    {
        LOCK2(cs_main, cs_wallet);
        int nInstantSendConfirmationsRequired = Params().GetConsensus().nInstantSendConfirmationsRequired;

        CAmount nTotal = 0;

        /** ASSET START */
        bool fDYNLimitHit = false;
        // A set of the hashes that have already been used
        std::set<uint256> usedMempoolHashes;

        std::map<std::string, CAmount> mapAssetTotals;
        std::map<uint256, COutPoint> mapOutPoints;
        std::set<std::string> setAssetMaxFound;
        // Turn the OutPoints into a map that is easily interatable.
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const uint256 &wtxid = it->first;
            const CWalletTx *pcoin = &(*it).second;

            if (!CheckFinalTx(*pcoin))
                continue;

            if (fOnlyConfirmed && !pcoin->IsTrusted())
                continue;

            if ((pcoin->IsCoinBase() || pcoin->IsCoinStake()) && pcoin->GetBlocksToMaturity() > 0)
                continue;

            int nDepth = pcoin->GetDepthInMainChain();
            if (fUseInstantSend && nDepth < nInstantSendConfirmationsRequired)
                continue;

            // We should not consider coins which aren't at least in our mempool
            // It's possible for these to be conflicted via ancestors which we may never be able to detect
            if (nDepth == 0 && !pcoin->InMempool())
                continue;

            for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++) {
                bool found = false;
                if (nCoinType == ONLY_DENOMINATED) {
                    found = CPrivateSend::IsDenominatedAmount(pcoin->tx->vout[i].nValue);
                } else if (nCoinType == ONLY_NONDENOMINATED) {
                    if (CPrivateSend::IsCollateralAmount(pcoin->tx->vout[i].nValue))
                        continue; // do not use collateral amounts
                    found = !CPrivateSend::IsDenominatedAmount(pcoin->tx->vout[i].nValue);
                } else if (nCoinType == ONLY_1000) {
                    found = pcoin->tx->vout[i].nValue == 1000 * COIN;
                } else if (nCoinType == ONLY_PRIVATESEND_COLLATERAL) {
                    found = CPrivateSend::IsCollateralAmount(pcoin->tx->vout[i].nValue);
                } else {
                    found = true;
                }
                if (!found)
                    continue;

                bool fIsBDAP = false;
                if (pcoin->tx->vout[i].IsBDAP()) {
                    fIsBDAP = true;
                    int opCode1 = -1; int opCode2 = -1;
                    std::vector<std::vector<unsigned char>> vParameters;
                    pcoin->tx->vout[i].GetBDAPOpCodes(opCode1, opCode2, vParameters);
                    std::string strOpType = GetBDAPOpTypeString(opCode1, opCode2);
                    LogPrint("bdap", "%s -- strOpType %s, vParameters.size %d, nValue %d\n", __func__, 
                                strOpType, vParameters.size(), FormatMoney(pcoin->tx->vout[i].nValue));
                    // Only use BDAP credit assets for available coins and filter out all other BDAP outputs
                    if (!(strOpType == "bdap_move_asset"))
                        continue;

                    std::vector<unsigned char> vchMoveSource = vchFromString(std::string("DYN"));
                    std::vector<unsigned char> vchMoveDestination = vchFromString(std::string("BDAP"));
                    if (!(vParameters.size() == 2 && vParameters[0] == vchMoveSource && vParameters[1] == vchMoveDestination))
                        continue;
                }

                if (!fUseBDAP && fIsBDAP)
                    continue;

                
                isminetype mine = IsMine(pcoin->tx->vout[i]);
                if (!(IsSpent(wtxid, i)) && mine != ISMINE_NO &&
                    (!IsLockedCoin((*it).first, i) || nCoinType == ONLY_1000) &&
                    (pcoin->tx->vout[i].nValue > 0 || fIncludeZeroValue) &&
                    (!coinControl || !coinControl->HasSelected() || coinControl->fAllowOtherInputs || coinControl->IsSelected(COutPoint((*it).first, i))))
                    vCoins.push_back(COutput(pcoin, i, nDepth,
                        ((mine & ISMINE_SPENDABLE) != ISMINE_NO) ||
                            (coinControl && coinControl->fAllowWatchOnly && (mine & ISMINE_WATCH_SOLVABLE) != ISMINE_NO),
                        (mine & (ISMINE_SPENDABLE | ISMINE_WATCH_SOLVABLE)) != ISMINE_NO, pcoin->IsTrusted()));
            }

            bool safeTx = pcoin->IsTrusted();

            // We should not consider coins from transactions that are replacing
            // other transactions.
            //
            // Example: There is a transaction A which is replaced by bumpfee
            // transaction B. In this case, we want to prevent creation of
            // a transaction B' which spends an output of B.
            //
            // Reason: If transaction A were initially confirmed, transactions B
            // and B' would no longer be valid, so the user would have to create
            // a new transaction C to replace B'. However, in the case of a
            // one-block reorg, transactions B' and C might BOTH be accepted,
            // when the user only wanted one of them. Specifically, there could
            // be a 1-block reorg away from the chain where transactions A and C
            // were accepted to another chain where B, B', and C were all
            // accepted.
            if (nDepth == 0 && pcoin->mapValue.count("replaces_txid")) {
                safeTx = false;
            }

            // Similarly, we should not consider coins from transactions that
            // have been replaced. In the example above, we would want to prevent
            // creation of a transaction A' spending an output of A, because if
            // transaction B were initially confirmed, conflicting with A and
            // A', we wouldn't want to the user to create a transaction D
            // intending to replace A', but potentially resulting in a scenario
            // where A, A', and D could all be accepted (instead of just B and
            // D, or just A and A' like the user would want).
            if (nDepth == 0 && pcoin->mapValue.count("replaced_by_txid")) {
                safeTx = false;
            }

            if (fOnlySafe && !safeTx) {
                continue;
            }

            if (nDepth < nMinDepth || nDepth > nMaxDepth)
                continue;

            for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++) {

                int nType;
                bool fIsOwner;
                bool isAssetScript = pcoin->tx->vout[i].scriptPubKey.IsAssetScript(nType, fIsOwner);
                if (coinControl && !isAssetScript && coinControl->HasSelected() && !coinControl->fAllowOtherInputs && !coinControl->IsSelected(COutPoint((*it).first, i)))
                    continue;

                if (coinControl && isAssetScript && coinControl->HasAssetSelected() && !coinControl->fAllowOtherInputs && !coinControl->IsAssetSelected(COutPoint((*it).first, i)))
                    continue;

                if (IsLockedCoin((*it).first, i))
                    continue;

                if (IsSpent(wtxid, i))
                    continue;

                isminetype mine = IsMine(pcoin->tx->vout[i]);

                if (mine == ISMINE_NO) {
                    continue;
                }

                bool fSpendableIn = ((mine & ISMINE_SPENDABLE) != ISMINE_NO) ||
                                    (coinControl && coinControl->fAllowWatchOnly &&
                                     (mine & ISMINE_WATCH_SOLVABLE) != ISMINE_NO);
                bool fSolvableIn = (mine & (ISMINE_SPENDABLE | ISMINE_WATCH_SOLVABLE)) != ISMINE_NO;

                std::string address;

                // Looking for Asset Tx OutPoints Only
                if (fGetAssets && AreAssetsDeployed() && isAssetScript) {

                    CAssetOutputEntry output_data;
                    if (!GetAssetData(pcoin->tx->vout[i].scriptPubKey, output_data))
                        continue;

                    address = EncodeDestination(output_data.destination);

                    // If we already have the maximum amount or size for this asset, skip it
                    if (setAssetMaxFound.count(output_data.assetName))
                        continue;

                    if (IsAssetNameAnRestricted(output_data.assetName)) {
                        if (passets->CheckForAddressRestriction(output_data.assetName, address, true)) {
                            continue;
                        }
                    }

                    // Initialize the map vector is it doesn't exist yet
                    if (!mapAssetCoins.count(output_data.assetName)) {
                        std::vector<COutput> vOutput;
                        mapAssetCoins.insert(std::make_pair(output_data.assetName, vOutput));
                    }

                    // Add the COutput to the map of available Asset Coins
                    mapAssetCoins.at(output_data.assetName).push_back(
                            COutput(pcoin, i, nDepth, fSpendableIn, fSolvableIn, safeTx));

                    // Initialize the map of current asset totals
                    if (!mapAssetTotals.count(output_data.assetName))
                        mapAssetTotals[output_data.assetName] = 0;

                    // Update the map of totals depending the which type of asset tx we are looking at
                    mapAssetTotals[output_data.assetName] += output_data.nAmount;

                    // Checks the sum amount of all UTXO's, and adds to the set of assets that we found the max for
                    if (nMinimumSumAmount != MAX_MONEY) {
                        if (mapAssetTotals[output_data.assetName] >= nMinimumSumAmount)
                            setAssetMaxFound.insert(output_data.assetName);
                    }

                    // Checks the maximum number of UTXO's, and addes to set of of asset that we found the max for
                    if (nMaximumCount > 0 && mapAssetCoins[output_data.assetName].size() >= nMaximumCount) {
                        setAssetMaxFound.insert(output_data.assetName);
                    }
                }

                if (fGetDYN) { // Looking for DYN Tx OutPoints Only
                    if (fDYNLimitHit) // We hit our limit
                        continue;

                    // We only want DYN OutPoints. Don't include Asset OutPoints
                    if (isAssetScript)
                        continue;

                    vCoins.push_back(COutput(pcoin, i, nDepth, fSpendableIn, fSolvableIn, safeTx));

                    // Checks the sum amount of all UTXO's.
                    if (nMinimumSumAmount != MAX_MONEY) {
                        nTotal += pcoin->tx->vout[i].nValue;

                        if (nTotal >= nMinimumSumAmount) {
                            fDYNLimitHit = true;
                        }
                    }

                    // Checks the maximum number of UTXO's.
                    if (nMaximumCount > 0 && vCoins.size() >= nMaximumCount) {
                        fDYNLimitHit = true;
                    }
                    continue;
                }
            }
        }
        /** ASSET END */
    }
}

void CWallet::AvailableBDAPCredits(std::vector<std::pair<CTxOut, COutPoint>>& vCredits, bool fOnlyConfirmed) const
{
    vCredits.clear();

    {
        LOCK2(cs_main, cs_wallet);

        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const uint256& wtxid = it->first;
            const CWalletTx* pcoin = &(*it).second;

            if (!CheckFinalTx(*pcoin))
                continue;

            if (fOnlyConfirmed && !pcoin->IsTrusted())
                continue;

            if (pcoin->IsCoinBase() && pcoin->GetBlocksToMaturity() > 0)
                continue;

            int nDepth = pcoin->GetDepthInMainChain();
            // We should not consider coins which aren't at least in our mempool
            // It's possible for these to be conflicted via ancestors which we may never be able to detect
            if (nDepth == 0 && !pcoin->InMempool())
                continue;

            for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++) {
                if (!pcoin->tx->vout[i].IsBDAP())
                    continue;

                if (fOnlyConfirmed && pcoin->tx->vout[i].nValue == 0)
                    continue;

                isminetype mine = IsMine(pcoin->tx->vout[i]);
                if (!(IsSpent(wtxid, i)) && mine != ISMINE_NO)
                    vCredits.push_back(std::make_pair(pcoin->tx->vout[i], COutPoint(pcoin->tx->GetHash(), i)));
            }
        }
    }
}

std::map<CDynamicAddress, std::vector<COutput> > CWallet::AvailableCoinsByAddress(bool fConfirmed, CAmount maxCoinValue)
{
    std::vector<COutput> vCoins;
    AvailableCoins(vCoins, fConfirmed);

    std::map<CDynamicAddress, std::vector<COutput> > mapCoins;
    for (COutput out : vCoins) {
        if (maxCoinValue > 0 && out.tx->tx->vout[out.i].nValue > maxCoinValue)
            continue;

        CTxDestination address;
        if (!ExtractDestination(out.tx->tx->vout[out.i].scriptPubKey, address))
            continue;

        mapCoins[CDynamicAddress(address)].push_back(out);
    }

    return mapCoins;
}
/** ASSET START */
std::map<CTxDestination, std::vector<COutput>> CWallet::ListAssets() const
{
    // TODO: Add AssertLockHeld(cs_wallet) here.
    //
    // Because the return value from this function contains pointers to
    // CWalletTx objects, callers to this function really should acquire the
    // cs_wallet lock before calling it. However, the current caller doesn't
    // acquire this lock yet. There was an attempt to add the missing lock in
    // https://github.com/RavenProject/Ravencoin/pull/10340, but that change has been
    // postponed until after https://github.com/RavenProject/Ravencoin/pull/10244 to
    // avoid adding some extra complexity to the Qt code.

    std::map<CTxDestination, std::vector<COutput>> result;

    std::map<std::string, std::vector<COutput> > mapAssets;
    AvailableAssets(mapAssets);

    LOCK2(cs_main, cs_wallet);
    for (auto asset : mapAssets) {
        for (auto &coin : asset.second) {
            CTxDestination address;
            if (coin.fSpendable &&
                ExtractDestination(FindNonChangeParentOutput(*coin.tx->tx, coin.i).scriptPubKey, address)) {
                result[address].emplace_back(std::move(coin));
            }
        }
    }

    std::vector<COutPoint> lockedCoins;
    ListLockedCoins(lockedCoins);
    for (const auto& output : lockedCoins) {
        auto it = mapWallet.find(output.hash);
        if (it != mapWallet.end()) {
            if (!it->second.tx->vout[output.n].scriptPubKey.IsAssetScript()) // If not an asset script skip it
                continue;
            int depth = it->second.GetDepthInMainChain();
            if (depth >= 0 && output.n < it->second.tx->vout.size() &&
                IsMine(it->second.tx->vout[output.n]) == ISMINE_SPENDABLE) {
                CTxDestination address;
                if (ExtractDestination(FindNonChangeParentOutput(*it->second.tx, output.n).scriptPubKey, address)) {
                    result[address].emplace_back(
                        &it->second, output.n, depth, true /* spendable */, true /* solvable */, false /* safe */);
                }
            }
        }
    }

    return result;
}

std::map<CTxDestination, std::vector<COutput>> CWallet::ListCoins() const
{
    // TODO: Add AssertLockHeld(cs_wallet) here.
    //
    // Because the return value from this function contains pointers to
    // CWalletTx objects, callers to this function really should acquire the
    // cs_wallet lock before calling it. However, the current caller doesn't
    // acquire this lock yet. There was an attempt to add the missing lock in
    // https://github.com/RavenProject/Ravencoin/pull/10340, but that change has been
    // postponed until after https://github.com/RavenProject/Ravencoin/pull/10244 to
    // avoid adding some extra complexity to the Qt code.

    std::map<CTxDestination, std::vector<COutput>> result;

    std::vector<COutput> availableCoins;
    AvailableCoins(availableCoins);

    LOCK2(cs_main, cs_wallet);
    for (auto& coin : availableCoins) {
        CTxDestination address;
        if (coin.fSpendable &&
            ExtractDestination(FindNonChangeParentOutput(*coin.tx->tx, coin.i).scriptPubKey, address)) {
            result[address].emplace_back(std::move(coin));
        }
    }

    std::vector<COutPoint> lockedCoins;
    ListLockedCoins(lockedCoins);
    for (const auto& output : lockedCoins) {
        auto it = mapWallet.find(output.hash);
        if (it != mapWallet.end()) {
            int depth = it->second.GetDepthInMainChain();
            if (depth >= 0 && output.n < it->second.tx->vout.size() &&
                IsMine(it->second.tx->vout[output.n]) == ISMINE_SPENDABLE) {
                CTxDestination address;
                if (ExtractDestination(FindNonChangeParentOutput(*it->second.tx, output.n).scriptPubKey, address)) {
                    result[address].emplace_back(
                            &it->second, output.n, depth, true /* spendable */, true /* solvable */, false /* safe */);
                }
            }
        }
    }

    return result;
}

const CTxOut& CWallet::FindNonChangeParentOutput(const CTransaction& tx, int output) const
{
    const CTransaction* ptx = &tx;
    int n = output;
    while (IsChange(ptx->vout[n]) && ptx->vin.size() > 0) {
        const COutPoint& prevout = ptx->vin[0].prevout;
        auto it = mapWallet.find(prevout.hash);
        if (it == mapWallet.end() || it->second.tx->vout.size() <= prevout.n ||
            !IsMine(it->second.tx->vout[prevout.n])) {
            break;
        }
        ptx = it->second.tx.get();
        n = prevout.n;
    }
    return ptx->vout[n];
}

static void ApproximateBestAssetSubset(const std::vector<std::pair<CInputCoin, CAmount> >& vValue, const CAmount& nTotalLower, const CAmount& nTargetValue,
                                  std::vector<char>& vfBest, CAmount& nBest, int iterations = 1000)
{
    std::vector<char> vfIncluded;

    vfBest.assign(vValue.size(), true);
    nBest = nTotalLower;

    FastRandomContext insecure_rand;

    for (int nRep = 0; nRep < iterations && nBest != nTargetValue; nRep++)
    {
        vfIncluded.assign(vValue.size(), false);
        CAmount nTotal = 0;
        bool fReachedTarget = false;
        for (int nPass = 0; nPass < 2 && !fReachedTarget; nPass++)
        {
            for (unsigned int i = 0; i < vValue.size(); i++)
            {
                //The solver here uses a randomized algorithm,
                //the randomness serves no real security purpose but is just
                //needed to prevent degenerate behavior and it is important
                //that the rng is fast. We do not use a constant random sequence,
                //because there may be some privacy improvement by making
                //the selection random.
                if (nPass == 0 ? insecure_rand.randbool() : !vfIncluded[i])
                {
                    nTotal += vValue[i].second;
                    vfIncluded[i] = true;
                    if (nTotal >= nTargetValue)
                    {
                        fReachedTarget = true;
                        if (nTotal < nBest)
                        {
                            nBest = nTotal;
                            vfBest = vfIncluded;
                        }
                        nTotal -= vValue[i].second;
                        vfIncluded[i] = false;
                    }
                }
            }
        }
    }
}
/** ASSET END */
static void ApproximateBestSubset(const std::vector<CInputCoin>& vValue, const CAmount& nTotalLower, const CAmount& nTargetValue, std::vector<char>& vfBest, CAmount& nBest, bool fUseInstantSend = false, int iterations = 1000)
{
    std::vector<char> vfIncluded;

    vfBest.assign(vValue.size(), true);
    nBest = nTotalLower;

    FastRandomContext insecure_rand;

    for (int nRep = 0; nRep < iterations && nBest != nTargetValue; nRep++) {
        vfIncluded.assign(vValue.size(), false);
        CAmount nTotal = 0;
        bool fReachedTarget = false;
        for (int nPass = 0; nPass < 2 && !fReachedTarget; nPass++) {
            for (unsigned int i = 0; i < vValue.size(); i++) {
                if (fUseInstantSend && nTotal + vValue[i].txout.nValue > sporkManager.GetSporkValue(SPORK_5_INSTANTSEND_MAX_VALUE) * COIN) {
                    continue;
                }
                //The solver here uses a randomized algorithm,
                //the randomness serves no real security purpose but is just
                //needed to prevent degenerate behavior and it is important
                //that the rng is fast. We do not use a constant random sequence,
                //because there may be some privacy improvement by making
                //the selection random.
                if (nPass == 0 ? insecure_rand.rand32() & 1 : !vfIncluded[i]) {
                    nTotal += vValue[i].txout.nValue;
                    vfIncluded[i] = true;
                    if (nTotal >= nTargetValue) {
                        fReachedTarget = true;
                        if (nTotal < nBest) {
                            nBest = nTotal;
                            vfBest = vfIncluded;
                        }
                        nTotal -= vValue[i].txout.nValue;
                        vfIncluded[i] = false;
                    }
                }
            }
        }
    }
}

struct CompareByPriority {
    bool operator()(const COutput& t1,
        const COutput& t2) const
    {
        return t1.Priority() > t2.Priority();
    }
};

// move denoms down
bool less_then_denom(const COutput& out1, const COutput& out2)
{
    const CWalletTx* pcoin1 = out1.tx;
    const CWalletTx* pcoin2 = out2.tx;

    bool found1 = false;
    bool found2 = false;
    for (CAmount d : CPrivateSend::GetStandardDenominations()) // loop through predefined denoms
    {
        if (pcoin1->tx->vout[out1.i].nValue == d)
            found1 = true;
        if (pcoin2->tx->vout[out2.i].nValue == d)
            found2 = true;
    }
    return (!found1 && found2);
}

bool CWallet::SelectCoinsMinConf(const CAmount& nTargetValue, const int nConfMine, const int nConfTheirs, const uint64_t nMaxAncestors, std::vector<COutput> vCoins,
                                 std::set<CInputCoin>& setCoinsRet, CAmount& nValueRet, AvailableCoinsType nCoinType, bool fUseInstantSend) const
{
    setCoinsRet.clear();
    nValueRet = 0;

    // List of values less than target
    boost::optional<CInputCoin> coinLowestLarger;
    std::vector<CInputCoin> vValue;
    CAmount nTotalLower = 0;

    random_shuffle(vCoins.begin(), vCoins.end(), GetRandInt);

    int tryDenomStart = 0;
    CAmount nMinChange = MIN_CHANGE;

    if (nCoinType == ONLY_DENOMINATED) {
        // larger denoms first
        std::sort(vCoins.rbegin(), vCoins.rend(), CompareByPriority());
        // we actually want denoms only, so let's skip "non-denom only" step
        tryDenomStart = 1;
        // no change is allowed
        nMinChange = 0;
    } else {
        // move denoms down on the list
        // try not to use denominated coins when not needed, save denoms for privatesend
        std::sort(vCoins.begin(), vCoins.end(), less_then_denom);
    }

    // try to find nondenom first to prevent unneeded spending of mixed coins
    for (unsigned int tryDenom = tryDenomStart; tryDenom < 2; tryDenom++)
    {
        LogPrint("selectcoins", "tryDenom: %d\n", tryDenom);
        vValue.clear();
        nTotalLower = 0;
        for (const COutput &output : vCoins)
        {
            if (!output.fSpendable)
                continue;

            const CWalletTx *pcoin = output.tx;

//            if (fDebug) LogPrint("selectcoins", "value %s confirms %d\n", FormatMoney(pcoin->vout[output.i].nValue), output.nDepth);
            int minDepth;
            if (!pcoin->IsRelevantToMe(ISMINE_ALL))
                minDepth = nConfTheirs;
            else if (pcoin->IsFromMe(ISMINE_ALL))
                minDepth = nConfMine;
            else
                minDepth = std::max(nConfMine, nConfTheirs);

            if (output.nDepth < minDepth)
                continue;

            if (!mempool.TransactionWithinChainLimit(pcoin->GetHash(), nMaxAncestors))
                continue;

            int i = output.i;
            CInputCoin coin = CInputCoin(pcoin, i);
            if (tryDenom == 0 && CPrivateSend::IsDenominatedAmount(coin.txout.nValue)) continue; // we don't want denom values on first run

            if (nCoinType == ONLY_DENOMINATED) {
                // Make sure it's actually anonymized
                COutPoint outpoint = COutPoint(pcoin->GetHash(), i);
                int nRounds = GetRealOutpointPrivateSendRounds(outpoint);
                if (nRounds < privateSendClient.nPrivateSendRounds) continue;
            }

            if (coin.txout.nValue == nTargetValue) 
            {
                setCoinsRet.insert(coin);
                nValueRet += coin.txout.nValue;
                return true;
            }
            else if (coin.txout.nValue < nTargetValue + nMinChange)
            {
                vValue.push_back(coin);
                nTotalLower += coin.txout.nValue;
            }
            else if (!coinLowestLarger || coin.txout.nValue < coinLowestLarger->txout.nValue)
            {
                coinLowestLarger = coin;
            }
        }

        if (nTotalLower == nTargetValue)
        {
            for (unsigned int i = 0; i < vValue.size(); ++i)
            {
                setCoinsRet.insert(vValue[i]);
                nValueRet += vValue[i].txout.nValue;
            }
            return true;
        }

        if (nTotalLower < nTargetValue)
        {
            if (!coinLowestLarger) // there is no input larger than nTargetValue
            {
                if (tryDenom == 0)
                    // we didn't look at denom yet, let's do it
                    continue;
                else
                    // we looked at everything possible and didn't find anything, no luck
                    return false;
            }
            setCoinsRet.insert(coinLowestLarger.get());
            nValueRet += coinLowestLarger->txout.nValue;
            // There is no change in PS, so we know the fee beforehand,
            // can see if we exceeded the max fee and thus fail quickly.
            return nCoinType == ONLY_DENOMINATED ? (nValueRet - nTargetValue <= maxTxFee) : true;
        }

        // nTotalLower > nTargetValue
        break;

    }

    // Solve subset sum by stochastic approximation
    std::sort(vValue.begin(), vValue.end(), CompareValueOnly());
    std::reverse(vValue.begin(), vValue.end());
    std::vector<char> vfBest;
    CAmount nBest;

    ApproximateBestSubset(vValue, nTotalLower, nTargetValue, vfBest, nBest, fUseInstantSend);
    if (nBest != nTargetValue && nTotalLower >= nTargetValue + nMinChange)
        ApproximateBestSubset(vValue, nTotalLower, nTargetValue + nMinChange, vfBest, nBest, fUseInstantSend);

    // If we have a bigger coin and (either the stochastic approximation didn't find a good solution,
    //                                   or the next bigger coin is closer), return the bigger coin
    if (coinLowestLarger &&
        ((nBest != nTargetValue && nBest < nTargetValue + nMinChange) || coinLowestLarger->txout.nValue <= nBest))
    {
        setCoinsRet.insert(coinLowestLarger.get());
        nValueRet += coinLowestLarger->txout.nValue;
    }
    else {
        std::string s = "CWallet::SelectCoinsMinConf best subset: ";
        for (unsigned int i = 0; i < vValue.size(); i++)
        {
            if (vfBest[i])
            {
                setCoinsRet.insert(vValue[i]);
                nValueRet += vValue[i].txout.nValue;
                s += FormatMoney(vValue[i].txout.nValue) + " ";
            }
        }
        LogPrint("selectcoins", "%s - total %s\n", s, FormatMoney(nBest));
    }

    // There is no change in PS, so we know the fee beforehand,
    // can see if we exceeded the max fee and thus fail quickly.
    return nCoinType == ONLY_DENOMINATED ? (nValueRet - nTargetValue <= maxTxFee) : true;
}

bool CWallet::SelectCoins(const std::vector<COutput>& vAvailableCoins, const CAmount& nTargetValue, std::set<CInputCoin>& setCoinsRet, CAmount& nValueRet, const CCoinControl* coinControl, AvailableCoinsType nCoinType, bool fUseInstantSend) const
{
    // Note: this function should never be used for "always free" tx types like pstx

    std::vector<COutput> vCoins(vAvailableCoins);

    // coin control -> return all selected outputs (we want all selected to go into the transaction for sure)
    if (coinControl && coinControl->HasSelected() && !coinControl->fAllowOtherInputs)
    {
        for(const COutput& out : vCoins)
        {
            if(!out.fSpendable)
                continue;

            if(nCoinType == ONLY_DENOMINATED) {
                COutPoint outpoint = COutPoint(out.tx->GetHash(),out.i);
                int nRounds = GetOutpointPrivateSendRounds(outpoint);
                // make sure it's actually anonymized
                if(nRounds < privateSendClient.nPrivateSendRounds) continue;
            }
            nValueRet += out.tx->tx->vout[out.i].nValue;
            setCoinsRet.insert(CInputCoin(out.tx, out.i));
        }

        return (nValueRet >= nTargetValue);
    }

    // calculate value from preset inputs and store them
    std::set<CInputCoin> setPresetCoins;
    CAmount nValueFromPresetInputs = 0;

    std::vector<COutPoint> vPresetInputs;
    if (coinControl)
        coinControl->ListSelected(vPresetInputs);
    for(const COutPoint& outpoint : vPresetInputs)
    {
        std::map<uint256, CWalletTx>::const_iterator it = mapWallet.find(outpoint.hash);
        if (it != mapWallet.end())
        {
            const CWalletTx* pcoin = &it->second;
            // Clearly invalid input, fail
            if (pcoin->tx->vout.size() <= outpoint.n)
                return false;
            if (nCoinType == ONLY_DENOMINATED) {
                // Make sure to include anonymized preset inputs only,
                // even if some non-anonymized inputs were manually selected via CoinControl
                int nRounds = GetRealOutpointPrivateSendRounds(outpoint);
                if (nRounds < privateSendClient.nPrivateSendRounds) continue;
            }
            nValueFromPresetInputs += pcoin->tx->vout[outpoint.n].nValue;
            setPresetCoins.insert(CInputCoin(pcoin, outpoint.n));
        } else
            return false; // TODO: Allow non-wallet inputs
    }

    // remove preset inputs from vCoins
    for (std::vector<COutput>::iterator it = vCoins.begin(); it != vCoins.end() && coinControl && coinControl->HasSelected();)
    {
        if (setPresetCoins.count(CInputCoin(it->tx, it->i)))
            it = vCoins.erase(it);
        else
            ++it;
    }

    size_t nMaxChainLength = std::min(gArgs.GetArg("-limitancestorcount", DEFAULT_ANCESTOR_LIMIT), gArgs.GetArg("-limitdescendantcount", DEFAULT_DESCENDANT_LIMIT));
    bool fRejectLongChains = gArgs.GetBoolArg("-walletrejectlongchains", DEFAULT_WALLET_REJECT_LONG_CHAINS);

    bool res = nTargetValue <= nValueFromPresetInputs ||
        SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 1, 6, 0, vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend) ||
        SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 1, 1, 0, vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend) ||
        (bSpendZeroConfChange && SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 0, 1, 2, vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend)) ||
        (bSpendZeroConfChange && SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 0, 1, std::min((size_t)4, nMaxChainLength/3), vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend)) ||
        (bSpendZeroConfChange && SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 0, 1, nMaxChainLength/2, vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend)) ||
        (bSpendZeroConfChange && SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 0, 1, nMaxChainLength, vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend)) ||
        (bSpendZeroConfChange && !fRejectLongChains && SelectCoinsMinConf(nTargetValue - nValueFromPresetInputs, 0, 1, std::numeric_limits<uint64_t>::max(), vCoins, setCoinsRet, nValueRet, nCoinType, fUseInstantSend));

    // because SelectCoinsMinConf clears the setCoinsRet, we now add the possible inputs to the coinset
    setCoinsRet.insert(setPresetCoins.begin(), setPresetCoins.end());

    // add preset inputs to the total value selected
    nValueRet += nValueFromPresetInputs;

    return res;
}


/** ASSET START */
bool CWallet::CreateNewChangeAddress(CReserveKey& reservekey, CKeyID& keyID, std::string& strFailReason)
{
    // Called with coin control doesn't have a change_address
    // no coin control: send change to newly generated address
    // Note: We use a new key here to keep it from being obvious which side is the change.
    //  The drawback is that by not reusing a previous key, the change may be lost if a
    //  backup is restored, if the backup doesn't have the new private key for the change.
    //  If we reused the old key, it would be possible to add code to look for and
    //  rediscover unknown transactions that were written with keys of ours to recover
    //  post-backup change.

    // Reserve a new key pair from key pool
    CPubKey vchPubKey;
    bool ret;
    ret = reservekey.GetReservedKey(vchPubKey, true);
    if (!ret)
    {
        strFailReason = _("Keypool ran out, please call keypoolrefill first");
        return false;
    }

    keyID = vchPubKey.GetID();
    return true;
}

bool CWallet::SelectAssetsMinConf(const CAmount& nTargetValue, const int nConfMine, const int nConfTheirs, const uint64_t nMaxAncestors, const std::string& strAssetName, std::vector<COutput> vCoins,
                                 std::set<CInputCoin>& setCoinsRet, CAmount& nValueRet) const
{
    setCoinsRet.clear();
    nValueRet = 0;

    // List of values less than target
    boost::optional<CInputCoin> coinLowestLarger;
    boost::optional<CAmount> coinLowestLargerAmount;
    std::vector<std::pair<CInputCoin, CAmount> > vValue;
    std::map<COutPoint, CAmount> mapValueAmount;
    CAmount nTotalLower = 0;

    random_shuffle(vCoins.begin(), vCoins.end(), GetRandInt);
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    for (const COutput &output : vCoins)
    {
        if (!output.fSpendable)
            continue;

        const CWalletTx *pcoin = output.tx;

        if (output.nDepth < (pcoin->IsFromMe(ISMINE_ALL) ? nConfMine : nConfTheirs))
            continue;

        if (!mempool.TransactionWithinChainLimit(pcoin->GetHash(), nMaxAncestors))
            continue;

        int i = output.i;

        CInputCoin coin = CInputCoin(pcoin, i);

        //-------------------------------

        int nType = -1;
        bool fIsOwner = false;
        if (!coin.txout.scriptPubKey.IsAssetScript(nType, fIsOwner)) {
            continue;
        }

        CAmount nTempAmount = 0;
        if (nType == TX_NEW_ASSET && !fIsOwner) { // Root/Sub Asset
            CNewAsset assetTemp;
            std::string address;
            if (!AssetFromScript(coin.txout.scriptPubKey, assetTemp, address))
                continue;
            nTempAmount = assetTemp.nAmount;
        } else if (nType == TX_TRANSFER_ASSET) { // Transfer Asset
            CAssetTransfer transferTemp;
            std::string address;
            if (!TransferAssetFromScript(coin.txout.scriptPubKey, transferTemp, address))
                continue;
            nTempAmount = transferTemp.nAmount;
        } else if (nType == TX_NEW_ASSET && fIsOwner) { // Owner Asset
            std::string ownerName;
            std::string address;
            if (!OwnerAssetFromScript(coin.txout.scriptPubKey, ownerName, address))
                continue;
            nTempAmount = OWNER_ASSET_AMOUNT;
        } else if (nType == TX_REISSUE_ASSET) { // Reissue Asset
            CReissueAsset reissueTemp;
            std::string address;
            if (!ReissueAssetFromScript(coin.txout.scriptPubKey, reissueTemp, address))
                continue;
            nTempAmount = reissueTemp.nAmount;
        } else {
            continue;
        }

        if (nTempAmount == nTargetValue)
        {
            setCoinsRet.insert(coin);
            nValueRet += nTempAmount;
            return true;
        }
        else if (nTempAmount < nTargetValue + MIN_CHANGE)
        {
            vValue.push_back(std::make_pair(coin, nTempAmount));
            nTotalLower += nTempAmount;
        }
        else if (!coinLowestLarger || !coinLowestLargerAmount || nTempAmount < coinLowestLargerAmount)
        {
            coinLowestLarger = coin;
            coinLowestLargerAmount = nTempAmount;
        }
    }

    if (nTotalLower == nTargetValue)
    {
        for (const auto& pair : vValue)
        {
            setCoinsRet.insert(pair.first);
            nValueRet += pair.second;
        }
        return true;
    }

    if (nTotalLower < nTargetValue)
    {
        if (!coinLowestLarger || !coinLowestLargerAmount)
            return false;
        setCoinsRet.insert(coinLowestLarger.get());

        #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        nValueRet += coinLowestLargerAmount.get();
        return true;
    }

    // Solve subset sum by stochastic approximation
    std::sort(vValue.begin(), vValue.end(), CompareAssetValueOnly());
    std::reverse(vValue.begin(), vValue.end());
    std::vector<char> vfBest;
    CAmount nBest;

    ApproximateBestAssetSubset(vValue, nTotalLower, nTargetValue, vfBest, nBest);
    if (nBest != nTargetValue && nTotalLower >= nTargetValue + MIN_CHANGE)
        ApproximateBestAssetSubset(vValue, nTotalLower, nTargetValue + MIN_CHANGE, vfBest, nBest);

    // If we have a bigger coin and (either the stochastic approximation didn't find a good solution,
    //                                   or the next bigger coin is closer), return the bigger coin
    if (coinLowestLarger && coinLowestLargerAmount &&
        ((nBest != nTargetValue && nBest < nTargetValue + MIN_CHANGE) || coinLowestLargerAmount <= nBest))
    {
        setCoinsRet.insert(coinLowestLarger.get());
        nValueRet += coinLowestLargerAmount.get();
    }
    else {
        for (unsigned int i = 0; i < vValue.size(); i++)
            if (vfBest[i])
            {
                setCoinsRet.insert(vValue[i].first);
                nValueRet += vValue[i].second;
            }

        if (LogAcceptCategory("SelectCoins")) {
            LogPrint("SelectCoins", "SelectAssets() best subset: ");
            for (unsigned int i = 0; i < vValue.size(); i++) {
                if (vfBest[i]) {
                    LogPrint("SelectCoins", "%s : %s", strAssetName, FormatMoney(vValue[i].second));
                }
            }
            LogPrint("SelectCoins", "total %s : %s\n", strAssetName, FormatMoney(nBest));
        }
    }

    return true;
}


bool CWallet::SelectAssets(const std::map<std::string, std::vector<COutput> >& mapAvailableAssets, const std::map<std::string, CAmount>& mapAssetTargetValue, std::set<CInputCoin>& setCoinsRet, std::map<std::string, CAmount>& mapValueRet) const
{
    if (!AreAssetsDeployed())
        return false;

    size_t nMaxChainLength = std::min(gArgs.GetArg("-limitancestorcount", DEFAULT_ANCESTOR_LIMIT), gArgs.GetArg("-limitdescendantcount", DEFAULT_DESCENDANT_LIMIT));
    bool fRejectLongChains = gArgs.GetBoolArg("-walletrejectlongchains", DEFAULT_WALLET_REJECT_LONG_CHAINS);

    for (auto assetVector : mapAvailableAssets) {
        // Setup temporay variables
        std::vector<COutput> vAssets(assetVector.second);

        std::set<CInputCoin> tempCoinsRet;
        CAmount nTempAmountRet;
        CAmount nTempTargetValue;
        std::string strAssetName = assetVector.first;

        CAmount nValueFromPresetInputs = 0; // This is used with coincontrol, which assets doesn't support yet

        // If we dont have a target value for this asset, don't select coins for it
        if (!mapAssetTargetValue.count(strAssetName))
            continue;

        // If we dont have a target value greater than zero, don't select coins for it
        if (mapAssetTargetValue.at(strAssetName) <= 0)
            continue;

        // Add the starting value into the mapValueRet
        if (!mapValueRet.count(strAssetName))
            mapValueRet.insert(std::make_pair(strAssetName, 0));

        // assign our temporary variable
        nTempAmountRet = mapValueRet.at(strAssetName);
        nTempTargetValue = mapAssetTargetValue.at(strAssetName);

        bool res = nTempTargetValue <= nValueFromPresetInputs ||
                   SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 1, 6, 0, strAssetName, vAssets, tempCoinsRet, nTempAmountRet) ||
                   SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 1, 1, 0, strAssetName, vAssets, tempCoinsRet, nTempAmountRet) ||
                   (bSpendZeroConfChange && SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 0, 1, 2, strAssetName, vAssets, tempCoinsRet, nTempAmountRet)) ||
                   (bSpendZeroConfChange && SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 0, 1, std::min((size_t)4, nMaxChainLength/3), strAssetName, vAssets, tempCoinsRet, nTempAmountRet)) ||
                   (bSpendZeroConfChange && SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 0, 1, nMaxChainLength/2, strAssetName, vAssets, tempCoinsRet, nTempAmountRet)) ||
                   (bSpendZeroConfChange && SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 0, 1, nMaxChainLength, strAssetName, vAssets, tempCoinsRet, nTempAmountRet)) ||
                   (bSpendZeroConfChange && !fRejectLongChains && SelectAssetsMinConf(nTempTargetValue - nValueFromPresetInputs, 0, 1, std::numeric_limits<uint64_t>::max(), strAssetName, vAssets, tempCoinsRet, nTempAmountRet));

        if (res) {
            setCoinsRet.insert(tempCoinsRet.begin(), tempCoinsRet.end());
            mapValueRet.at(strAssetName) = nTempAmountRet + nValueFromPresetInputs;
        } else {
            return false;
        }
    }

    return true;
}
/** ASSET END */

bool CWallet::SignTransaction(CMutableTransaction &tx)
{
    AssertLockHeld(cs_wallet); // mapWallet

    // sign the new tx
    CTransaction txNewConst(tx);
    int nIn = 0;
    for (const auto& input : tx.vin) {
        std::map<uint256, CWalletTx>::const_iterator mi = mapWallet.find(input.prevout.hash);
        if(mi == mapWallet.end() || input.prevout.n >= mi->second.tx->vout.size()) {
            return false;
        }
        const CScript scriptPubKey = mi->second.tx->vout[input.prevout.n].scriptPubKey;
        const CAmount amount = mi->second.tx->vout[input.prevout.n].nValue;
        SignatureData sigdata;
        if (!ProduceSignature(TransactionSignatureCreator(this, &txNewConst, nIn, amount, SIGHASH_ALL), scriptPubKey, sigdata)) {
            return false;
        }
        UpdateTransaction(tx, nIn, sigdata);
        nIn++;
    }
    return true;
}

bool CWallet::FundTransaction(CMutableTransaction& tx, CAmount& nFeeRet, int& nChangePosInOut, std::string& strFailReason, bool includeWatching, bool lockUnspents, const std::set<int>& setSubtractFeeFromOutputs, CCoinControl coinControl)
{
    std::vector<CRecipient> vecSend;

    // Turn the txout set into a CRecipient vector
    for (size_t idx = 0; idx < tx.vout.size(); idx++) {
        const CTxOut& txOut = tx.vout[idx];
        CRecipient recipient = {txOut.scriptPubKey, txOut.nValue, setSubtractFeeFromOutputs.count(idx) == 1};
        vecSend.push_back(recipient);
    }

    coinControl.fAllowOtherInputs = true;
    coinControl.fAllowWatchOnly = includeWatching;

    for (const CTxIn& txin : tx.vin)
        coinControl.Select(txin.prevout);

    CReserveKey reservekey(this);
    CWalletTx wtx;
    if (!CreateTransaction(vecSend, wtx, reservekey, nFeeRet, nChangePosInOut, strFailReason, coinControl, false))
        return false;

    if (nChangePosInOut != -1)
        tx.vout.insert(tx.vout.begin() + nChangePosInOut, wtx.tx->vout[nChangePosInOut]);

    // Copy output sizes from new transaction; they may have had the fee subtracted from them
    for (unsigned int idx = 0; idx < tx.vout.size(); idx++) {
        // copy entire txout instead of just the value to prevent destination/amount misalignment
        tx.vout[idx] = wtx.tx->vout[idx];
    }

    // Add new txins (keeping original txin scriptSig/order)
    for (const CTxIn& txin : wtx.tx->vin) {
        if (!coinControl.IsSelected(txin.prevout)) {
            tx.vin.push_back(txin);

            if (lockUnspents) {
                LOCK2(cs_main, cs_wallet);
                LockCoin(txin.prevout);
            }
        }
    }

    return true;
}

bool CWallet::SelectPSInOutPairsByDenominations(int nDenom, CAmount nValueMin, CAmount nValueMax, std::vector< std::pair<CTxPSIn, CTxOut> >& vecPSInOutPairsRet)
{
    CAmount nValueTotal{0};
    int nDenomResult{0};

    std::set<uint256> setRecentTxIds;
    std::vector<COutput> vCoins;

    vecPSInOutPairsRet.clear();

    std::vector<int> vecBits;
    if (!CPrivateSend::GetDenominationsBits(nDenom, vecBits)) {
        return false;
    }

    AvailableCoins(vCoins, true, nullptr, false, ONLY_DENOMINATED);
    LogPrintf("CWallet::%s -- vCoins.size(): %d\n", __func__, vCoins.size());

    std::random_shuffle(vCoins.rbegin(), vCoins.rend(), GetRandInt);

    std::vector<CAmount> vecPrivateSendDenominations = CPrivateSend::GetStandardDenominations();
    for (const auto& out : vCoins) {
        uint256 txHash = out.tx->GetHash();
        int nValue = out.tx->tx->vout[out.i].nValue;
        if (setRecentTxIds.find(txHash) != setRecentTxIds.end()) continue; // no duplicate txids
        if (nValueTotal + nValue > nValueMax) continue;

        CTxIn txin = CTxIn(txHash, out.i);
        CScript scriptPubKey = out.tx->tx->vout[out.i].scriptPubKey;
        int nRounds = GetRealOutpointPrivateSendRounds(txin.prevout);
        if (nRounds >= privateSendClient.nPrivateSendRounds) continue;

        for (const auto& nBit : vecBits) {
            if (nValue != vecPrivateSendDenominations[nBit]) continue;
            nValueTotal += nValue;
            vecPSInOutPairsRet.emplace_back(CTxPSIn(txin, scriptPubKey), CTxOut(nValue, scriptPubKey, nRounds));
            setRecentTxIds.emplace(txHash);
            nDenomResult |= 1 << nBit;
            LogPrint("privatesend", "CWallet::%s -- hash: %s, nValue: %d.%08d, nRounds: %d\n",
                            __func__, txHash.ToString(), nValue / COIN, nValue % COIN, nRounds);
        }
    }

    LogPrintf("CWallet::%s -- setRecentTxIds.size(): %d\n", __func__, setRecentTxIds.size());

    return nValueTotal >= nValueMin && nDenom == nDenomResult;
}

bool CWallet::SelectCoinsGroupedByAddresses(std::vector<CompactTallyItem>& vecTallyRet, bool fSkipDenominated, bool fAnonymizable, bool fSkipUnconfirmed, int nMaxOupointsPerAddress) const
{
    LOCK2(cs_main, cs_wallet);

    isminefilter filter = ISMINE_SPENDABLE;

    // try to use cache for already confirmed anonymizable inputs, no cache should be used when the limit is specified
    if(nMaxOupointsPerAddress != -1 && fAnonymizable && fSkipUnconfirmed) {
        if (fSkipDenominated && fAnonymizableTallyCachedNonDenom) {
            vecTallyRet = vecAnonymizableTallyCachedNonDenom;
            LogPrint("selectcoins", "SelectCoinsGroupedByAddresses - using cache for non-denom inputs\n");
            return vecTallyRet.size() > 0;
        }
        if (!fSkipDenominated && fAnonymizableTallyCached) {
            vecTallyRet = vecAnonymizableTallyCached;
            LogPrint("selectcoins", "SelectCoinsGroupedByAddresses - using cache for all inputs\n");
            return vecTallyRet.size() > 0;
        }
    }

    CAmount nSmallestDenom = CPrivateSend::GetSmallestDenomination();

    // Tally
    std::map<CTxDestination, CompactTallyItem> mapTally;
    std::set<uint256> setWalletTxesCounted;
    for (auto& outpoint : setWalletUTXO) {
        if (setWalletTxesCounted.find(outpoint.hash) != setWalletTxesCounted.end())
            continue;
        setWalletTxesCounted.insert(outpoint.hash);

        std::map<uint256, CWalletTx>::const_iterator it = mapWallet.find(outpoint.hash);
        if (it == mapWallet.end())
            continue;

        const CWalletTx& wtx = (*it).second;

        if (wtx.IsCoinBase() && wtx.GetBlocksToMaturity() > 0)
            continue;
        if (fSkipUnconfirmed && !wtx.IsTrusted())
            continue;

        for (unsigned int i = 0; i < wtx.tx->vout.size(); i++) {
            CTxDestination txdest;
            if (!ExtractDestination(wtx.tx->vout[i].scriptPubKey, txdest))
                continue;

            isminefilter mine = ::IsMine(*this, txdest);
            if (!(mine & filter))
                continue;

            auto itTallyItem = mapTally.find(txdest);
            if (nMaxOupointsPerAddress != -1 && itTallyItem != mapTally.end() && (int) itTallyItem->second.vecOutPoints.size() >= nMaxOupointsPerAddress)
                continue;

            if (IsSpent(outpoint.hash, i) || IsLockedCoin(outpoint.hash, i))
                continue;

            if (fSkipDenominated && CPrivateSend::IsDenominatedAmount(wtx.tx->vout[i].nValue))
                continue;

            if (fAnonymizable) {
                // ignore collaterals
                if (CPrivateSend::IsCollateralAmount(wtx.tx->vout[i].nValue))
                    continue;
                if (fDynodeMode && wtx.tx->vout[i].nValue == 1000 * COIN)
                    continue;
                // ignore outputs that are 10 times smaller then the smallest denomination
                // otherwise they will just lead to higher fee / lower priority
                if (wtx.tx->vout[i].nValue <= nSmallestDenom / 10)
                    continue;
                // ignore anonymized
                if (GetOutpointPrivateSendRounds(COutPoint(outpoint.hash, i)) >= privateSendClient.nPrivateSendRounds)
                    continue;
            }

            if (itTallyItem == mapTally.end()) {
                itTallyItem = mapTally.emplace(txdest, CompactTallyItem()).first;
                itTallyItem->second.txdest = txdest;
            }
            itTallyItem->second.nAmount += wtx.tx->vout[i].nValue;
            itTallyItem->second.vecOutPoints.emplace_back(outpoint.hash, i);
        }
    }

    // construct resulting vector
    // NOTE: vecTallyRet is "sorted" by txdest (i.e. address), just like mapTally
    vecTallyRet.clear();
    for (const auto& item : mapTally) {
        if (fAnonymizable && item.second.nAmount < nSmallestDenom)
            continue;
        vecTallyRet.push_back(item.second);
    }

    // cache already confirmed anonymizable entries for later use, no cache should be saved when the limit is specified
    if(nMaxOupointsPerAddress != -1 && fAnonymizable && fSkipUnconfirmed) {
        if (fSkipDenominated) {
            vecAnonymizableTallyCachedNonDenom = vecTallyRet;
            fAnonymizableTallyCachedNonDenom = true;
        } else {
            vecAnonymizableTallyCached = vecTallyRet;
            fAnonymizableTallyCached = true;
        }
    }

    // debug
    if (LogAcceptCategory("selectcoins")) {
        std::string strMessage = "SelectCoinsGroupedByAddresses - vecTallyRet:\n";
        for (const auto& item : vecTallyRet)
            strMessage += strprintf("  %s %f\n", CDynamicAddress(item.txdest).ToString().c_str(), float(item.nAmount) / COIN);
        LogPrint("selectcoins", "%s", strMessage);
    }
    return vecTallyRet.size() > 0;
}

bool CWallet::SelectPrivateCoins(CAmount nValueMin, CAmount nValueMax, std::vector<CTxIn>& vecTxInRet, CAmount& nValueRet, int nPrivateSendRoundsMin, int nPrivateSendRoundsMax) const
{
    CCoinControl* coinControl = nullptr;

    vecTxInRet.clear();
    nValueRet = 0;

    std::vector<COutput> vCoins;
    AvailableCoins(vCoins, true, coinControl, false, nPrivateSendRoundsMin < 0 ? ONLY_NONDENOMINATED : ONLY_DENOMINATED);

    //order the array so largest nondenom are first, then denominations, then very small inputs.
    sort(vCoins.rbegin(), vCoins.rend(), CompareByPriority());

    for (const auto& out : vCoins)
    {
        //do not allow inputs less than 1/10th of minimum value
        if (out.tx->tx->vout[out.i].nValue < nValueMin / 10)
            continue;
        //do not allow collaterals to be selected
        if (CPrivateSend::IsCollateralAmount(out.tx->tx->vout[out.i].nValue))
            continue;
        if (fDynodeMode && out.tx->tx->vout[out.i].nValue == 1000 * COIN)
            continue; //dynode input

        if (nValueRet + out.tx->tx->vout[out.i].nValue <= nValueMax) {
            CTxIn txin = CTxIn(out.tx->GetHash(), out.i);

            int nRounds = GetOutpointPrivateSendRounds(txin.prevout);
            if (nRounds >= nPrivateSendRoundsMax)
                continue;
            if (nRounds < nPrivateSendRoundsMin)
                continue;

            nValueRet += out.tx->tx->vout[out.i].nValue;
            vecTxInRet.push_back(txin);
        }
    }

    return nValueRet >= nValueMin;
}

bool CWallet::GetCollateralTxPSIn(CTxPSIn& txpsinRet, CAmount& nValueRet) const
{
    LOCK2(cs_main, cs_wallet);

    std::vector<COutput> vCoins;

    AvailableCoins(vCoins);

    for (const auto& out : vCoins) {
        if (CPrivateSend::IsCollateralAmount(out.tx->tx->vout[out.i].nValue)) {
            txpsinRet = CTxPSIn(CTxIn(out.tx->tx->GetHash(), out.i), out.tx->tx->vout[out.i].scriptPubKey);
            nValueRet = out.tx->tx->vout[out.i].nValue;
            return true;
        }
    }

    return false;
}

bool CWallet::GetDynodeOutpointAndKeys(COutPoint& outpointRet, CPubKey& pubKeyRet, CKey& keyRet, std::string strTxHash, std::string strOutputIndex)
{
    // wait for reindex and/or import to finish
    if (fImporting || fReindex)
        return false;

    // Find possible candidates
    std::vector<COutput> vPossibleCoins;
    AvailableCoins(vPossibleCoins, true, nullptr, false, ONLY_1000);
    if (vPossibleCoins.empty()) {
        LogPrintf("CWallet::GetDynodeOutpointAndKeys -- Could not locate any valid dynode vin\n");
        return false;
    }

    if (strTxHash.empty()) // No output specified, select the first one
        return GetOutpointAndKeysFromOutput(vPossibleCoins[0], outpointRet, pubKeyRet, keyRet);

    // Find specific vin
    uint256 txHash = uint256S(strTxHash);
    int nOutputIndex = atoi(strOutputIndex);

    for (const auto& out : vPossibleCoins)
        if (out.tx->GetHash() == txHash && out.i == nOutputIndex) // found it!
            return GetOutpointAndKeysFromOutput(out, outpointRet, pubKeyRet, keyRet);

    LogPrintf("CWallet::GetDynodeOutpointAndKeys -- Could not locate specified dynode vin\n");
    return false;
}

bool CWallet::GetOutpointAndKeysFromOutput(const COutput& out, COutPoint& outpointRet, CPubKey& pubKeyRet, CKey& keyRet)
{
    // wait for reindex and/or import to finish
    if (fImporting || fReindex)
        return false;

    CScript pubScript;

    outpointRet = COutPoint(out.tx->GetHash(), out.i);
    pubScript = out.tx->tx->vout[out.i].scriptPubKey; // the inputs PubKey

    CTxDestination address1;
    ExtractDestination(pubScript, address1);
    CDynamicAddress address2(address1);

    CKeyID keyID;
    if (!address2.GetKeyID(keyID)) {
        LogPrintf("CWallet::GetOutpointAndKeysFromOutput -- Address does not refer to a key\n");
        return false;
    }

    if (!GetKey(keyID, keyRet)) {
        LogPrintf("CWallet::GetOutpointAndKeysFromOutput -- Private key for address is not known\n");
        return false;
    }

    pubKeyRet = keyRet.GetPubKey();
    return true;
}

int CWallet::CountInputsWithAmount(CAmount nInputAmount)
{
    CAmount nTotal = 0;
    {
        LOCK2(cs_main, cs_wallet);
        for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
            const CWalletTx* pcoin = &(*it).second;
            if (pcoin->IsTrusted()) {
                int nDepth = pcoin->GetDepthInMainChain();

                for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++) {
                    COutput out = COutput(pcoin, i, nDepth, true, true, pcoin->IsTrusted());
                    COutPoint outpoint = COutPoint(out.tx->GetHash(), out.i);

                    if (out.tx->tx->vout[out.i].nValue != nInputAmount)
                        continue;
                    if (!CPrivateSend::IsDenominatedAmount(pcoin->tx->vout[i].nValue))
                        continue;
                    if (IsSpent(out.tx->GetHash(), i) || IsMine(pcoin->tx->vout[i]) != ISMINE_SPENDABLE || !IsDenominated(outpoint))
                        continue;

                    nTotal++;
                }
            }
        }
    }

    return nTotal;
}

bool CWallet::HasCollateralInputs(bool fOnlyConfirmed) const
{
    std::vector<COutput> vCoins;
    AvailableCoins(vCoins, fOnlyConfirmed, nullptr, false, ONLY_PRIVATESEND_COLLATERAL);

    return !vCoins.empty();
}

bool CWallet::CreateCollateralTransaction(CMutableTransaction& txCollateral, std::string& strReason)
{
    LOCK2(cs_main, cs_wallet);

    txCollateral.vin.clear();
    txCollateral.vout.clear();

    CReserveKey reservekey(this);
    CAmount nValue = 0;
    CTxPSIn txpsinCollateral;

    if (!GetCollateralTxPSIn(txpsinCollateral, nValue)) {
        strReason = "PrivateSend requires a collateral transaction and could not locate an acceptable input!";
        return false;
    }

    txCollateral.vin.push_back(txpsinCollateral);

    // pay collateral charge in fees
    // NOTE: no need for protobump patch here,
    // CPrivateSend::IsCollateralAmount in GetCollateralTxPSIn should already take care of this
    if (nValue >= CPrivateSend::GetCollateralAmount() * 2) {
        // make our change address
        CScript scriptChange;
        CPubKey vchPubKey;
        bool success = reservekey.GetReservedKey(vchPubKey, true);
        assert(success); // should never fail, as we just unlocked
        scriptChange = GetScriptForDestination(vchPubKey.GetID());
        reservekey.KeepKey();
        // return change
        txCollateral.vout.push_back(CTxOut(nValue - CPrivateSend::GetCollateralAmount(), scriptChange));
    } else { // nValue < CPrivateSend::GetCollateralAmount() * 2
        // create dummy data output only and pay everything as a fee
        txCollateral.vout.push_back(CTxOut(0, CScript() << OP_RETURN));
    }
    unsigned int nIn = 0; //TODO: Assets properly set nIn
    if (!SignSignature(*this, txpsinCollateral.prevPubKey, txCollateral, nIn, nValue, 0)) {
        strReason = "Unable to sign collateral transaction!";
        return false;
    }

    return true;
}

bool CWallet::GetBudgetSystemCollateralTX(CTransactionRef& tx, uint256 hash, CAmount amount, bool fUseInstantSend)
{
    CWalletTx wtx;
    if (GetBudgetSystemCollateralTX(wtx, hash, amount, fUseInstantSend)) {
        tx = wtx.tx;
        return true;
    }
    return false;
}

bool CWallet::GetBudgetSystemCollateralTX(CWalletTx& tx, uint256 hash, CAmount amount, bool fUseInstantSend)
{
    // make our change address
    CReserveKey reservekey(this);

    CScript scriptChange;
    scriptChange << OP_RETURN << ToByteVector(hash);

    CAmount nFeeRet = 0;
    int nChangePosRet = -1;
    std::string strFail = "";
    std::vector<CRecipient> vecSend;
    vecSend.push_back((CRecipient){scriptChange, amount, false});

    CCoinControl coinControl;
    bool success = CreateTransaction(vecSend, tx, reservekey, nFeeRet, nChangePosRet, strFail, coinControl, true, ALL_COINS, fUseInstantSend);
    if (!success) {
        LogPrintf("CWallet::GetBudgetSystemCollateralTX -- Error: %s\n", strFail);
        return false;
    }
    return true;
}


bool CWallet::ConvertList(std::vector<CTxIn> vecTxIn, std::vector<CAmount>& vecAmounts)
{
    for (const auto& txin : vecTxIn) {
        if (mapWallet.count(txin.prevout.hash)) {
            CWalletTx& wtx = mapWallet[txin.prevout.hash];
            if (txin.prevout.n < wtx.tx->vout.size()) {
                vecAmounts.push_back(wtx.tx->vout[txin.prevout.n].nValue);
            }
        } else {
            LogPrintf("CWallet::ConvertList -- Couldn't find transaction\n");
        }
    }
    return true;
}

/** ASSET START */

bool CWallet::CreateTransactionWithAssets(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                               std::string& strFailReason, const CCoinControl& coinControl, const std::vector<CNewAsset> assets, const CTxDestination destination, const AssetType& type, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP)
{
    CReissueAsset reissueAsset;
    return CreateTransactionAll(vecSend, wtxNew, reservekey, nFeeRet, nChangePosInOut, strFailReason, coinControl, true, assets, destination, false, false, reissueAsset, type, sign, nCoinType, false, false);
}

bool CWallet::CreateTransactionWithTransferAsset(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                                         std::string& strFailReason, const CCoinControl& coinControl, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP)
{
    CNewAsset asset;
    CReissueAsset reissueAsset;
    CTxDestination destination;
    AssetType assetType = AssetType::INVALID;
    return CreateTransactionAll(vecSend, wtxNew, reservekey, nFeeRet, nChangePosInOut, strFailReason, coinControl, false, asset, destination, true, false, reissueAsset, assetType, sign, nCoinType, false, false);
}

bool CWallet::CreateTransactionWithReissueAsset(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                                         std::string& strFailReason, const CCoinControl& coinControl, const CReissueAsset& reissueAsset, const CTxDestination destination, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP)
{
    CNewAsset asset;
    AssetType assetType = AssetType::REISSUE;
    return CreateTransactionAll(vecSend, wtxNew, reservekey, nFeeRet, nChangePosInOut, strFailReason, coinControl, false, asset, destination, false, true, reissueAsset, assetType, sign, nCoinType, false, false);
}

bool CWallet::CreateTransaction(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                                        std::string& strFailReason, const CCoinControl& coinControl, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP)
{

    CNewAsset asset;
    CReissueAsset reissueAsset;
    CTxDestination destination;
    AssetType assetType = AssetType::INVALID;
    return CreateTransactionAll(vecSend, wtxNew, reservekey, nFeeRet, nChangePosInOut, strFailReason, coinControl, false,  asset, destination, false, false, reissueAsset, assetType, sign, nCoinType, false, false);
}

bool CWallet::CreateTransactionAll(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey,
                                   CAmount& nFeeRet, int& nChangePosInOut, std::string& strFailReason,
                                   const CCoinControl& coinControl, bool fNewAsset, const CNewAsset& asset,
                                   const CTxDestination destination, bool fTransferAsset, bool fReissueAsset,
                                   const CReissueAsset& reissueAsset, const AssetType& assetType, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP)
{
    std::vector<CNewAsset> assets;
    assets.push_back(asset);
    return CreateTransactionAll(vecSend, wtxNew, reservekey, nFeeRet, nChangePosInOut, strFailReason, coinControl,
                                fNewAsset, assets, destination, fTransferAsset, fReissueAsset, reissueAsset, assetType,
                                sign, nCoinType, false, false);
}

bool CWallet::CreateTransactionAll(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey,
                                   CAmount& nFeeRet, int& nChangePosInOut, std::string& strFailReason,
                                   const CCoinControl& coinControl, bool fNewAsset,
                                   const std::vector<CNewAsset> assets, const CTxDestination destination,
                                   bool fTransferAsset, bool fReissueAsset, const CReissueAsset& reissueAsset,
                                   const AssetType& assetType, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP)
{
    CAmount nFeePay = fUseInstantSend ? CTxLockRequest().GetMinFee(true) : 0;
    std::string strOpType;
    CScript prevScriptPubKey;

/** ASSET START */
    if (!AreAssetsDeployed() && (fTransferAsset || fNewAsset || fReissueAsset))
        return false;

    if (fNewAsset && (assets.size() < 1 || !IsValidDestination(destination)))
        return error("%s : Tried creating a new asset transaction and the asset was null or the destination was invalid", __func__);

    if ((fNewAsset && fTransferAsset) || (fReissueAsset && fTransferAsset) || (fReissueAsset && fNewAsset))
        return error("%s : Only one type of asset transaction allowed per transaction");

    if (fReissueAsset && (reissueAsset.IsNull() || !IsValidDestination(destination)))
        return error("%s : Tried reissuing an asset and the reissue data was null or the destination was invalid", __func__);

    std::map<std::string, CAmount> mapAssetValue;
/* ASSET END */
    CAmount nValue = 0;
    int nChangePosRequest = nChangePosInOut;
    unsigned int nSubtractFeeFromAmount = 0;
    for (const auto& recipient : vecSend) {
/** ASSET START */
        if (fTransferAsset || fReissueAsset || assetType == AssetType::SUB || assetType == AssetType::UNIQUE || assetType == AssetType::MSGCHANNEL || assetType == AssetType::SUB_QUALIFIER || assetType == AssetType::RESTRICTED) {
            CAssetTransfer assetTransfer;
            std::string address;
            if (TransferAssetFromScript(recipient.scriptPubKey, assetTransfer, address)) {
                if (!mapAssetValue.count(assetTransfer.strName))
                    mapAssetValue[assetTransfer.strName] = 0;

                if (assetTransfer.nAmount <= 0) {
                    strFailReason = _("Asset Transfer amounts must be greater than 0");
                    return false;
                }

                mapAssetValue[assetTransfer.strName] += assetTransfer.nAmount;
            }
        }
/** ASSET END */
        if (nValue < 0 || recipient.nAmount < 0) {
            strFailReason = _("Transaction amounts must not be negative");
            return false;
        }
        nValue += recipient.nAmount;

        if (recipient.fSubtractFeeFromAmount)
            nSubtractFeeFromAmount++;
    }
    if (vecSend.empty()) {
        strFailReason = _("Transaction must have at least one recipient");
        return false;
    }

    wtxNew.fTimeReceivedIsTxTime = true;
    wtxNew.BindWallet(this);
    CMutableTransaction txNew;

    // Use special version number when creating a BDAP transaction
    if (fIsBDAP)
        txNew.nVersion = BDAP_TX_VERSION;

    // Discourage fee sniping.
    //
    // For a large miner the value of the transactions in the best block and
    // the mempool can exceed the cost of deliberately attempting to mine two
    // blocks to orphan the current best block. By setting nLockTime such that
    // only the next block can include the transaction, we discourage this
    // practice as the height restricted and limited blocksize gives miners
    // considering fee sniping fewer options for pulling off this attack.
    //
    // A simple way to think about this is from the wallet's point of view we
    // always want the blockchain to move forward. By setting nLockTime this
    // way we're basically making the statement that we only want this
    // transaction to appear in the next block; we don't want to potentially
    // encourage reorgs by allowing transactions to appear at lower heights
    // than the next block in forks of the best chain.
    //
    // Of course, the subsidy is high enough, and transaction volume low
    // enough, that fee sniping isn't a problem yet, but by implementing a fix
    // now we ensure code won't be written that makes assumptions about
    // nLockTime that preclude a fix later.

    txNew.nLockTime = chainActive.Height();

    // Secondly occasionally randomly pick a nLockTime even further back, so
    // that transactions that are delayed after signing for whatever reason,
    // e.g. high-latency mix networks and some CoinJoin implementations, have
    // better privacy.
    if (GetRandInt(10) == 0)
        txNew.nLockTime = std::max(0, (int)txNew.nLockTime - GetRandInt(100));

    assert(txNew.nLockTime <= (unsigned int)chainActive.Height());
    assert(txNew.nLockTime < LOCKTIME_THRESHOLD);
    FeeCalculation feeCalc;
    CAmount nFeeNeeded; // needed for BDAP
    {
        std::set<CInputCoin> setCoins;
/** ASSET START */
        std::set<CInputCoin> setAssets;
/** ASSET END */
        std::vector<CTxPSIn> vecTxPSInTmp;
        LOCK2(cs_main, cs_wallet);
        {
            std::vector<COutput> vAvailableCoins;
/** ASSET START */
            std::map<std::string, std::vector<COutput> > mapAssetCoins;
            if (fTransferAsset || fReissueAsset || assetType == AssetType::SUB || assetType == AssetType::UNIQUE || assetType == AssetType::MSGCHANNEL || assetType == AssetType::SUB_QUALIFIER || assetType == AssetType::RESTRICTED)
                AvailableCoinsWithAssets(vAvailableCoins, mapAssetCoins, true, &coinControl);
            else
                AvailableCoins(vAvailableCoins, true, &coinControl);
/** ASSET END */

            int nInstantSendConfirmationsRequired = Params().GetConsensus().nInstantSendConfirmationsRequired;
            if (!fIsBDAP) {
                AvailableCoins(vAvailableCoins, true, &coinControl, false, nCoinType, fUseInstantSend, false);
            } else {
                std::vector<unsigned char> vchValue;
                CScript bdapOperationScript;
                if (!GetScriptOpTypeValue(vecSend, bdapOperationScript, strOpType, vchValue)) {
                    strFailReason = _("Failed to find BDAP operation script in the recipient array.");
                    return false;
                }
                if (strOpType == "bdap_new_account" || strOpType == "bdap_new_audit" ) {
                    // Use BDAP credits first.
                    AvailableCoins(vAvailableCoins, true, &coinControl, false, nCoinType, fUseInstantSend, true);
                }
                else if (strOpType == "bdap_update_account" || strOpType == "bdap_delete_account") {
                    CDomainEntry prevEntry;
                    if (CheckDomainEntryDB()) {
                        if (!pDomainEntryDB->GetDomainEntryInfo(vchValue, prevEntry)) {
                            strFailReason = _("GetDomainEntryInfo failed to find previous domanin entry.");
                            return false;
                        }
                        if (prevEntry.ObjectID.size() == 0) {
                            strFailReason = _("GetDomainEntryInfo returned a blank domanin entry.");
                            return false;
                        }
                    }
                    else {
                        strFailReason = _("CheckDomainEntryDB failed.");
                        return false;
                    }
                    CTransactionRef prevTx;
                    uint256 hashBlock;
                    if (!GetTransaction(prevEntry.txHash, prevTx, Params().GetConsensus(), hashBlock, true)) {
                        strFailReason = _("GetDomainEntryInfo failed to find previous domanin entry transaction.");
                        return false;
                    }
                    
                    GetBDAPOpScript(prevTx, prevScriptPubKey);
                    GetBDAPCoins(vAvailableCoins, prevScriptPubKey);
                }
                else if (strOpType == "bdap_new_link_request") {
                    uint256 txid;
                    if (GetLinkIndex(vchValue, txid)) {
                        strFailReason = _("Public key already used for a link request.");
                        return false;
                    }
                    AvailableCoins(vAvailableCoins, true, &coinControl, false, nCoinType, fUseInstantSend, true);
                }
                else if (strOpType == "bdap_new_link_accept") {
                    uint256 txid;
                    if (GetLinkIndex(vchValue, txid)) {
                        strFailReason = _("Public key already used for an accepted link.");
                        return false;
                    }
                    AvailableCoins(vAvailableCoins, true, &coinControl, false, nCoinType, fUseInstantSend, true);
                }
                else if (strOpType == "bdap_delete_link_request") {
                    uint256 prevTxId;
                    if (!GetLinkIndex(vchValue, prevTxId)) {
                        strFailReason = _("Link accept pubkey could not be found.");
                        return false;
                    }
                    CTransactionRef prevTx;
                    if (!GetPreviousTxRefById(prevTxId, prevTx)) {
                        strFailReason = _("Previous delete link request transaction could not be found.");
                        return false;
                    }
                    CScript prevScriptPubKey;
                    GetBDAPOpScript(prevTx, prevScriptPubKey);
                    GetBDAPCoins(vAvailableCoins, prevScriptPubKey);
                }
                else if (strOpType == "bdap_delete_link_accept") {
                    uint256 prevTxId;
                    if (!GetLinkIndex(vchValue, prevTxId)) {
                        strFailReason = _("Link accept pubkey could not be found.");
                        return false;
                    }
                    CTransactionRef prevTx;
                    if (!GetPreviousTxRefById(prevTxId, prevTx)) {
                        strFailReason = _("Previous delete link accept transaction could not be found.");
                        return false;
                    }
                    CScript prevScriptPubKey;
                    GetBDAPOpScript(prevTx, prevScriptPubKey);
                    GetBDAPCoins(vAvailableCoins, prevScriptPubKey);
                }
                else if (strOpType == "bdap_move_asset") {
                    AvailableCoins(vAvailableCoins, true, &coinControl, false, nCoinType, fUseInstantSend, false);
                    fIsBDAP = false; // Treat like a standard transaction so standard fees are applied.
                }
                else if (strOpType == "bdap_update_link_accept") {
                    strFailReason = strOpType + _(" not implemented yet.");
                    return false;
                }
                else if (strOpType == "bdap_update_link_request") {
                    strFailReason = strOpType + _(" not implemented yet.");
                    return false;
                }
                else {
                    strFailReason = strOpType + _(" is an uknown BDAP operation.");
                    return false;
                }
            }

            // Create change script that will be used if we need change
            // TODO: pass in scriptChange instead of reservekey so
            // change transaction isn't always pay-to-dynamic-address
            CScript scriptChange;
            CScript assetScriptChange;

            // coin control: send change to custom address
            if (!boost::get<CNoDestination>(&coinControl.destChange)) {
                scriptChange = GetScriptForDestination(coinControl.destChange);
            } else {

                // no coin control: send change to newly generated address
                CKeyID keyID;
                if (!CreateNewChangeAddress(reservekey, keyID, strFailReason))
                    return false;

                scriptChange = GetScriptForDestination(keyID);
            }

            /** ASSET START */
            if (!boost::get<CNoDestination>(&coinControl.assetDestChange)) {
                assetScriptChange = GetScriptForDestination(coinControl.assetDestChange);
            } else {
                assetScriptChange = scriptChange;
            }
            /** ASSET END */

            CTxOut change_prototype_txout(0, scriptChange);
            size_t change_prototype_size = GetSerializeSize(change_prototype_txout, SER_DISK, 0);

            CFeeRate discard_rate = GetDiscardRate(::feeEstimator);
            nFeeRet = 0;
            bool pick_new_inputs = true;

            // Start with no fee and loop until there is enough fee
            while (true) {
/** ASSET START */
                std::map<std::string, CAmount> mapAssetsIn;
/** ASSET END */
                nChangePosInOut = nChangePosRequest;
                txNew.vin.clear();
                txNew.vout.clear();
                wtxNew.fFromMe = true;
                bool fFirst = true;

                CAmount nValueToSelect = nValue;
                if (nSubtractFeeFromAmount == 0)
                    nValueToSelect += nFeeRet;
                double dPriority = 0;
                // vouts to the payees
                for (const auto& recipient : vecSend) {
                    CTxOut txout(recipient.nAmount, recipient.scriptPubKey);
                    
/** ASSET START */
                    // Check to see if you need to make an asset data outpoint OP_DYN_ASSET data
                    if (recipient.scriptPubKey.IsNullAssetTxDataScript()) {
                        assert(txout.nValue == 0);
                        txNew.vout.push_back(txout);
                        continue;
                    }
/** ASSET END */

                    if (IsTransactionFluid(recipient.scriptPubKey)) {
                        // Check if fluid transaction is already in the mempool
                        if (fluid.CheckIfExistsInMemPool(mempool, recipient.scriptPubKey, strFailReason)) {
                            // fluid transaction is already in the mempool.  Invalid transaction.
                            return false;
                        }
                        // Check the validity of the fluid transaction's public script.
                        if (!fluid.CheckFluidOperationScript(recipient.scriptPubKey, GetTime(), strFailReason)) {
                            return false;
                        }
                    }

                    if (recipient.fSubtractFeeFromAmount) {
                        txout.nValue -= nFeeRet / nSubtractFeeFromAmount; // Subtract fee equally from each selected recipient

                        if (fFirst) // first receiver pays the remainder not divisible by output count
                        {
                            fFirst = false;
                            txout.nValue -= nFeeRet % nSubtractFeeFromAmount;
                        }
                    }

                    if (txout.IsDust(dustRelayFee)) {
                        if (recipient.fSubtractFeeFromAmount && nFeeRet > 0) {
                            if (txout.nValue < 0)
                                strFailReason = _("The transaction amount is too small to pay the fee");
                            else
                                strFailReason = _("The transaction amount is too small to send after the fee has been deducted");
                        } else
                            strFailReason = _("Transaction amount too small");
                        return false;
                    }
                    txNew.vout.push_back(txout);
                }

                // Choose coins to use
                CAmount nValueIn = 0;
                setCoins.clear();
                if (!SelectCoins(vAvailableCoins, nValueToSelect, setCoins, nValueIn, &coinControl, nCoinType, fUseInstantSend)) {
                    if (nCoinType == ONLY_NONDENOMINATED) {
                        strFailReason = _("Unable to locate enough PrivateSend non-denominated funds for this transaction.");
                    } else if (nCoinType == ONLY_DENOMINATED) {
                        strFailReason = _("Unable to locate enough PrivateSend denominated funds for this transaction.");
                        strFailReason += " " + _("PrivateSend uses exact denominated amounts to send funds, you might simply need to anonymize some more coins.");
                    } else if (nValueIn < nValueToSelect) {
                        strFailReason = _("Insufficient funds.");
                        if (fUseInstantSend) {
                            // could be not true but most likely that's the reason
                            strFailReason += " " + strprintf(_("InstantSend requires inputs with at least %d confirmations, you might need to wait a few minutes and try again."), nInstantSendConfirmationsRequired);
                        }
                    }
/** ASSET START */
                    if (AreAssetsDeployed()) {
                        setAssets.clear();
                        mapAssetsIn.clear();
                        if (!SelectAssets(mapAssetCoins, mapAssetValue, setAssets, mapAssetsIn)) {
                            strFailReason = _("Insufficient asset funds");
                            return false;
                        }
                    }
/** ASSET END */
                    return false;
                }
                if (fUseInstantSend && nValueIn > sporkManager.GetSporkValue(SPORK_5_INSTANTSEND_MAX_VALUE) * COIN) {
                    strFailReason += " " + strprintf(_("InstantSend doesn't support sending values that high yet. Transactions are currently limited to %1 DYN."), sporkManager.GetSporkValue(SPORK_5_INSTANTSEND_MAX_VALUE));
                    return false;
                }

                bool fUsingBDAPCredits = false;
                CScript scriptBdapChange;
                for (const auto& pcoin : setCoins) {
                    CAmount nCredit = pcoin.txout.nValue;
                    //The coin age after the next block (depth+1) is used instead of the current,
                    //reflecting an assumption the user would accept a bit more delay for
                    //a chance at a free transaction.
                    //But mempool inputs might still be in the mempool, so their age stays 0
                    int age = pcoin.walletTx->GetDepthInMainChain();
                    assert(age >= 0);
                    if (age != 0)
                        age += 1;
                    dPriority += (double)nCredit * age;
                    int opCode1 = -1; int opCode2 = -1;
                    pcoin.txout.GetBDAPOpCodes(opCode1, opCode2);
                    std::string strOpType = GetBDAPOpTypeString(opCode1, opCode2);
                    if (strOpType == "bdap_move_asset") {
                        GetBDAPCreditScript(pcoin.walletTx->tx, scriptBdapChange);
                        CDynamicAddress address = GetScriptAddress(scriptBdapChange);
                        LogPrintf("%s -- address %s, scriptBdapChange %s\n", __func__, address.ToString(), ScriptToAsmStr(scriptBdapChange));
                        fUsingBDAPCredits = true;
                    }
                }

                const CAmount nChange = nValueIn - nValueToSelect;
                CTxOut newTxOut;

/** ASSET START */
                if (AreAssetsDeployed()) {
                    // Add the change for the assets
                    std::map<std::string, CAmount> mapAssetChange;
                    for (auto asset : mapAssetValue) {
                        if (mapAssetsIn.count(asset.first))
                            mapAssetChange.insert(
                                    std::make_pair(asset.first, (mapAssetsIn.at(asset.first) - asset.second)));
                    }

                    for (auto assetChange : mapAssetChange) {
                        if (assetChange.second > 0) {
                            if (IsAssetNameAnRestricted(assetChange.first))
                            {
                                // Get the verifier string for the restricted asset
                                CNullAssetTxVerifierString verifier;
                                if (!passets->GetAssetVerifierStringIfExists(assetChange.first, verifier)) {
                                    strFailReason = _("Verifier String for asset trasnfer, not found");
                                    return false;
                                }

                                // Get the change address
                                CTxDestination dest;
                                if (!ExtractDestination(assetScriptChange, dest)) {
                                    strFailReason = _("Failed to extract destination from change script");
                                    return false;
                                }

                                std::string change_address = EncodeDestination(dest);
                                bool fFoundValueChangeAddress = false;
                                // Check the verifier string against the change address, if it fails, we will try to send the change back to the same input that created this transaction
                                if (!ContextualCheckVerifierString(passets, verifier.verifier_string, change_address, strFailReason)) {
                                    // Loop through all assets that are inputs into the transaction
                                    for (auto asset: setAssets) {
                                        if (asset.txout.scriptPubKey.IsAssetScript()) {
                                            CAssetOutputEntry outputData;
                                            if (!GetAssetData(asset.txout.scriptPubKey, outputData)) {
                                                strFailReason = _("Failed to get asset data from script");
                                                return false;
                                            }

                                            // If the asset names don't match, continue through the set of assets
                                            if (outputData.assetName != assetChange.first)
                                                continue;

                                            std::string check_address = EncodeDestination(outputData.destination);

                                            if (ContextualCheckVerifierString(passets, verifier.verifier_string, check_address, strFailReason)) {
                                                fFoundValueChangeAddress = true;

                                                CScript scriptAssetChange = GetScriptForDestination(outputData.destination);
                                                CAssetTransfer assetTransfer(assetChange.first, assetChange.second);

                                                assetTransfer.ConstructTransaction(scriptAssetChange);
                                                CTxOut newAssetTxOut(0, scriptAssetChange);

                                                txNew.vout.emplace_back(newAssetTxOut);
                                                break;
                                            }
                                        }
                                    }
                                } else  {
                                    fFoundValueChangeAddress = true;
                                    CScript scriptAssetChange = assetScriptChange;
                                    CAssetTransfer assetTransfer(assetChange.first, assetChange.second);

                                    assetTransfer.ConstructTransaction(scriptAssetChange);
                                    CTxOut newAssetTxOut(0, scriptAssetChange);

                                    txNew.vout.emplace_back(newAssetTxOut);
                                }
                                if (!fFoundValueChangeAddress) {
                                    strFailReason = _("Failed to find restricted asset change address from inputs");
                                    return false;
                                }
                            } else {
                                CScript scriptAssetChange = assetScriptChange;
                                CAssetTransfer assetTransfer(assetChange.first, assetChange.second);

                                assetTransfer.ConstructTransaction(scriptAssetChange);
                                CTxOut newAssetTxOut(0, scriptAssetChange);

                                txNew.vout.emplace_back(newAssetTxOut);
                            }
                        }
                    }
                }
/** ASSET END */

                if (nChange > 0) {
                    //over pay for denominated transactions
                    if (nCoinType == ONLY_DENOMINATED) {
                        nFeeRet += nChange;
                        wtxNew.mapValue["PS"] = "1";
                        // recheck skipped denominations during next mixing
                        privateSendClient.ClearSkippedDenominations();
                    } else {
                        // Fill a vout to ourself
                        // TODO: pass in scriptChange instead of reservekey so
                        // change transaction isn't always pay-to-dynamic-address
                        CScript scriptChange;
/** ASSET START */
                        CScript assetScriptChange;
/** ASSET END */
                        // coin control: send change to custom address
                        if (boost::get<CNoDestination>(&coinControl.destChange)) {
                            scriptChange = GetScriptForDestination(coinControl.destChange);
                        } else if (strOpType == "bdap_update_account" || strOpType == "bdap_delete_account") {
                            // send deposit change back to original account.
                            scriptChange = prevScriptPubKey;
                        } else if (fUsingBDAPCredits) {
                            scriptChange = scriptBdapChange;
                        } else {
                            // no coin control: send change to newly generated address
                            // Note: We use a new key here to keep it from being obvious which side is the change.
                            //  The drawback is that by not reusing a previous key, the change may be lost if a
                            //  backup is restored, if the backup doesn't have the new private key for the change.
                            //  If we reused the old key, it would be possible to add code to look for and
                            //  rediscover unknown transactions that were written with keys of ours to recover
                            //  post-backup change.

                            // Reserve a new key pair from key pool
                            CPubKey vchPubKey;
                            if (!reservekey.GetReservedKey(vchPubKey, true)) {
                                strFailReason = _("Keypool ran out, please call keypoolrefill first");
                                return false;
                            }
                            scriptChange = GetScriptForDestination(vchPubKey.GetID());
                        }

/** ASSET START */
                        if (!boost::get<CNoDestination>(&coinControl.assetDestChange)) {
                            assetScriptChange = GetScriptForDestination(coinControl.assetDestChange);
                        } else {
                            assetScriptChange = scriptChange;
                        }
/** ASSET END */

                        newTxOut = CTxOut(nChange, scriptChange);

                        // We do not move dust-change to fees, because the sender would end up paying more than requested.
                        // This would be against the purpose of the all-inclusive feature.
                        // So instead we raise the change and deduct from the recipient.
                        if (nSubtractFeeFromAmount > 0 && newTxOut.IsDust(dustRelayFee)) {
                            CAmount nDust = newTxOut.GetDustThreshold(dustRelayFee) - newTxOut.nValue;
                            newTxOut.nValue += nDust;                         // raise change until no more dust
                            for (unsigned int i = 0; i < vecSend.size(); i++) // subtract from first recipient
                            {
                                if (vecSend[i].fSubtractFeeFromAmount) {
                                    txNew.vout[i].nValue -= nDust;
                                    if (txNew.vout[i].IsDust(dustRelayFee)) {
                                        strFailReason = _("The transaction amount is too small to send after the fee has been deducted");
                                        return false;
                                    }
                                    break;
                                }
                            }
                        }

                        // Never create dust outputs; if we would, just
                        // add the dust to the fee.
                        if (newTxOut.IsDust(dustRelayFee)) {
                            nChangePosInOut = -1;
                            nFeeRet += nChange;
                            reservekey.ReturnKey();
                        } else {
                            if (nChangePosInOut == -1) {
                                // Insert change txn at random position:
                                nChangePosInOut = GetRandInt(txNew.vout.size() + 1);
                            } else if ((unsigned int)nChangePosInOut > txNew.vout.size()) {
                                strFailReason = _("Change index out of range");
                                return false;
                            }

                            std::vector<CTxOut>::iterator position = txNew.vout.begin() + nChangePosInOut;
                            txNew.vout.insert(position, newTxOut);
                        }
                    }
                } else {
                    reservekey.ReturnKey();
                    nChangePosInOut = -1;
                }

/** ASSET START */
                if (AreAssetsDeployed()) {
                    if (fNewAsset) {
                        for (auto asset : assets) {
                            // Create the owner token output for non-unique assets
                            if (assetType != AssetType::UNIQUE && assetType != AssetType::MSGCHANNEL && assetType != AssetType::QUALIFIER && assetType != AssetType::SUB_QUALIFIER && assetType != AssetType::RESTRICTED) {
                                CScript ownerScript = GetScriptForDestination(destination);
                                asset.ConstructOwnerTransaction(ownerScript);
                                CTxOut ownerTxOut(0, ownerScript);
                                txNew.vout.push_back(ownerTxOut);
                            }

                            // Create the asset transaction and push it back so it is the last CTxOut in the transaction
                            CScript scriptPubKey = GetScriptForDestination(destination);
                            asset.ConstructTransaction(scriptPubKey);
                            CTxOut newTxOut(0, scriptPubKey);
                            txNew.vout.push_back(newTxOut);
                        }
                    } else if (fReissueAsset) {
                        // Create the asset transaction and push it back so it is the last CTxOut in the transaction
                        CScript reissueScript = GetScriptForDestination(destination);

                        // Create the scriptPubKeys for the reissue data, and that owner asset
                        reissueAsset.ConstructTransaction(reissueScript);

                        CTxOut reissueTxOut(0, reissueScript);
                        txNew.vout.push_back(reissueTxOut);
                    }
                }
/** ASSET END */

                // Fill vin
                //
                // Note how the sequence number is set to max()-1 so that the
                // nLockTime set above actually works.
                vecTxPSInTmp.clear();
                for (const auto& coin : setCoins) {
                    CTxIn txin = CTxIn(coin.outpoint, CScript(),
                        std::numeric_limits<unsigned int>::max() - 1);
                    vecTxPSInTmp.push_back(CTxPSIn(txin, coin.txout.scriptPubKey));
                    txNew.vin.push_back(txin);
                }

                sort(txNew.vin.begin(), txNew.vin.end(), CompareInputBIP69());
                sort(vecTxPSInTmp.begin(), vecTxPSInTmp.end(), CompareInputBIP69());
                sort(txNew.vout.begin(), txNew.vout.end(), CompareOutputBIP69());

                // If there was change output added before, we must update its position now
                if (nChangePosInOut != -1) {
                    int i = 0;
                    for (const CTxOut& txOut : txNew.vout) {
                        if (txOut == newTxOut) {
                            nChangePosInOut = i;
                            break;
                        }
                        i++;
                    }
                }

                //
                // BIP125 defines opt-in RBF as any nSequence < maxint-1, so
                // we use the highest possible value in that range (maxint-2)
                // to avoid conflicting with other possible uses of nSequence,
                // and in the spirit of "smallest possible change from prior
                // behavior."
                const uint32_t nSequence = CTxIn::SEQUENCE_FINAL - 1;
                for (const auto& coin : setCoins)
                    txNew.vin.push_back(CTxIn(coin.outpoint,CScript(),
                                              nSequence));

/** ASSET START */
                if (AreAssetsDeployed()) {
                    for (const auto &asset : setAssets)
                        txNew.vin.push_back(CTxIn(asset.outpoint, CScript(),
                                                  nSequence));
                }

                // Add the new asset inputs into the tempSet so the dummysigntx will add the correct amount of sigsß
                std::set<CInputCoin> tempSet = setCoins;
                tempSet.insert(setAssets.begin(), setAssets.end());
/** ASSET END */

                // Fill in dummy signatures for fee calculation.
                int nIn = 0;
                for (const auto& txpsin : vecTxPSInTmp) {
                    const CScript scriptPubKey = txpsin.prevPubKey;
                    const CScript scriptSigRes = txNew.vin[nIn].scriptSig;
                    SignatureData sigdata(scriptSigRes);
                    if (!ProduceSignature(DummySignatureCreator(this), scriptPubKey, sigdata)) {
                        strFailReason = _("Signing transaction failed");
                        return false;
                    }
                    nIn++;
                }

                unsigned int nBytes = ::GetSerializeSize(txNew, SER_NETWORK, PROTOCOL_VERSION);
                // TODO: remove for assets?
                if (nBytes > MAX_STANDARD_TX_SIZE) {
                    // Do not create oversized transactions (bad-txns-oversize).
                    strFailReason = _("Transaction too large");
                    return false;
                }

                CTransaction txNewConst(txNew);
                dPriority = txNewConst.ComputePriority(dPriority, nBytes);

                // Remove scriptSigs to eliminate the fee calculation dummy signatures
                for (auto& txin : txNew.vin) {
                    txin.scriptSig = CScript();
                }

                nFeeNeeded = GetMinimumFee(nBytes, coinControl, ::mempool, ::feeEstimator, &feeCalc);

                if (fIsBDAP) {
                    if (strOpType == "bdap_new_account" || strOpType == "bdap_delete_account" || strOpType == "bdap_update_account") {
                        // Check the memory pool for a pending tranaction for the same domain entry
                        CTransactionRef pTxNew = MakeTransactionRef(txNew);
                        CDomainEntry domainEntry(pTxNew);
                        if (domainEntry.CheckIfExistsInMemPool(mempool, strFailReason)) {
                            return false;
                        }
                    }
                }

                // Allow to override the default confirmation target over the CoinControl instance
                int currentConfirmationTarget = nTxConfirmTarget;
                if (coinControl.nConfirmTarget > 0)
                    currentConfirmationTarget = coinControl.nConfirmTarget;

                // Can we complete this as a free transaction?
                // Note: InstantSend transaction can't be a free one
                if (!fUseInstantSend && fSendFreeTransactions && nBytes <= MAX_FREE_TRANSACTION_CREATE_SIZE) {
                    // Not enough fee: enough priority?
                    double dPriorityNeeded = mempool.estimateSmartPriority(currentConfirmationTarget);
                    // Require at least hard-coded AllowFree.
                    if (dPriority >= dPriorityNeeded && AllowFree(dPriority))
                        break;

                    // Small enough, and priority high enough, to send for free
                    //                    if (dPriorityNeeded > 0 && dPriority >= dPriorityNeeded)
                    //                        break;
                }
                // Standard fee not needed for BDAP
                CAmount nFeeNeeded = !fIsBDAP ? std::max(nFeePay, GetMinimumFee(nBytes, coinControl, mempool, ::feeEstimator, nullptr)) : 0;

                if (fUseInstantSend) {
                    nFeeNeeded = std::max(nFeeNeeded, CTxLockRequest(txNew).GetMinFee(true));
                }

                if (coinControl.fOverrideFeeRate)
                    nFeeNeeded = coinControl.m_feerate->GetFee(nBytes);

                // If we made it here and we aren't even able to meet the relay fee on the next pass, give up
                // because we must be at the maximum allowed fee.
                if (!fIsBDAP && nFeeNeeded < ::minRelayTxFee.GetFee(nBytes)) {
                    strFailReason = _("Transaction too large for fee policy");
                    return false;
                }

                if (nFeeRet >= nFeeNeeded) {
                    // Reduce fee to only the needed amount if possible. This
                    // prevents potential overpayment in fees if the coins
                    // selected to meet nFeeNeeded result in a transaction that
                    // requires less fee than the prior iteration.

                    // If we have no change and a big enough excess fee, then
                    // try to construct transaction again only without picking
                    // new inputs. We now know we only need the smaller fee
                    // (because of reduced tx size) and so we should add a
                    // change output. Only try this once.
                    if (nChangePosInOut == -1 && nSubtractFeeFromAmount == 0 && pick_new_inputs) {
                        unsigned int tx_size_with_change = nBytes + change_prototype_size + 2; // Add 2 as a buffer in case increasing # of outputs changes compact size
                        CAmount fee_needed_with_change = GetMinimumFee(tx_size_with_change, coinControl, ::mempool, ::feeEstimator, nullptr);
                        CAmount minimum_value_for_change = GetDustThreshold(change_prototype_txout, discard_rate);
                        if (nFeeRet >= fee_needed_with_change + minimum_value_for_change) {
                            pick_new_inputs = false;
                            nFeeRet = fee_needed_with_change;
                            continue;
                        }
                    }

                    // If we have change output already, just increase it
                    if (nFeeRet > nFeeNeeded && nChangePosInOut != -1 && nSubtractFeeFromAmount == 0) {
                        CAmount extraFeePaid = nFeeRet - nFeeNeeded;
                        std::vector<CTxOut>::iterator change_position = txNew.vout.begin()+nChangePosInOut;
                        change_position->nValue += extraFeePaid;
                        nFeeRet -= extraFeePaid;
                    }
                    break; // Done, enough fee included.
                }
                else if (!pick_new_inputs) {
                    // This shouldn't happen, we should have had enough excess
                    // fee to pay for the new output and still meet nFeeNeeded
                    // Or we should have just subtracted fee from recipients and
                    // nFeeNeeded should not have changed
                    strFailReason = _("Transaction fee and change calculation failed");
                    return false;
                }

                // Try to reduce change to include necessary fee
                if (nChangePosInOut != -1 && nSubtractFeeFromAmount == 0) {
                    CAmount additionalFeeNeeded = nFeeNeeded - nFeeRet;
                    std::vector<CTxOut>::iterator change_position = txNew.vout.begin()+nChangePosInOut;
                    // Only reduce change if remaining amount is still a large enough output.
                    if (change_position->nValue >= MIN_FINAL_CHANGE + additionalFeeNeeded) {
                        change_position->nValue -= additionalFeeNeeded;
                        nFeeRet += additionalFeeNeeded;
                        break; // Done, able to increase fee from change
                    }
                }

                // If subtracting fee from recipients, we now know what fee we
                // need to subtract, we have no reason to reselect inputs
                if (nSubtractFeeFromAmount > 0) {
                    pick_new_inputs = false;
                }

                // Include more fee and try again.
                nFeeRet = nFeeNeeded;
                continue;
            }
        }

        if (sign)
        {
            CTransaction txNewConst(txNew);
            int nIn = 0;
            for (const auto& txpsin : vecTxPSInTmp) 
            {
                const CScript scriptPubKey = txpsin.prevPubKey;
                const CScript scriptSigRes = txNew.vin[nIn].scriptSig;
                SignatureData sigdata(scriptSigRes);

                if (!ProduceSignature(TransactionSignatureCreator(this, &txNewConst, nIn, SIGHASH_ALL), scriptPubKey, sigdata)) 
                {
                    strFailReason = _("Signing transaction failed");
                    return false;
                } else {
/** ASSET START */
                    UpdateTransaction(txNew, nIn, sigdata);
                }
                nIn++;
            }
            if (AreAssetsDeployed()) {
                for (const auto &asset : setAssets) {
                    const CScript &scriptPubKey = asset.txout.scriptPubKey;
                    SignatureData sigdata;

                    if (!ProduceSignature(
                            TransactionSignatureCreator(this, &txNewConst, nIn, asset.txout.nValue, SIGHASH_ALL),
                            scriptPubKey, sigdata)) {
                        strFailReason = _("Signing asset transaction failed");
                        return false;
                    } else {
                        UpdateTransaction(txNew, nIn, sigdata);
                    }

                    nIn++;
                }
            }
/** ASSET END */
        }

        // Embed the constructed transaction data in wtxNew.
        wtxNew.SetTx(MakeTransactionRef(std::move(txNew)));

        // Limit size
        if (GetTransactionWeight(wtxNew) >= MAX_STANDARD_TX_WEIGHT)
        {
            strFailReason = _("Transaction too large");
            return false;
        }
    }

    if (gArgs.GetBoolArg("-walletrejectlongchains", DEFAULT_WALLET_REJECT_LONG_CHAINS)) {
        // Lastly, ensure this tx will pass the mempool's chain limits
        LockPoints lp;
        CTxMemPoolEntry entry(wtxNew.tx, 0, 0, 0, 0, 0, false, 0, lp);
        CTxMemPool::setEntries setAncestors;
        size_t nLimitAncestors = gArgs.GetArg("-limitancestorcount", DEFAULT_ANCESTOR_LIMIT);
        size_t nLimitAncestorSize = gArgs.GetArg("-limitancestorsize", DEFAULT_ANCESTOR_SIZE_LIMIT) * 1000;
        size_t nLimitDescendants = gArgs.GetArg("-limitdescendantcount", DEFAULT_DESCENDANT_LIMIT);
        size_t nLimitDescendantSize = gArgs.GetArg("-limitdescendantsize", DEFAULT_DESCENDANT_SIZE_LIMIT) * 1000;
        std::string errString;
        if (!mempool.CalculateMemPoolAncestors(entry, setAncestors, nLimitAncestors, nLimitAncestorSize, nLimitDescendants, nLimitDescendantSize, errString)) {
            strFailReason = _("Transaction has too long of a mempool chain");
            return false;
        }
    }
    return true;
}
/** ASSET END */


/**
 * Call after CreateTransaction unless you want to abort
 */
bool CWallet::CommitTransaction(CWalletTx& wtxNew, CReserveKey& reservekey, CConnman* connman, CValidationState& state, const std::string& strCommand)
{
    {
        LOCK2(cs_main, cs_wallet);
        LogPrintf("CommitTransaction:\n%s", wtxNew.tx->ToString());
        {
            // Take key pair from key pool so it won't be used again
            reservekey.KeepKey();

            // Add tx to wallet, because if it has change it's also ours,
            // otherwise just for transaction history.
            AddToWallet(wtxNew);

            // Notify that old coins are spent
            std::set<uint256> updated_hahes;
            for (const CTxIn& txin : wtxNew.tx->vin) {
                // notify only once
                if (updated_hahes.find(txin.prevout.hash) != updated_hahes.end())
                    continue;

                CWalletTx& coin = mapWallet[txin.prevout.hash];
                coin.BindWallet(this);
                NotifyTransactionChanged(this, txin.prevout.hash, CT_UPDATED);
                updated_hahes.insert(txin.prevout.hash);
            }
        }

        // Track how many getdata requests our transaction gets
        mapRequestCount[wtxNew.GetHash()] = 0;

        if (fBroadcastTransactions) {
            // Broadcast
            if (!wtxNew.AcceptToMemoryPool(maxTxFee, state)) {
                LogPrintf("CommitTransaction(): Transaction cannot be broadcast immediately, %s\n", state.GetRejectReason());
                // TODO: if we expect the failure to be long term or permanent, instead delete wtx from the wallet and return failure.
            } else {
                wtxNew.RelayWalletTransaction(connman, strCommand);
            }
        }
    }
    return true;
}

void CWallet::ListAccountCreditDebit(const std::string& strAccount, std::list<CAccountingEntry>& entries)
{
    CWalletDB walletdb(strWalletFile);
    return walletdb.ListAccountCreditDebit(strAccount, entries);
}

bool CWallet::AddAccountingEntry(const CAccountingEntry& acentry)
{
    CWalletDB walletdb(strWalletFile);

    return AddAccountingEntry(acentry, &walletdb);
}

bool CWallet::AddAccountingEntry(const CAccountingEntry& acentry, CWalletDB* pwalletdb)
{
    if (!pwalletdb->WriteAccountingEntry_Backend(acentry))
        return false;

    laccentries.push_back(acentry);
    CAccountingEntry& entry = laccentries.back();
    wtxOrdered.insert(make_pair(entry.nOrderPos, TxPair((CWalletTx*)0, &entry)));

    return true;
}

DBErrors CWallet::LoadWallet(bool& fFirstRunRet, const bool fImportMnemonic)
{
    if (!fFileBacked)
        return DB_LOAD_OK;
    fFirstRunRet = false;
    DBErrors nLoadWalletRet = CWalletDB(strWalletFile, "cr+").LoadWallet(this);
    if (nLoadWalletRet == DB_NEED_REWRITE) {
        if (CDB::Rewrite(strWalletFile, "\x04pool")) {
            LOCK(cs_wallet);
            setInternalKeyPool.clear();
            setExternalKeyPool.clear();
            nKeysLeftSinceAutoBackup = 0;
            // Note: can't top-up keypool here, because wallet is locked.
            // User will be prompted to unlock wallet the next operation
            // that requires a new key.
        }
    }

    {
        LOCK2(cs_main, cs_wallet);
        for (auto& pair : mapWallet) {
            for (unsigned int i = 0; i < pair.second.tx->vout.size(); ++i) {
                if (IsMine(pair.second.tx->vout[i]) && !IsSpent(pair.first, i)) {
                    setWalletUTXO.insert(COutPoint(pair.first, i));
                }
            }
        }
    }

    if (nLoadWalletRet != DB_LOAD_OK)
        return nLoadWalletRet;
    fFirstRunRet = !vchDefaultKey.IsValid();

    //Do not execute if Importing a Mnemonic
    if (!fImportMnemonic) {
        uiInterface.LoadWallet(this);
    }

    return DB_LOAD_OK;
}

// Goes through all wallet transactions and checks if they are dynode collaterals, in which case these are locked
// This avoids accidential spending of collaterals. They can still be unlocked manually if a spend is really intended.
void CWallet::AutoLockDynodeCollaterals()
{
    LOCK2(cs_main, cs_wallet);
    for (const auto& pair : mapWallet) {
        for (unsigned int i = 0; i < pair.second.tx->vout.size(); ++i) {
            if (IsMine(pair.second.tx->vout[i]) && !IsSpent(pair.first, i)) {
                    LockCoin(COutPoint(pair.first, i));
            }
        }
    }
}

DBErrors CWallet::ZapSelectTx(std::vector<uint256>& vHashIn, std::vector<uint256>& vHashOut)
{
    if (!fFileBacked)
        return DB_LOAD_OK;
    DBErrors nZapSelectTxRet = CWalletDB(strWalletFile, "cr+").ZapSelectTx(this, vHashIn, vHashOut);
    if (nZapSelectTxRet == DB_NEED_REWRITE) {
        if (CDB::Rewrite(strWalletFile, "\x04pool")) {
            LOCK(cs_wallet);
            setInternalKeyPool.clear();
            setExternalKeyPool.clear();
            // Note: can't top-up keypool here, because wallet is locked.
            // User will be prompted to unlock wallet the next operation
            // that requires a new key.
        }
    }

    if (nZapSelectTxRet != DB_LOAD_OK)
        return nZapSelectTxRet;

    MarkDirty();

    return DB_LOAD_OK;
}

DBErrors CWallet::ZapWalletTx(std::vector<CWalletTx>& vWtx)
{
    if (!fFileBacked)
        return DB_LOAD_OK;
    DBErrors nZapWalletTxRet = CWalletDB(strWalletFile, "cr+").ZapWalletTx(this, vWtx);
    if (nZapWalletTxRet == DB_NEED_REWRITE) {
        if (CDB::Rewrite(strWalletFile, "\x04pool")) {
            LOCK(cs_wallet);
            setInternalKeyPool.clear();
            setExternalKeyPool.clear();
            nKeysLeftSinceAutoBackup = 0;
            // Note: can't top-up keypool here, because wallet is locked.
            // User will be prompted to unlock wallet the next operation
            // that requires a new key.
        }
    }

    if (nZapWalletTxRet != DB_LOAD_OK)
        return nZapWalletTxRet;

    return DB_LOAD_OK;
}


bool CWallet::SetAddressBook(const CTxDestination& address, const std::string& strName, const std::string& strPurpose)
{
    bool fUpdated = false;
    {
        LOCK(cs_wallet); // mapAddressBook
        std::map<CTxDestination, CAddressBookData>::iterator mi = mapAddressBook.find(address);
        fUpdated = mi != mapAddressBook.end();
        mapAddressBook[address].name = strName;
        if (!strPurpose.empty()) /* update purpose only if requested */
            mapAddressBook[address].purpose = strPurpose;
    }
    NotifyAddressBookChanged(this, address, strName, ::IsMine(*this, address) != ISMINE_NO,
        strPurpose, (fUpdated ? CT_UPDATED : CT_NEW));
    if (!fFileBacked)
        return false;
    if (!strPurpose.empty() && !CWalletDB(strWalletFile).WritePurpose(CDynamicAddress(address).ToString(), strPurpose))
        return false;
    return CWalletDB(strWalletFile).WriteName(CDynamicAddress(address).ToString(), strName);
}

bool CWallet::DelAddressBook(const CTxDestination& address)
{
    {
        LOCK(cs_wallet); // mapAddressBook

        if (fFileBacked) {
            // Delete destdata tuples associated with address
            std::string strAddress = CDynamicAddress(address).ToString();
            for (const std::pair<std::string, std::string>& item : mapAddressBook[address].destdata) {
                CWalletDB(strWalletFile).EraseDestData(strAddress, item.first);
            }
        }
        mapAddressBook.erase(address);
    }

    NotifyAddressBookChanged(this, address, "", ::IsMine(*this, address) != ISMINE_NO, "", CT_DELETED);

    if (!fFileBacked)
        return false;
    CWalletDB(strWalletFile).ErasePurpose(CDynamicAddress(address).ToString());
    return CWalletDB(strWalletFile).EraseName(CDynamicAddress(address).ToString());
}

bool CWallet::SetDefaultKey(const CPubKey& vchPubKey)
{
    if (fFileBacked) {
        if (!CWalletDB(strWalletFile).WriteDefaultKey(vchPubKey))
            return false;
    }
    vchDefaultKey = vchPubKey;
    return true;
}

/**
 * Mark old keypool keys as used,
 * and generate all new keys 
 */
bool CWallet::NewKeyPool()
{
    {
        LOCK(cs_wallet);
        CWalletDB walletdb(strWalletFile);
        for (int64_t nIndex : setInternalKeyPool) {
            walletdb.ErasePool(nIndex);
        }
        setInternalKeyPool.clear();
        for (int64_t nIndex : setExternalKeyPool) {
            walletdb.ErasePool(nIndex);
        }
        setExternalKeyPool.clear();
        privateSendClient.fEnablePrivateSend = false;
        nKeysLeftSinceAutoBackup = 0;

        if (!TopUpKeyPoolCombo()) //was TopUpKeyPool
            return false;

        LogPrintf("CWallet::NewKeyPool rewrote keypool\n");
    }
    return true;
}

bool CWallet::NewEdKeyPool()
{
    {
        LOCK(cs_wallet);
        CWalletDB walletdb(strWalletFile);
        for (int64_t nIndex : setInternalEdKeyPool) {
            walletdb.EraseEdPool(nIndex);
        }
        setInternalEdKeyPool.clear();
        for (int64_t nIndex : setExternalEdKeyPool) {
            walletdb.EraseEdPool(nIndex);
        }
        setExternalEdKeyPool.clear();

        if (!TopUpKeyPoolCombo()) //was topupedkeypool
            return false;

        LogPrintf("CWallet::NewEdKeyPool rewrote edkeypool\n");
    }
    return true;
}

size_t CWallet::KeypoolCountExternalKeys()
{
    AssertLockHeld(cs_wallet); // setExternalKeyPool
    return setExternalKeyPool.size();
}

size_t CWallet::KeypoolCountInternalKeys()
{
    AssertLockHeld(cs_wallet); // setInternalKeyPool
    return setInternalKeyPool.size();
}

size_t CWallet::EdKeypoolCountExternalKeys()
{
    AssertLockHeld(cs_wallet); // setExternalEdKeyPool
    return setExternalEdKeyPool.size();
}

size_t CWallet::EdKeypoolCountInternalKeys()
{
    AssertLockHeld(cs_wallet); // setInternalEdKeyPool
    return setInternalEdKeyPool.size();
}

bool CWallet::SyncEdKeyPool() 
{
    if (pwalletMain->IsLocked())
        return false;
    
    CWalletDB walletdb(strWalletFile);

    CKeyPool keypool;
    CPubKey retrievedPubKey;
    CKey retrievedKey;

    CKeyPool keypool2;
    CPubKey retrievedPubKey2;
    CKey retrievedKey2;

    for (int64_t nIndex : setInternalKeyPool) {
        if (!walletdb.ReadPool(nIndex, keypool)) {
            throw std::runtime_error(std::string(__func__) + ": read failed");
        }
        retrievedPubKey = keypool.vchPubKey;
        GetKey(retrievedPubKey.GetID(), retrievedKey);

        std::array<char, 32> edSeed = ConvertSecureVector32ToArray(retrievedKey.getKeyData());
        CKeyEd25519 childKey(edSeed);
        CharString vchDHTPubKey = childKey.GetPubKey();

        CKeyID vchDHTPubKeyID = GetIdFromCharVector(vchDHTPubKey);

        //only add if doesn't currently exist
        if (!pwalletMain->HaveDHTKey(vchDHTPubKeyID)) {
            if (!walletdb.WriteEdPool(nIndex, CEdKeyPool(GenerateNewEdKey(0, true, retrievedKey), true)))
                throw std::runtime_error("SyncEdKeyPool(): writing generated key failed");
            else {
                setInternalEdKeyPool.insert(nIndex);
            }
        }
    }

    for (int64_t nIndex : setExternalKeyPool) {
        if (!walletdb.ReadPool(nIndex, keypool2)) {
            throw std::runtime_error(std::string(__func__) + ": read failed");
        }
        retrievedPubKey2 = keypool2.vchPubKey;
        GetKey(retrievedPubKey2.GetID(), retrievedKey2);

        std::array<char, 32> edSeed2 = ConvertSecureVector32ToArray(retrievedKey2.getKeyData());
        CKeyEd25519 childKey2(edSeed2);
        CharString vchDHTPubKey2 = childKey2.GetPubKey();

        CKeyID vchDHTPubKeyID2 = GetIdFromCharVector(vchDHTPubKey2);

        //only add if doesn't currently exist
        if (!pwalletMain->HaveDHTKey(vchDHTPubKeyID2)) {
            if (!walletdb.WriteEdPool(nIndex, CEdKeyPool(GenerateNewEdKey(0, false, retrievedKey2), false)))
                throw std::runtime_error("SyncEdKeyPool(): writing generated key failed");
            else {
                setExternalEdKeyPool.insert(nIndex);
            }
        }
    }

    return true;

} //SyncEdKeyPool

bool CWallet::TopUpKeyPoolCombo(unsigned int kpSize, bool fIncreaseSize)
{
    {
        LOCK(cs_wallet);

        if (IsLocked(true))
            return false;

        int64_t amountExternal = setExternalKeyPool.size();
        int64_t amountInternal = setInternalKeyPool.size();

        // Top up key pool
        unsigned int nTargetSize;
        unsigned int defaultKeyPoolSize = std::max(gArgs.GetArg("-keypool", DEFAULT_KEYPOOL_SIZE), (int64_t)0);
        if (kpSize > 0)
            nTargetSize = kpSize;
        else {
            if (defaultKeyPoolSize >= DynamicKeyPoolSize) {
                DynamicKeyPoolSize = defaultKeyPoolSize;
            }

            if (fIncreaseSize) {
                DynamicKeyPoolSize = DynamicKeyPoolSize + 1;
            } //if fIncreaseSize

            nTargetSize = DynamicKeyPoolSize; 
        }

        // count amount of available keys (internal, external)
        // make sure the keypool of external and internal keys fits the user selected target (-keypool)

        int64_t missingExternal = std::max(std::max((int64_t)nTargetSize, (int64_t)1) - amountExternal, (int64_t)0);
        int64_t missingInternal = std::max(std::max((int64_t)nTargetSize, (int64_t)1) - amountInternal, (int64_t)0);

        if (!IsHDEnabled()) {
            // don't create extra internal keys
            missingInternal = 0;
        } else {
            nTargetSize *= 2;
        }
        bool fInternal = false;
        CWalletDB walletdb(strWalletFile);
        for (int64_t i = missingInternal + missingExternal; i--;) {
            int64_t nEnd = 1;
            //int64_t nEdEnd = 1;
            if (i < missingInternal) {
                fInternal = true;
            }
            if (!setInternalKeyPool.empty()) {
                nEnd = *(--setInternalKeyPool.end()) + 1;
            }
            if (!setExternalKeyPool.empty()) {
                nEnd = std::max(nEnd, *(--setExternalKeyPool.end()) + 1);
            }
            // TODO: implement keypools for all accounts?

            //get seed for ed29915 keys
            CPubKey retrievedPubKey;
            CKey retrievedKey;
            
            retrievedPubKey = GenerateNewKey(0, fInternal);
            GetKey(retrievedPubKey.GetID(), retrievedKey);

            if (!walletdb.WritePool(nEnd, CKeyPool(retrievedPubKey, fInternal)))
                throw std::runtime_error("TopUpKeyPoolCombo(): writing generated key failed");

            if (fInternal) {
                setInternalKeyPool.insert(nEnd);
            } else {
                setExternalKeyPool.insert(nEnd);
            }

            // TODO: implement keypools for all accounts?
            if (!walletdb.WriteEdPool(nEnd, CEdKeyPool(GenerateNewEdKey(0, fInternal, retrievedKey), fInternal)))
                throw std::runtime_error("TopUpKeyPoolCombo(): writing generated key failed");

            if (fInternal) {
                setInternalEdKeyPool.insert(nEnd);
            } else {
                setExternalEdKeyPool.insert(nEnd);
            }

            double dProgress = 100.f * nEnd / (nTargetSize + 1);
            std::string strMsg = "";
            if (dProgress <= 100)
                strMsg = strprintf(_("Loading wallet... (%3.2f %%)"), dProgress);
            else
                strMsg = strprintf(_("Increasing keypool... (%d)"),amountExternal);
            uiInterface.InitMessage(strMsg);
        }
    }
    return true;
} //TopUpKeyPoolCombo

void CWallet::UpdateKeyPoolsFromTransactions(const std::string& strOpType, const std::vector<std::vector<unsigned char>>& vvchOpParameters)
{
    if (vvchOpParameters.size() <= 1)
        return;
    
    std::vector<unsigned char> key0 = vvchOpParameters[0];
    std::vector<unsigned char> key1 = vvchOpParameters[1];

    if (strOpType == "bdap_new_account") {
        ReserveEdKeyForTransactions(key1);
    }
    else if (strOpType == "bdap_new_link_request" || strOpType == "bdap_new_link_accept") {
        ReserveEdKeyForTransactions(key0);
        fNeedToUpdateLinks = true;
    }

} //UpdateKeyPoolsFromTransactions


void CWallet::ReserveKeysFromKeyPools(int64_t& nIndex, CKeyPool& keypool, CEdKeyPool& edkeypool, bool fInternal)
{
    nIndex = -1;
    int64_t nEdIndex = nIndex;
    keypool.vchPubKey = CPubKey();
    {
        LOCK(cs_wallet);

        if (!IsLocked(true))
            TopUpKeyPoolCombo(); //was TopUpKeyPool();

        fInternal = fInternal && IsHDEnabled();
        std::set<int64_t>& setKeyPool = fInternal ? setInternalKeyPool : setExternalKeyPool;
        std::set<int64_t>& setEdKeyPool = fInternal ? setInternalEdKeyPool : setExternalEdKeyPool;

        // Get the oldest key
        if (setKeyPool.empty())
            return;

        CWalletDB walletdb(strWalletFile);

        nIndex = *setKeyPool.begin();
        setKeyPool.erase(nIndex);
        if (!walletdb.ReadPool(nIndex, keypool)) {
            throw std::runtime_error(std::string(__func__) + ": read failed [keypool]");
        }
        if (!HaveKey(keypool.vchPubKey.GetID())) {
            throw std::runtime_error(std::string(__func__) + ": unknown key in key pool");
        }
        if (keypool.fInternal != fInternal) {
            throw std::runtime_error(std::string(__func__) + ": keypool entry misclassified");
        }

        assert(keypool.vchPubKey.IsValid());
        LogPrintf("keypool reserve %d\n", nIndex);

        nEdIndex = *setEdKeyPool.begin();
        setEdKeyPool.erase(nEdIndex);
        if (!walletdb.ReadEdPool(nEdIndex, edkeypool)) {
            throw std::runtime_error(std::string(__func__) + ": read failed [edkeypool]");
        }
        if (edkeypool.fInternal != fInternal) {
            throw std::runtime_error(std::string(__func__) + ": keypool entry misclassified");
        }

        LogPrintf("edkeypool reserve %d\n", nEdIndex);

    }
} //ReserveKeysFromKeyPools

void CWallet::ReserveEdKeyForTransactions(const std::vector<unsigned char>& pubKeyToReserve)
{
        CWalletDB walletdb(strWalletFile);
        CEdKeyPool edkeypool;
        std::vector<unsigned char> edPubKey;
        std::vector<int64_t> keypoolIndexes;
        bool EraseIndex = false;
        int64_t IndexToErase = 0;
        int64_t nIndex = 0;
        std::set<std::int64_t>::iterator it = setInternalEdKeyPool.begin();

        while ((it != setInternalEdKeyPool.end()) && (!EraseIndex)) {
            nIndex = *it;
            if (!walletdb.ReadEdPool(nIndex, edkeypool)) {
                throw std::runtime_error(std::string(__func__) + ": read failed");
            }
            edPubKey = edkeypool.edPubKey;

            if(pubKeyToReserve == edPubKey) {
                KeepKey(nIndex);
                fNeedToUpdateKeyPools = true;
                EraseIndex = true;
                IndexToErase = nIndex;
                ReserveKeyCount++;
            }
            it++;
        }

        if (EraseIndex) {
            std::set<int64_t>::iterator eraseIndexEd = setInternalEdKeyPool.find(IndexToErase);
            std::set<int64_t>::iterator eraseIndex = setInternalKeyPool.find(IndexToErase);
            if (eraseIndexEd != setInternalEdKeyPool.end())
                setInternalEdKeyPool.erase(eraseIndexEd);
            if (eraseIndex != setInternalKeyPool.end())
                setInternalKeyPool.erase(eraseIndex);
        }

} //ReserveEdKeyForTransactions

bool CWallet::ReserveKeyForTransactions(const CPubKey& pubKeyToReserve)
{
        bool foundPubKey = false;

        CWalletDB walletdb(strWalletFile);
        CKeyPool keypool;
        CPubKey PubKey;
        std::vector<int64_t> keypoolIndexes;
        bool EraseIndex = false;
        int64_t IndexToErase = 0;
        int64_t nIndex = 0;
        std::set<std::int64_t>::iterator it = setExternalKeyPool.begin();

        while ((it != setExternalKeyPool.end()) && (!EraseIndex)) {
            nIndex = *it;
            if (!walletdb.ReadPool(nIndex, keypool)) {
                throw std::runtime_error(std::string(__func__) + ": read failed");
            }
            PubKey = keypool.vchPubKey;

            if(pubKeyToReserve == PubKey) {
                foundPubKey = true;
                KeepKey(nIndex);
                EraseIndex = true;
                IndexToErase = nIndex;
                ReserveKeyCount++;
                if (ReserveKeyCount <= DEFAULT_RESCAN_THRESHOLD) {
                    SaveRescanIndex = true;
                }
            }
            it++;
        }

        if (EraseIndex) {
            // Wallets before v2.4 do not have an external Ed25519 key pool.
            if (setExternalEdKeyPool.size() > 0) {
                std::set<int64_t>::iterator eraseIndexEd = setExternalEdKeyPool.find(IndexToErase);
                if (eraseIndexEd != setExternalKeyPool.end())
                    setExternalEdKeyPool.erase(eraseIndexEd);
            }
            if (setExternalKeyPool.size() > 0) {
                std::set<int64_t>::iterator eraseIndex = setExternalKeyPool.find(IndexToErase);
                if (eraseIndex != setExternalKeyPool.end())
                    setExternalKeyPool.erase(eraseIndex);
            }
        }

        return foundPubKey;

} //ReserveKeyForTransactions

void CWallet::KeepKey(int64_t nIndex)
{
    // Remove from key pool
    if (fFileBacked) {
        CWalletDB walletdb(strWalletFile);
        walletdb.ErasePool(nIndex);
        walletdb.EraseEdPool(nIndex);
        nKeysLeftSinceAutoBackup = nWalletBackups ? nKeysLeftSinceAutoBackup - 1 : 0;
    }
}

void CWallet::ReturnKey(int64_t nIndex, bool fInternal)
{
    // Return to key pool
    {
        LOCK(cs_wallet);
        if (fInternal) {
            setInternalKeyPool.insert(nIndex);
            setInternalEdKeyPool.insert(nIndex);
        } else {
            setExternalKeyPool.insert(nIndex);
            setExternalEdKeyPool.insert(nIndex);
        }
    }
    LogPrintf("keypool return %d\n", nIndex);
    LogPrintf("edkeypool return %d\n", nIndex);
}

bool CWallet::GetKeysFromPool(CPubKey& pubkeyWallet, std::vector<unsigned char>& vchEd25519PubKey, CStealthAddress& sxAddr, bool fInternal)
{
    if (!GetKeysFromPool(pubkeyWallet, vchEd25519PubKey, fInternal))
        return false;

    CKey walletKey, spendKey, scanKey;
    if (!GetKey(pubkeyWallet.GetID(), walletKey))
        return false;

    if (!walletKey.DeriveChildKey(spendKey))
        return false;

    if (!spendKey.DeriveChildKey(scanKey))
        return false;

    CStealthAddress sx(scanKey, spendKey);
    sxAddr = sx;
    return true;
}

bool CWallet::GetKeysFromPool(CPubKey& pubkeyWallet, std::vector<unsigned char>& vchEd25519PubKey, bool fInternal)
{
    int64_t nIndex = 0;
    int64_t nEdIndex = 0;
    CKeyPool keypool;
    CEdKeyPool edkeypool;
    bool keptAlready = false;
    {
        LOCK(cs_wallet);
        ReserveKeysFromKeyPools(nIndex, keypool, edkeypool, fInternal);
        nEdIndex = nIndex;
        if (nIndex == -1) {
            if (IsLocked(true))
                return false;
            // TODO: implement keypool for all accouts?
            pubkeyWallet = GenerateNewKey(0, fInternal);
        }
        else {
            keptAlready = true;
            KeepKey(nIndex);
            pubkeyWallet = keypool.vchPubKey;
        }

        if (nEdIndex == -1) {
            if (IsLocked(true))
                return false;
            CKey keyRetrieved;
            GetKey(pubkeyWallet.GetID(), keyRetrieved);
            vchEd25519PubKey = GenerateNewEdKey(0, fInternal, keyRetrieved);
        }
        else {
            if (!keptAlready) 
                KeepKey(nIndex);
            vchEd25519PubKey = edkeypool.edPubKey;
        }
    }
    return true;
}

static int64_t GetOldestKeyInPool(const std::set<int64_t>& setKeyPool, CWalletDB& walletdb)
{
    CKeyPool keypool;
    int64_t nIndex = *(setKeyPool.begin());
    if (!walletdb.ReadPool(nIndex, keypool)) {
        throw std::runtime_error(std::string(__func__) + ": read oldest key in keypool failed");
    }
    assert(keypool.vchPubKey.IsValid());
    return keypool.nTime;
}

int64_t CWallet::GetOldestKeyPoolTime()
{
    LOCK(cs_wallet);

    // if the keypool is empty, return <NOW>
    if (setExternalKeyPool.empty() && setInternalKeyPool.empty())
        return GetTime();

    CWalletDB walletdb(strWalletFile);
    int64_t oldestKey = -1;

    // load oldest key from keypool, get time and return
    if (!setInternalKeyPool.empty()) {
        oldestKey = std::max(GetOldestKeyInPool(setInternalKeyPool, walletdb), oldestKey);
    }
    if (!setExternalKeyPool.empty()) {
        oldestKey = std::max(GetOldestKeyInPool(setExternalKeyPool, walletdb), oldestKey);
    }
    return oldestKey;
}

std::map<CTxDestination, CAmount> CWallet::GetAddressBalances()
{
    std::map<CTxDestination, CAmount> balances;

    {
        LOCK(cs_wallet);
        for (const auto& walletEntry : mapWallet)
        {
            const CWalletTx *pcoin = &walletEntry.second;

            if (!pcoin->IsTrusted())
                continue;

            if (pcoin->IsCoinBase() && pcoin->GetBlocksToMaturity() > 0)
                continue;

            int nDepth = pcoin->GetDepthInMainChain();
            if ((nDepth < (pcoin->IsFromMe(ISMINE_ALL) ? 0 : 1)) && !pcoin->IsLockedByInstantSend())
                continue;

            for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++) 
            {
                CTxDestination addr;
                if (!IsMine(pcoin->tx->vout[i]))
                    continue;
                if (!ExtractDestination(pcoin->tx->vout[i].scriptPubKey, addr))
                    continue;

                CAmount n = IsSpent(walletEntry.first, i) ? 0 : pcoin->tx->vout[i].nValue;

                if (!balances.count(addr))
                    balances[addr] = 0;
                balances[addr] += n;
            }
        }
    }

    return balances;
}

std::set<std::set<CTxDestination> > CWallet::GetAddressGroupings()
{
    AssertLockHeld(cs_wallet); // mapWallet
    std::set<std::set<CTxDestination> > groupings;
    std::set<CTxDestination> grouping;

    for (const auto& walletEntry : mapWallet)
    {
        const CWalletTx *pcoin = &walletEntry.second;

        if (pcoin->tx->vin.size() > 0)
        {
            bool any_mine = false;
            // group all input addresses with each other
            for (CTxIn txin : pcoin->tx->vin) {
                CTxDestination address;
                if (!IsMine(txin)) /* If this input isn't mine, ignore it */
                    continue;
                if (!ExtractDestination(mapWallet[txin.prevout.hash].tx->vout[txin.prevout.n].scriptPubKey, address))
                    continue;
                grouping.insert(address);
                any_mine = true;
            }

            // group change with input addresses
            if (any_mine) {
                for (CTxOut txout : pcoin->tx->vout)
                    if (IsChange(txout)) {
                        CTxDestination txoutAddr;
                        if (!ExtractDestination(txout.scriptPubKey, txoutAddr))
                            continue;
                        grouping.insert(txoutAddr);
                    }
            }
            if (grouping.size() > 0) {
                groupings.insert(grouping);
                grouping.clear();
            }
        }

        // group lone addrs by themselves
        for (unsigned int i = 0; i < pcoin->tx->vout.size(); i++)
            if (IsMine(pcoin->tx->vout[i])) {
                CTxDestination address;
                if (!ExtractDestination(pcoin->tx->vout[i].scriptPubKey, address))
                    continue;
                grouping.insert(address);
                groupings.insert(grouping);
                grouping.clear();
            }
    }

    std::set<std::set<CTxDestination>*> uniqueGroupings;        // a set of pointers to groups of addresses
    std::map<CTxDestination, std::set<CTxDestination>*> setmap; // map addresses to the unique group containing it
    for (std::set<CTxDestination> _grouping : groupings)
    {
        // make a set of all the groups hit by this new group
        std::set<std::set<CTxDestination>*> hits;
        std::map<CTxDestination, std::set<CTxDestination>*>::iterator it;
        for (CTxDestination address : _grouping)
            if ((it = setmap.find(address)) != setmap.end())
                hits.insert((*it).second);

        // merge all hit groups into a new single group and delete old groups
        std::set<CTxDestination>* merged = new std::set<CTxDestination>(_grouping);
        for (std::set<CTxDestination>* hit : hits)
        {
            merged->insert(hit->begin(), hit->end());
            uniqueGroupings.erase(hit);
            delete hit;
        }
        uniqueGroupings.insert(merged);

        // update setmap
        for (CTxDestination element : *merged)
            setmap[element] = merged;
    }

    std::set<std::set<CTxDestination> > ret;
    for (std::set<CTxDestination>* uniqueGrouping : uniqueGroupings)
    {
        ret.insert(*uniqueGrouping);
        delete uniqueGrouping;
    }

    return ret;
}

CAmount CWallet::GetAccountBalance(const std::string& strAccount, int nMinDepth, const isminefilter& filter, bool fAddLocked)
{
    CWalletDB walletdb(strWalletFile);
    return GetAccountBalance(walletdb, strAccount, nMinDepth, filter, fAddLocked);
}

CAmount CWallet::GetAccountBalance(CWalletDB& walletdb, const std::string& strAccount, int nMinDepth, const isminefilter& filter, bool fAddLocked)
{
    CAmount nBalance = 0;

    // Tally wallet transactions
    for (std::map<uint256, CWalletTx>::iterator it = mapWallet.begin(); it != mapWallet.end(); ++it) {
        const CWalletTx& wtx = (*it).second;
        if (!CheckFinalTx(wtx) || wtx.GetBlocksToMaturity() > 0 || wtx.GetDepthInMainChain() < 0)
            continue;

        CAmount nReceived, nSent, nFee;
        wtx.GetAccountAmounts(strAccount, nReceived, nSent, nFee, filter);

        if (nReceived != 0 && ((wtx.GetDepthInMainChain() >= nMinDepth) || (fAddLocked && wtx.IsLockedByInstantSend())))
            nBalance += nReceived;
        nBalance -= nSent + nFee;
    }

    // Tally internal accounting entries
    nBalance += walletdb.GetAccountCreditDebit(strAccount);

    return nBalance;
}

std::set<CTxDestination> CWallet::GetAccountAddresses(const std::string& strAccount) const
{
    LOCK(cs_wallet);
    std::set<CTxDestination> result;
    for (const std::pair<CTxDestination, CAddressBookData>& item : mapAddressBook) {
        const CTxDestination& address = item.first;
        const std::string& strName = item.second.name;
        if (strName == strAccount)
            result.insert(address);
    }
    return result;
}

bool CReserveKey::GetReservedKey(CPubKey& pubkey, bool fInternalIn)
{
    if (nIndex == -1) {
        CKeyPool keypool;
        CEdKeyPool edkeypool;
        pwallet->ReserveKeysFromKeyPools(nIndex, keypool, edkeypool, fInternalIn);
        if (nIndex != -1) {
            vchPubKey = keypool.vchPubKey;
        } else {
            return false;
        }
        fInternal = keypool.fInternal;
    }
    assert(vchPubKey.IsValid());
    pubkey = vchPubKey;
    return true;
}

void CReserveKey::KeepKey()
{
    if (nIndex != -1) {
        pwallet->KeepKey(nIndex);
    }
    nIndex = -1;
    vchPubKey = CPubKey();
}

void CReserveKey::ReturnKey()
{
    if (nIndex != -1) {
        pwallet->ReturnKey(nIndex, fInternal);
    }
    nIndex = -1;
    vchPubKey = CPubKey();
}

static void LoadReserveKeysToSet(std::set<CKeyID>& setAddress, const std::set<int64_t>& setKeyPool, CWalletDB& walletdb)
{
    for (const int64_t& id : setKeyPool) {
        CKeyPool keypool;
        if (!walletdb.ReadPool(id, keypool))
            throw std::runtime_error(std::string(__func__) + ": read failed");
        assert(keypool.vchPubKey.IsValid());
        CKeyID keyID = keypool.vchPubKey.GetID();
        setAddress.insert(keyID);
    }
}

void CWallet::GetAllReserveKeys(std::set<CKeyID>& setAddress) const
{
    setAddress.clear();

    CWalletDB walletdb(strWalletFile);

    LOCK2(cs_main, cs_wallet);
    LoadReserveKeysToSet(setAddress, setInternalKeyPool, walletdb);
    LoadReserveKeysToSet(setAddress, setExternalKeyPool, walletdb);

    for (const CKeyID& keyID : setAddress) {
        if (!HaveKey(keyID)) {
            throw std::runtime_error(std::string(__func__) + ": unknown key in key pool");
        }
    }
}

bool CWallet::UpdatedTransaction(const uint256& hashTx)
{
    {
        LOCK(cs_wallet);
        // Only notify UI if this transaction is in this wallet
        std::map<uint256, CWalletTx>::const_iterator mi = mapWallet.find(hashTx);
        if (mi != mapWallet.end()) {
            NotifyTransactionChanged(this, hashTx, CT_UPDATED);
            return true;
        }
    }
    return false;
}

void CWallet::GetScriptForMining(std::shared_ptr<CReserveScript>& script)
{
    std::shared_ptr<CReserveKey> rKey(new CReserveKey(this));
    CPubKey pubkey;
    if (!rKey->GetReservedKey(pubkey, false))
        return;

    script = rKey;
    script->reserveScript = CScript() << ToByteVector(pubkey) << OP_CHECKSIG;
}

void CWallet::LockCoin(const COutPoint& output)
{
    AssertLockHeld(cs_wallet); // setLockedCoins
    setLockedCoins.insert(output);
    std::map<uint256, CWalletTx>::iterator it = mapWallet.find(output.hash);
    if (it != mapWallet.end())
        it->second.MarkDirty(); // recalculate all credits for this tx

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;
}

void CWallet::UnlockCoin(const COutPoint& output)
{
    AssertLockHeld(cs_wallet); // setLockedCoins
    setLockedCoins.erase(output);
    std::map<uint256, CWalletTx>::iterator it = mapWallet.find(output.hash);
    if (it != mapWallet.end())
        it->second.MarkDirty(); // recalculate all credits for this tx

    fAnonymizableTallyCached = false;
    fAnonymizableTallyCachedNonDenom = false;
}

void CWallet::UnlockAllCoins()
{
    AssertLockHeld(cs_wallet); // setLockedCoins
    setLockedCoins.clear();
}

bool CWallet::IsLockedCoin(uint256 hash, unsigned int n) const
{
    AssertLockHeld(cs_wallet); // setLockedCoins
    COutPoint outpt(hash, n);

    return (setLockedCoins.count(outpt) > 0);
}

void CWallet::ListLockedCoins(std::vector<COutPoint>& vOutpts) const
{
    AssertLockHeld(cs_wallet); // setLockedCoins
    for (std::set<COutPoint>::iterator it = setLockedCoins.begin();
        it != setLockedCoins.end(); it++) {
        COutPoint outpt = (*it);
        vOutpts.push_back(outpt);
    }
}

/** @} */ // end of Actions

class CAffectedKeysVisitor : public boost::static_visitor<void>
{
private:
    const CKeyStore& keystore;
    std::vector<CKeyID>& vKeys;

public:
    CAffectedKeysVisitor(const CKeyStore& keystoreIn, std::vector<CKeyID>& vKeysIn) : keystore(keystoreIn), vKeys(vKeysIn) {}

    void Process(const CScript& script)
    {
        txnouttype type;
        std::vector<CTxDestination> vDest;
        int nRequired;
        if (ExtractDestinations(script, type, vDest, nRequired)) {
            for (const CTxDestination& dest : vDest)
                boost::apply_visitor(*this, dest);
        }
    }

    void operator()(const CKeyID& keyId)
    {
        if (keystore.HaveKey(keyId))
            vKeys.push_back(keyId);
    }

    void operator()(const CScriptID& scriptId)
    {
        CScript script;
        if (keystore.GetCScript(scriptId, script))
            Process(script);
    }

    void operator()(const CNoDestination& none) {}

    void operator()(const CStealthAddress& sxAddr) {}

};

void CWallet::GetKeyBirthTimes(std::map<CTxDestination, int64_t>& mapKeyBirth) const
{
    AssertLockHeld(cs_wallet); // mapKeyMetadata
    mapKeyBirth.clear();

    // get birth times for keys with metadata
    for (const auto& entry : mapKeyMetadata) {
        if (entry.second.nCreateTime) {
            mapKeyBirth[entry.first] = entry.second.nCreateTime;
        }
    }

    // map in which we'll infer heights of other keys
    CBlockIndex* pindexMax = chainActive[std::max(0, chainActive.Height() - 144)]; // the tip can be reorganised; use a 144-block safety margin
    std::map<CKeyID, CBlockIndex*> mapKeyFirstBlock;
    std::set<CKeyID> setKeys;
    GetKeys(setKeys);
    for (const CKeyID& keyid : setKeys) {
        if (mapKeyBirth.count(keyid) == 0)
            mapKeyFirstBlock[keyid] = pindexMax;
    }
    setKeys.clear();

    // if there are no such keys, we're done
    if (mapKeyFirstBlock.empty())
        return;

    // find first block that affects those keys, if there are any left
    std::vector<CKeyID> vAffected;
    for (std::map<uint256, CWalletTx>::const_iterator it = mapWallet.begin(); it != mapWallet.end(); it++) {
        // iterate over all wallet transactions...
        const CWalletTx& wtx = (*it).second;
        BlockMap::const_iterator blit = mapBlockIndex.find(wtx.hashBlock);
        if (blit != mapBlockIndex.end() && chainActive.Contains(blit->second)) {
            // ... which are already in a block
            int nHeight = blit->second->nHeight;
            for (const CTxOut& txout : wtx.tx->vout) {
                // iterate over all their outputs
                CAffectedKeysVisitor(*this, vAffected).Process(txout.scriptPubKey);
                for (const CKeyID& keyid : vAffected) {
                    // ... and all their affected keys
                    std::map<CKeyID, CBlockIndex*>::iterator rit = mapKeyFirstBlock.find(keyid);
                    if (rit != mapKeyFirstBlock.end() && nHeight < rit->second->nHeight)
                        rit->second = blit->second;
                }
                vAffected.clear();
            }
        }
    }

    // Extract block timestamps for those keys
    for (std::map<CKeyID, CBlockIndex*>::const_iterator it = mapKeyFirstBlock.begin(); it != mapKeyFirstBlock.end(); it++)
        mapKeyBirth[it->first] = it->second->GetBlockTime() - 7200; // block times can be 2h off
}

bool CWallet::AddDestData(const CTxDestination& dest, const std::string& key, const std::string& value)
{
    if (boost::get<CNoDestination>(&dest))
        return false;

    mapAddressBook[dest].destdata.insert(std::make_pair(key, value));
    if (!fFileBacked)
        return true;
    return CWalletDB(strWalletFile).WriteDestData(CDynamicAddress(dest).ToString(), key, value);
}

bool CWallet::EraseDestData(const CTxDestination& dest, const std::string& key)
{
    if (!mapAddressBook[dest].destdata.erase(key))
        return false;
    if (!fFileBacked)
        return true;
    return CWalletDB(strWalletFile).EraseDestData(CDynamicAddress(dest).ToString(), key);
}

bool CWallet::LoadDestData(const CTxDestination& dest, const std::string& key, const std::string& value)
{
    mapAddressBook[dest].destdata.insert(std::make_pair(key, value));
    return true;
}

bool CWallet::GetDestData(const CTxDestination& dest, const std::string& key, std::string* value) const
{
    std::map<CTxDestination, CAddressBookData>::const_iterator i = mapAddressBook.find(dest);
    if (i != mapAddressBook.end()) {
        CAddressBookData::StringMap::const_iterator j = i->second.destdata.find(key);
        if (j != i->second.destdata.end()) {
            if (value)
                *value = j->second;
            return true;
        }
    }
    return false;
}

CWallet* CWallet::CreateWalletFromFile(const std::string walletFile, const bool fImportMnemonic)
{
   
    // needed to restore wallet transaction meta data after -zapwallettxes
    std::vector<CWalletTx> vWtx;

    if (gArgs.GetBoolArg("-zapwallettxes", false)) {
        uiInterface.InitMessage(_("Zapping all transactions from wallet..."));

        CWallet* tempWallet = new CWallet(walletFile);
        DBErrors nZapWalletRet = tempWallet->ZapWalletTx(vWtx);
        if (nZapWalletRet != DB_LOAD_OK) {
            InitError(strprintf(_("Error loading %s: Wallet corrupted"), walletFile));
            return nullptr;
        }

        delete tempWallet;
        tempWallet = nullptr;
    }

    uiInterface.InitMessage(_("Loading wallet..."));

    int64_t nStart = GetTimeMillis();
    bool fFirstRun = true;
    CWallet* walletInstance = new CWallet(walletFile);
    DBErrors nLoadWalletRet = walletInstance->LoadWallet(fFirstRun,true);
    if (nLoadWalletRet != DB_LOAD_OK) {
        if (nLoadWalletRet == DB_CORRUPT) {
            InitError(strprintf(_("Error loading %s: Wallet corrupted"), walletFile));
            return nullptr;
        } else if (nLoadWalletRet == DB_NONCRITICAL_ERROR) {
            InitWarning(strprintf(_("Error reading %s! All keys read correctly, but transaction data"
                                    " or address book entries might be missing or incorrect."),
                walletFile));
        } else if (nLoadWalletRet == DB_TOO_NEW) {
            InitError(strprintf(_("Error loading %s: Wallet requires newer version of %s"), walletFile, _(PACKAGE_NAME)));
            return nullptr;
        } else if (nLoadWalletRet == DB_NEED_REWRITE) {
            InitError(strprintf(_("Wallet needed to be rewritten: restart %s to complete"), _(PACKAGE_NAME)));
            return nullptr;
        } else {
            InitError(strprintf(_("Error loading %s"), walletFile));
            return nullptr;
        }
    }

    if (gArgs.GetBoolArg("-upgradewallet", fFirstRun)) {
        int nMaxVersion = gArgs.GetArg("-upgradewallet", 0);
        if (nMaxVersion == 0) // the -upgradewallet without argument case
        {
            LogPrintf("Performing wallet upgrade to %i\n", FEATURE_LATEST);
            nMaxVersion = CLIENT_VERSION;
            walletInstance->SetMinVersion(FEATURE_LATEST); // permanently upgrade the wallet immediately
        } else
            LogPrintf("Allowing wallet upgrade up to %i\n", nMaxVersion);
        if (nMaxVersion < walletInstance->GetVersion()) {
            InitError(_("Cannot downgrade wallet"));
            return nullptr;
        }
        walletInstance->SetMaxVersion(nMaxVersion);
    }

    if (fFirstRun) {
        // Create new keyUser and set as default key
        if (gArgs.GetBoolArg("-usehd", DEFAULT_USE_HD_WALLET) && !walletInstance->IsHDEnabled()) {
            if (gArgs.GetArg("-mnemonic", "").size() > 512) { //RESTRICTION REMOVED: was 256 
                InitError(_("Mnemonic is too long, must be at most 256 characters")); //these were checking passphrase but think it should be mnemonic
                return nullptr;
            }
            // generate a new master key
            walletInstance->GenerateNewHDChain();
            // ensure this wallet.dat can only be opened by clients supporting HD
            walletInstance->SetMinVersion(FEATURE_HD);
        }

        if (fImportMnemonic) {
            pwalletMain->ShowProgress("", 50); //show we're working in the background...
        }

        CPubKey newDefaultKey;
        std::vector<unsigned char> newDefaultEdKey;
        if (walletInstance->GetKeysFromPool(newDefaultKey, newDefaultEdKey, false)) {
            walletInstance->SetDefaultKey(newDefaultKey);
            if (!walletInstance->SetAddressBook(walletInstance->vchDefaultKey.GetID(), "", "receive")) {
                InitError(_("Cannot write default address") += "\n");
                return nullptr;
            }
        }

        if (fImportMnemonic) pwalletMain->ShowProgress("", 75); //show we're working in the background...

        walletInstance->SetBestChain(chainActive.GetLocator());

        // Try to create wallet backup right after new wallet was created
        std::string strBackupWarning;
        std::string strBackupError;

        gArgs.ForceRemoveArg("-skipmnemonicbackup"); //reset so backup occurs next time

        if (!AutoBackupWallet(walletInstance, "", strBackupWarning, strBackupError)) {
            if (!strBackupWarning.empty()) {
                InitWarning(strBackupWarning);
            }
            if (!strBackupError.empty()) {
                InitError(strBackupError);
                return nullptr;
            }
        }

    } else if (gArgs.IsArgSet("-usehd")) {
        bool useHD = gArgs.GetBoolArg("-usehd", DEFAULT_USE_HD_WALLET);
        if (walletInstance->IsHDEnabled() && !useHD) {
            InitError(strprintf(_("Error loading %s: You can't disable HD on a already existing HD wallet"),
                walletInstance->strWalletFile));
            return nullptr;
        }
        if (!walletInstance->IsHDEnabled() && useHD) {
            InitError(strprintf(_("Error loading %s: You can't enable HD on a already existing non-HD wallet"),
                walletInstance->strWalletFile));
            return nullptr;
        }
    }

    // Warn user every time he starts non-encrypted HD wallet
    //Do not execute the following if Importing a Mnemonic
    if (!fImportMnemonic) {  
        if (gArgs.GetBoolArg("-usehd", DEFAULT_USE_HD_WALLET) && !walletInstance->IsLocked()) {
            InitWarning(_("Make sure to encrypt your wallet and delete all non-encrypted backups after you verified that wallet works!"));
        }
    }

    LogPrintf(" wallet      %15dms\n", GetTimeMillis() - nStart);

    if (fImportMnemonic) pwalletMain->ShowProgress("", 90); //show we're working in the background...

    RegisterValidationInterface(walletInstance);

    CBlockIndex* pindexRescan = chainActive.Tip();
    if (gArgs.GetBoolArg("-rescan", false))
        pindexRescan = chainActive.Genesis();
    else {
        CWalletDB walletdb(walletFile);
        CBlockLocator locator;
        if (walletdb.ReadBestBlock(locator))
            pindexRescan = FindForkInGlobalIndex(chainActive, locator);
        else
            pindexRescan = chainActive.Genesis();
    }
    if (chainActive.Tip() && chainActive.Tip() != pindexRescan) {
        //We can't rescan beyond non-pruned blocks, stop and throw an error
        //this might happen if a user uses a old wallet within a pruned node
        // or if he ran -disablewallet for a longer time, then decided to re-enable
        if (fPruneMode) {
            CBlockIndex* block = chainActive.Tip();
            while (block && block->pprev && (block->pprev->nStatus & BLOCK_HAVE_DATA) && block->pprev->nTx > 0 && pindexRescan != block)
                block = block->pprev;

            if (pindexRescan != block) {
                InitError(_("Prune: last wallet synchronisation goes beyond pruned data. You need to -reindex (download the whole blockchain again in case of pruned node)"));
                return nullptr;
            }
        }

        uiInterface.InitMessage(_("Rescanning..."));
        LogPrintf("Rescanning last %i blocks (from block %i)...\n", chainActive.Height() - pindexRescan->nHeight, pindexRescan->nHeight);
        nStart = GetTimeMillis();
        walletInstance->ScanForWalletTransactions(pindexRescan, true);

        //rescan if boolean is set. go back 100 transactions from most recent transaction involving me.
        while ((walletInstance->fNeedToRescanTransactions) && (walletInstance->ReserveKeyCount > 0)) {
            //initialize values
            walletInstance->fNeedToRescanTransactions = false;
            walletInstance->ReserveKeyCount = 0;

            CBlockIndex* computed_rescan_index = walletInstance->rescan_index;
            if (computed_rescan_index->nHeight > 100) {
                computed_rescan_index = chainActive[computed_rescan_index->nHeight - 100];
            }
            walletInstance->ScanForWalletTransactions(computed_rescan_index, true);

        }
        
        LogPrintf(" rescan      %15dms\n", GetTimeMillis() - nStart);
        walletInstance->SetBestChain(chainActive.GetLocator());
        CWalletDB::IncrementUpdateCounter();

        // Restore wallet transaction metadata after -zapwallettxes=1
        if (gArgs.GetBoolArg("-zapwallettxes", false) && gArgs.GetArg("-zapwallettxes", "1") != "2") {
            CWalletDB walletdb(walletFile);

            for (const CWalletTx& wtxOld : vWtx) {
                uint256 hash = wtxOld.GetHash();
                std::map<uint256, CWalletTx>::iterator mi = walletInstance->mapWallet.find(hash);
                if (mi != walletInstance->mapWallet.end()) {
                    const CWalletTx* copyFrom = &wtxOld;
                    CWalletTx* copyTo = &mi->second;
                    copyTo->mapValue = copyFrom->mapValue;
                    copyTo->vOrderForm = copyFrom->vOrderForm;
                    copyTo->nTimeReceived = copyFrom->nTimeReceived;
                    copyTo->nTimeSmart = copyFrom->nTimeSmart;
                    copyTo->fFromMe = copyFrom->fFromMe;
                    copyTo->strFromAccount = copyFrom->strFromAccount;
                    copyTo->nOrderPos = copyFrom->nOrderPos;
                    walletdb.WriteTx(*copyTo);
                }
            }
        }
    }
    walletInstance->SetBroadcastTransactions(gArgs.GetBoolArg("-walletbroadcast", DEFAULT_WALLETBROADCAST));

    {
        LOCK(walletInstance->cs_wallet);
        LogPrintf("setExternalKeyPool.size() = %u\n", walletInstance->KeypoolCountExternalKeys());
        LogPrintf("setInternalKeyPool.size() = %u\n", walletInstance->KeypoolCountInternalKeys());
        LogPrintf("setExternalEdKeyPool.size() = %u\n", walletInstance->EdKeypoolCountExternalKeys());
        LogPrintf("setInternalEdKeyPool.size() = %u\n", walletInstance->EdKeypoolCountInternalKeys());
        LogPrintf("mapWallet.size() = %u\n", walletInstance->mapWallet.size());
        LogPrintf("mapAddressBook.size() = %u\n", walletInstance->mapAddressBook.size());
    }

    return walletInstance;
}

bool CWallet::InitLoadWallet()
{
    if (gArgs.GetBoolArg("-disablewallet", DEFAULT_DISABLE_WALLET)) {
        pwalletMain = nullptr;
        LogPrintf("Wallet disabled!\n");
        return true;
    }

    std::string walletFile = gArgs.GetArg("-wallet", DEFAULT_WALLET_DAT);

    CWallet* const pwallet = CreateWalletFromFile(walletFile);
    if (!pwallet) {
        return false;
    }
    pwalletMain = pwallet;

    if (pwallet->GetVersion() < FEATURE_HD) {
        LogPrintf("%s - Older wallet version detected. Need to upgrade.\n", __func__);
        if (!pwalletMain->IsLocked()) {
            if (!pwallet->SyncEdKeyPool()) {
                LogPrintf("%s - ERROR: Unable to sync Ed25519 Keypool.\n", __func__);
            }
            else {
                LogPrintf("%s - Upgrading wallet version\n", __func__);
                pwallet->SetMinVersion(FEATURE_HD);
                pwallet->fNeedToUpgradeWallet = false;
            }
        }
        else {
            LogPrintf("%s - Upgrade wallet pending unlock\n", __func__);
            pwallet->fNeedToUpgradeWallet = true; 
        }
    }

    ProcessLinkQueue(); // Process links in queue.

    return true;
}

std::atomic<bool> CWallet::fFlushThreadRunning(false);

void CWallet::postInitProcess(boost::thread_group& threadGroup)
{
    // Add wallet transactions that aren't already in a block to mempool
    // Do this here as mempool requires genesis block to be loaded
    ReacceptWalletTransactions();

    // Run a thread to flush wallet periodically
    if (!CWallet::fFlushThreadRunning.exchange(true)) {
        threadGroup.create_thread(ThreadFlushWalletDB);
    }
}

bool CWallet::InitAutoBackup()
{
    if (gArgs.GetBoolArg("-disablewallet", DEFAULT_DISABLE_WALLET))
        return true;

    std::string strWarning;
    std::string strError;

    nWalletBackups = gArgs.GetArg("-createwalletbackups", 10);
    nWalletBackups = std::max(0, std::min(10, nWalletBackups));

    std::string strWalletFile = gArgs.GetArg("-wallet", DEFAULT_WALLET_DAT);

    if (!AutoBackupWallet(nullptr, strWalletFile, strWarning, strError)) {
        if (!strWarning.empty())
            InitWarning(strWarning);
        if (!strError.empty())
            return InitError(strError);
    }

    return true;
}

bool CWallet::BackupWallet(const std::string& strDest)
{
    if (!fFileBacked)
        return false;
    while (true) {
        {
            LOCK(bitdb.cs_db);
            if (!bitdb.mapFileUseCount.count(strWalletFile) || bitdb.mapFileUseCount[strWalletFile] == 0) {
                // Flush log data to the dat file
                bitdb.CloseDb(strWalletFile);
                bitdb.CheckpointLSN(strWalletFile);
                bitdb.mapFileUseCount.erase(strWalletFile);

                // Copy wallet file
                boost::filesystem::path pathSrc = GetDataDir() / strWalletFile;
                boost::filesystem::path pathDest(strDest);
                if (boost::filesystem::is_directory(pathDest))
                    pathDest /= strWalletFile;

                try {
#if BOOST_VERSION >= 104000
                    boost::filesystem::copy_file(pathSrc, pathDest, boost::filesystem::copy_option::overwrite_if_exists);
#else
                    boost::filesystem::copy_file(pathSrc, pathDest);
#endif
                    LogPrintf("copied %s to %s\n", strWalletFile, pathDest.string());
                    return true;
                } catch (const boost::filesystem::filesystem_error& e) {
                    LogPrintf("error copying %s to %s - %s\n", strWalletFile, pathDest.string(), e.what());
                    return false;
                }
            }
        }
        MilliSleep(100);
    }
    return false;
}

//! Stealth Address Support
bool CWallet::GetStealthAddress(const CKeyID& keyid, CStealthAddress& sxAddr) const
{
    LOCK(cs_mapStealthAddresses);
    std::map<CKeyID, CStealthAddress>::const_iterator it = mapStealthAddresses.find(keyid);
    if (it != mapStealthAddresses.end()) {
        LogPrintf("CWallet::%s -- found %s\n", __func__, CDynamicAddress(keyid).ToString());
        sxAddr = (*it).second;
        return true;
    }
    LogPrintf("CWallet::%s -- Not found %s\n", __func__, CDynamicAddress(keyid).ToString());
    return false;
}

inline bool MatchPrefix(uint32_t nAddrBits, uint32_t addrPrefix, uint32_t outputPrefix, bool fHavePrefix)
{
    if (nAddrBits < 1) { // addresses without prefixes scan all incoming stealth outputs
        return true;
    }
    if (!fHavePrefix) { // don't check when address has a prefix and no prefix on output
        return false;
    }

    uint32_t mask = SetStealthMask(nAddrBits);

    return (addrPrefix & mask) == (outputPrefix & mask);
}

bool CWallet::ProcessStealthQueue()
{
    CWalletDB* pwdb = GetWalletDB();

    if (IsLocked()) {
        delete pwdb;
        return false;
    }

    LOCK(cs_vStealthKeyQueue);
    for (const std::pair<CKeyID, CStealthKeyQueueData>& data : vStealthKeyQueue) {
        CStealthKeyQueueData stealthData = data.second;
        CKey sSpend;
        if (!GetKey(stealthData.pkSpend.GetID(), sSpend)) {
            LogPrintf("%s: Error getting spend private key (%s) for stealth transaction.\n", __func__,  CDynamicAddress(stealthData.pkSpend.GetID()).ToString());
            continue;
        }
        CKey sSpendR;
        if (StealthSharedToSecretSpend(stealthData.SharedKey, sSpend, sSpendR) != 0) {
            LogPrintf("%s: StealthSharedToSecretSpend() failed.\n", __func__);
            continue;
        }
        CPubKey pkT = sSpendR.GetPubKey();
        if (!pkT.IsValid()) {
            LogPrintf("%s: pkT is invalid.\n", __func__);
            continue;
        }
        CKeyID keyID = pkT.GetID();
        if (keyID != data.first) {
            LogPrintf("%s: Spend key mismatch!\n", __func__);
            continue;
        }
        CKeyMetadata meta(GetTime());
        mapKeyMetadata[pkT.GetID()] = meta;
        if (!AddKeyPubKey(sSpendR, pkT)) {
            LogPrintf("%s: Stealth spending AddKeyPubKey failed.\n", __func__);
            continue;
        }
        nFoundStealth++;
        pwdb->EraseStealthKeyQueue(data.first);
    }
    vStealthKeyQueue.clear();
    delete pwdb;
    return true;
}

bool RunProcessStealthQueue()
{
    if (!pwalletMain)
        return false;

    return pwalletMain->ProcessStealthQueue();
}

bool CWallet::ProcessStealthOutput(const CTxDestination& address, std::vector<uint8_t>& vchEphemPK, uint32_t prefix, bool fHavePrefix, CKey& sShared)
{
    CWalletDB* pwdb = GetWalletDB();

    CKeyID idMatchShared = boost::get<CKeyID>(address);
    ec_point pkExtracted;
    CKey sSpend;
    LOCK(cs_mapStealthAddresses);
    std::map<CKeyID, CStealthAddress>::const_iterator it;
    for (it = mapStealthAddresses.begin(); it != mapStealthAddresses.end(); ++it) {
        CStealthAddress sxAddr = (*it).second;
        if (sxAddr.IsNull())
            continue;

        if (!MatchPrefix(sxAddr.prefix_number_bits, sxAddr.prefix_bitfield, prefix, fHavePrefix)) {
            continue;
        }
        if (!sxAddr.scan_secret.IsValid()) {
            continue; // invalid scan_secret
        }

        if (StealthSecret(sxAddr.scan_secret, vchEphemPK, sxAddr.spend_pubkey, sShared, pkExtracted) != 0) {
            LogPrintf("%s: StealthSecret failed.\n", __func__);
            continue;
        }

        CPubKey pkE(pkExtracted);
        if (!pkE.IsValid()) {
            continue;
        }
        CKeyID idExtracted = pkE.GetID();
        if (idMatchShared != idExtracted) {
            continue;
        }
        LogPrint("bdap", "%s -- Found txn output from address %s belongs to stealth address %s\n", __func__, CDynamicAddress(idExtracted).ToString(), sxAddr.Encoded());

        if (IsLocked()) {
            LogPrintf("%s: Wallet locked, adding stealth key to queue wallet.\n", __func__);
            // Add key without secret
            std::vector<uint8_t> vchEmpty;
            AddCryptedKey(pkE, vchEmpty);
            CPubKey cpkEphem(vchEphemPK);
            CPubKey cpkScan(sxAddr.scan_pubkey);
            CPubKey cpkSpend(sxAddr.spend_pubkey);
            CStealthKeyQueueData lockedSkQueueData(cpkEphem, cpkScan, cpkSpend, sShared);
            if (!pwdb->WriteStealthKeyQueue(idMatchShared, lockedSkQueueData)) {
                LogPrintf("%s: Error WriteStealthKeyQueue failed for %s.\n", __func__, CDynamicAddress(idExtracted).ToString());
                delete pwdb;
                return false;
            }
            LOCK(cs_vStealthKeyQueue);
            vStealthKeyQueue.push_back(std::make_pair(idExtracted, lockedSkQueueData));
            nFoundStealth++;
            delete pwdb;
            return true;
        }

        if (!GetKey(sxAddr.GetSpendKeyID(), sSpend)) {
            LogPrintf("%s: Error getting spend private key (%s) for stealth transaction.\n", __func__,  CDynamicAddress(sxAddr.GetSpendKeyID()).ToString());
            delete pwdb;
            return false;
        }
        CKey sSpendR;
        if (StealthSharedToSecretSpend(sShared, sSpend, sSpendR) != 0) {
            LogPrintf("%s: StealthSharedToSecretSpend() failed.\n", __func__);
            continue;
        }
        CPubKey pkT = sSpendR.GetPubKey();
        if (!pkT.IsValid()) {
            LogPrintf("%s: pkT is invalid.\n", __func__);
            continue;
        }
        CKeyID keyID = pkT.GetID();
        if (keyID != idMatchShared) {
            LogPrintf("%s: Spend key mismatch!\n", __func__);
            continue;
        }

        if (!AddKeyPubKey(sSpendR, pkT)) {
            continue;
        }
        nFoundStealth++;
        delete pwdb;
        return true;
    }
    delete pwdb;
    return false;
}

// TODO (BDAP): Move to script code file
bool IsDataScript(const CScript& data)
{
    CScript::const_iterator pc = data.begin();
    opcodetype opcode;
    if (!data.GetOp(pc, opcode))
        return false;
    if(opcode != OP_RETURN)
        return false;
    std::vector<uint8_t> vData;
    if (!data.GetOp(pc, opcode, vData))
        return false;

    return true;
}

// TODO (BDAP): Move to script code file
bool GetDataFromScript(const CScript& data, std::vector<uint8_t>& vData)
{
    CScript::const_iterator pc = data.begin();
    opcodetype opcode;
    if (!data.GetOp(pc, opcode))
        return false;
    if(opcode != OP_RETURN)
        return false;
    if (!data.GetOp(pc, opcode, vData))
        return false;

    return true;
}

// returns: -1 error, 0 nothing found, 2 stealth found
int CWallet::CheckForStealthTxOut(const CTxOut* pTxOut, const CTxOut* pTxData)
{
    LogPrint("stealth", "%s -- txout = %s\n", __func__, pTxOut->ToString());
    CKey sShared;
    std::vector<uint8_t> vchEphemPK;
    std::vector<uint8_t> vData;
    if (!GetDataFromScript(pTxData->scriptPubKey, vData)) {
        LogPrintf("%s -- GetDataFromScript failed.\n", __func__);
        return 0;
    }

    if (vData.size() < 1) {
        return -1;
    }

    if (vData[0] == DO_STEALTH) {
        if (vData.size() < 34 ) {
            return -1; // error
        }
        if (vData.size() > 40) {
            return 0; // Stealth OP_RETURN data is always less then 40 bytes
        }
        vchEphemPK.resize(33);
        memcpy(&vchEphemPK[0], &vData[1], 33);

        uint32_t prefix = 0;
        bool fHavePrefix = false;
        if (vData.size() >= 34 + 5 && vData[34] == DO_STEALTH_PREFIX) {
            fHavePrefix = true;
            memcpy(&prefix, &vData[35], 4);
        }

        CTxDestination address;
        if (!ExtractDestination(pTxOut->scriptPubKey, address)) {
            LogPrintf("%s: ExtractDestination failed.\n",  __func__);
            return -1;
        }

        if (address.type() != typeid(CKeyID)) {
            LogPrintf("%s: address.type() != typeid(CKeyID) failed\n",  __func__);
            return -1;
        }

        if (!ProcessStealthOutput(address, vchEphemPK, prefix, fHavePrefix, sShared)) {
            return 0;
        }
        return 2;
    }

    LogPrintf("%s: Unknown data output type %d.\n",  __func__, vData[0]);
    return -1;
}

bool CWallet::HasBDAPLinkTx(const CTransaction& tx, CScript& bdapOpScript)
{
    // Only support stealth when using links
    if (tx.nVersion == BDAP_TX_VERSION) {
        int op1, op2;
        std::vector<std::vector<unsigned char>> vvchOpParameters;
        CTransactionRef ptx = MakeTransactionRef(tx);
        if (GetBDAPOpScript(ptx, bdapOpScript, vvchOpParameters, op1, op2)) {
            std::string strOpType = GetBDAPOpTypeString(op1, op2);
            if (GetOpCodeType(strOpType) == "link" && vvchOpParameters.size() > 1) {
                return true;
            }
        }
    }
    return false;
}

bool CWallet::ScanForStealthOwnedOutputs(const CTransaction& tx)
{
    bool fIsMine = false;
    CScript bdapOpScript;
    //if (HasBDAPLinkTx(tx, bdapOpScript)) { // Only support stealth when using links
        AssertLockHeld(cs_wallet);

        bool fDataFound = false;
        CTxOut txOutData;
        for (const CTxOut& txData : tx.vout) {
            // TODO (BDAP): Do not count BDAP OP_RETURN account.
            bool fIsData = IsDataScript(txData.scriptPubKey);
            if (fIsData) {
                std::vector<uint8_t> vData;
                if (GetDataFromScript(txData.scriptPubKey, vData)) {
                    if (vData.size() < 65) {
                        txOutData = txData;
                        fDataFound = true;
                        LogPrint("stealth", "%s -- ASM Script = %s, txOutData = %s\n", __func__, ScriptToAsmStr(txOutData.scriptPubKey), txOutData.ToString());
                        break;
                    }
                }
            }
        }

        if (fDataFound) {
            int32_t nOutputId = 0;
            for (const CTxOut& txOut : tx.vout) {
                bool fIsData = IsDataScript(txOut.scriptPubKey);
                if (!fIsData) {
                    LogPrint("stealth", "%s --Checking txOut for my stealth %s\n",  __func__, txOut.ToString());
                    int nResult = CheckForStealthTxOut(&txOut, &txOutData);
                    if (nResult < 0) {
                        LogPrintf("%s txn %s, malformed data output %d.\n",  __func__, tx.GetHash().ToString(), nOutputId);
                    }
                    else if (nResult == 2)
                    {
                        if (IsMine(txOut)) {
                            fIsMine = true;
                        }
                    }
                }
                nOutputId++;
            }
        }
    //}
    return fIsMine;
}

bool CWallet::AddStealthAddress(const CStealthAddress& sxAddr)
{
    CWalletDB* pwdb = GetWalletDB();

    LOCK(cs_mapStealthAddresses);
    if (!pwdb->WriteStealthAddress(sxAddr)) {
        delete pwdb;
        return error("%s: WriteStealthAddress failed.", __func__);
    }
    mapStealthAddresses[sxAddr.GetSpendKeyID()] = sxAddr;
    delete pwdb;
    return true;
}

bool CWallet::AddStealthToMap(const std::pair<CKeyID, CStealthAddress>& pairStealthAddress)
{
    LOCK(cs_mapStealthAddresses);
    mapStealthAddresses[pairStealthAddress.first] = pairStealthAddress.second;
    return true;
}

bool CWallet::AddToStealthQueue(const std::pair<CKeyID, CStealthKeyQueueData>& pairStealthQueue)
{
    LOCK(cs_vStealthKeyQueue);
    vStealthKeyQueue.push_back(pairStealthQueue);
    return true;
}

CWalletDB* CWallet::GetWalletDB()
{
    CWalletDB* pwdb = new CWalletDB(strWalletFile, "r+");
    assert(pwdb);
    return pwdb;
}

bool CWallet::HaveStealthAddress(const CKeyID& address) const
{
    LOCK(cs_mapStealthAddresses);
    if (mapStealthAddresses.count(address) > 0)
        return true;
    return false;
}
//! End Stealth Address Support

//! Start Proof-of-Stake Support
bool CWallet::MintableCoins()
{
    LOCK(cs_main);
    CAmount nBalance = GetBalance();

    int chainHeight = chainActive.Height();

    if (nBalance > 0) {
        if (gArgs.IsArgSet("-reservebalance") && !ParseMoney(gArgs.GetArg("-reservebalance", "0"), nReserveBalance))
            return error("%s : invalid reserve balance amount", __func__);
        if (nBalance <= nReserveBalance)
            return false;

        std::vector<COutput> vCoins;
        AvailableCoins(vCoins, true);

        int64_t time = GetAdjustedTime();
        for (const COutput& out : vCoins) {
            CBlockIndex* utxoBlock = mapBlockIndex.at(out.tx->hashBlock);
            //check for maturity (min age/depth)
            if (Params().HasStakeMinDepth(chainHeight, utxoBlock->nHeight) && Params().HasStakeMinAge(time, utxoBlock->nTime))
                return true;
        }
    }

    return false;
}

bool CWallet::SelectStakeCoins(std::list<std::unique_ptr<CStakeInput> >& listInputs, const CAmount& nTargetAmount, const int& blockHeight, const bool fPrecompute)
{
    LOCK(cs_main);
    //Add DYN
    std::vector<COutput> vCoins;
    AvailableCoins(vCoins, true, nullptr, false, STAKABLE_COINS);
    CAmount nAmountSelected = 0;
    if (gArgs.GetBoolArg("-stake", true) && !fPrecompute) {
        for (const COutput &out : vCoins) {
            //make sure not to outrun target amount
            if (nAmountSelected + out.tx->tx->vout[out.i].nValue > nTargetAmount)
                continue;

            CBlockIndex* utxoBlock = mapBlockIndex.at(out.tx->hashBlock);
            const int nHeightBlockFrom = utxoBlock->nHeight + 1;

            //check for maturity (min age/depth)
            if (!Params().HasStakeMinDepth(blockHeight, nHeightBlockFrom))
                continue;

            if (!Params().HasStakeMinAge(GetAdjustedTime(), utxoBlock->GetBlockTime()))
                continue;

            if (blockHeight - nHeightBlockFrom < COINBASE_MATURITY)
                continue;

            //add to our stake set
            nAmountSelected += out.tx->tx->vout[out.i].nValue;

            std::unique_ptr<CDynamicStake> input(new CDynamicStake());
            input->SetInput((CTransaction) *out.tx->tx, out.i);
            listInputs.emplace_back(std::move(input));
        }
    }
    return true;
}

extern bool ShutdownRequested();

// ppcoin: create coin stake transaction
bool CWallet::CreateCoinStake(const CKeyStore& keystore, const unsigned int& nBits, const int64_t& nSearchInterval, CMutableTransaction& txNew, unsigned int& nTxNewTime)
{
    // The following split & combine thresholds are important to security
    // Should not be adjusted if you don't understand the consequences
    //int64_t nCombineThreshold = 0;
    const CBlockIndex* pindexPrev = chainActive.Tip();
    txNew.vin.clear();
    txNew.vout.clear();

    // Mark coin stake transaction
    CScript scriptEmpty;
    scriptEmpty.clear();
    txNew.vout.push_back(CTxOut(0, scriptEmpty));

    // Choose coins to use
    CAmount nBalance = GetBalance();
    if (gArgs.IsArgSet("-reservebalance") && !ParseMoney(gArgs.GetArg("-reservebalance", "0"), nReserveBalance))
        return error("CreateCoinStake : invalid reserve balance amount");

    if (nBalance > 0 && nBalance <= nReserveBalance)
        return false;

    // Get the list of stakable inputs
    std::list<std::unique_ptr<CStakeInput> > listInputs;
    if (!SelectStakeCoins(listInputs, nBalance - nReserveBalance, pindexPrev->nHeight + 1)) {
        LogPrintf("CreateCoinStake(): selectStakeCoins failed\n");
        return false;
    }

    if (listInputs.empty()) {
        LogPrint("staking", "CreateCoinStake(): listInputs empty\n");
        MilliSleep(50000);
        return false;
    }

    if (GetAdjustedTime() - chainActive.Tip()->GetBlockTime() < 60) {
        if (Params().NetworkIDString() == CBaseChainParams::REGTEST) {
            MilliSleep(1000);
        }
    }

    CAmount nCredit;
    CScript scriptPubKeyKernel;
    bool fKernelFound = false;
    int nAttempts = 0;

    // Block time.
    nTxNewTime = GetAdjustedTime();
    // If the block time is in the future, then starts there.
    if (pindexPrev->nTime > nTxNewTime) {
        nTxNewTime = pindexPrev->nTime;
    }

    for (std::unique_ptr<CStakeInput>& stakeInput : listInputs) {
        nCredit = 0;
        // Make sure the wallet is unlocked and shutdown hasn't been requested
        if (IsLocked() || ShutdownRequested())
            return false;

        uint256 hashProofOfStake = uint256Zero;
        nAttempts++;
        //iterates each utxo inside of CheckStakeKernelHash()
        if (Stake(pindexPrev, stakeInput.get(), nBits, nTxNewTime, hashProofOfStake)) {

            // Found a kernel
            LogPrintf("CreateCoinStake : kernel found\n");
            nCredit += stakeInput->GetValue();

            // Calculate reward
            CAmount nReward;
            nReward = GetFluidStakingReward(chainActive.Height() + 1);
            nCredit += nReward;

            // Create the output transaction(s)
            std::vector<CTxOut> vout;
            if (!stakeInput->CreateTxOuts(this, vout, nCredit)) {
                LogPrintf("%s : failed to get scriptPubKey\n", __func__);
                continue;
            }
            txNew.vout.insert(txNew.vout.end(), vout.begin(), vout.end());

            CAmount nMinFee = 0;

            // Set output amount
            // Do not pay Dynodes
            if (txNew.vout.size() == 3) {
                txNew.vout[1].nValue = ((nCredit - nMinFee) / 2 / CENT) * CENT;
                txNew.vout[2].nValue = nCredit - nMinFee - txNew.vout[1].nValue;
            } else {
                txNew.vout[1].nValue = nCredit - nMinFee;
            }

            // Limit size
            unsigned int nBytes = ::GetSerializeSize(txNew, SER_NETWORK, PROTOCOL_VERSION);
            if (nBytes >= DEFAULT_BLOCK_MAX_SIZE / 5)
                return error("CreateCoinStake : exceeded coinstake size limit");

            {
                uint256 hashTxOut = txNew.GetHash();
                CTxIn in;
                if (!stakeInput->CreateTxIn(this, in, hashTxOut)) {
                    LogPrintf("%s : failed to create TxIn\n", __func__);
                    txNew.vin.clear();
                    txNew.vout.clear();
                    continue;
                }
                txNew.vin.emplace_back(in);
            }

            fKernelFound = true;
            break;
        }
    }
    LogPrint("staking", "%s: attempted staking %d times\n", __func__, nAttempts);

    if (!fKernelFound)
        return false;

    // Sign Proof-of-Stake
    unsigned int nIn = 0;
    for (CTxIn txIn : txNew.vin) {
        const CWalletTx *wtx = GetWalletTx(txIn.prevout.hash);
        if (!SignSignature(*this, *wtx, txNew, nIn++))
            return error("CreateCoinStake : failed to sign coinstake");
    }

    // Successfully generated coinstake
    return true;
}
//! End Proof-of-Stake Support

// This should be called carefully:
// either supply "wallet" (if already loaded) or "strWalletFile" (if wallet wasn't loaded yet)
bool AutoBackupWallet(CWallet* wallet, std::string strWalletFile, std::string& strBackupWarning, std::string& strBackupError)
{

    //don't do an initial backup if importing mnemonic 
    if (gArgs.GetBoolArg("-skipmnemonicbackup", false)) {
        gArgs.ForceRemoveArg("-skipmnemonicbackup"); //reset, so backup next time
        return true;
    } //end skipmnemonicbackup

    namespace fs = boost::filesystem;

    strBackupWarning = strBackupError = "";

    if (nWalletBackups > 0) {
        fs::path backupsDir = GetBackupsDir();

        if (!fs::exists(backupsDir)) {
            // Always create backup folder to not confuse the operating system's file browser
            LogPrintf("Creating backup folder %s\n", backupsDir.string());
            if (!fs::create_directories(backupsDir)) {
                // smth is wrong, we shouldn't continue until it's resolved
                strBackupError = strprintf(_("Wasn't able to create wallet backup folder %s!"), backupsDir.string());
                LogPrintf("%s\n", strBackupError);
                nWalletBackups = -1;
                return false;
            }
        } else if (!fs::is_directory(backupsDir)) {
            // smth is wrong, we shouldn't continue until it's resolved
            strBackupError = strprintf(_("%s is not a valid backup folder!"), backupsDir.string());
            LogPrintf("%s\n", strBackupError);
            nWalletBackups = -1;
            return false;
        }

        // Create backup of the ...
        std::string dateTimeStr = DateTimeStrFormat(".%Y-%m-%d-%H-%M-%S", GetTime());
        if (wallet) {
            // ... opened wallet
            LOCK2(cs_main, wallet->cs_wallet);
            strWalletFile = wallet->strWalletFile;
            fs::path backupFile = backupsDir / (strWalletFile + dateTimeStr);
            if (!wallet->BackupWallet(backupFile.string())) {
                strBackupWarning = strprintf(_("Failed to create backup %s!"), backupFile.string());
                LogPrintf("%s\n", strBackupWarning);
                nWalletBackups = -1;
                return false;
            }
            // Update nKeysLeftSinceAutoBackup using current external keypool size
            wallet->nKeysLeftSinceAutoBackup = wallet->KeypoolCountExternalKeys();
            LogPrintf("nKeysLeftSinceAutoBackup: %d\n", wallet->nKeysLeftSinceAutoBackup);
            if (wallet->IsLocked(true)) {
                strBackupWarning = _("Wallet is locked, can't replenish keypool! Automatic backups and mixing are disabled, please unlock your wallet to replenish keypool.");
                LogPrintf("%s\n", strBackupWarning);
                nWalletBackups = -2;
                return false;
            }
        } else {
            // ... strWalletFile file
            fs::path sourceFile = GetDataDir() / strWalletFile;
            fs::path backupFile = backupsDir / (strWalletFile + dateTimeStr);
            sourceFile.make_preferred();
            backupFile.make_preferred();
            if (fs::exists(backupFile)) {
                strBackupWarning = _("Failed to create backup, file already exists! This could happen if you restarted wallet in less than 60 seconds. You can continue if you are ok with this.");
                LogPrintf("%s\n", strBackupWarning);
                return false;
            }
            if (fs::exists(sourceFile)) {
                try {
                    fs::copy_file(sourceFile, backupFile);
                    LogPrintf("Creating backup of %s -> %s\n", sourceFile.string(), backupFile.string());
                } catch (fs::filesystem_error& error) {
                    strBackupWarning = strprintf(_("Failed to create backup, error: %s"), error.what());
                    LogPrintf("%s\n", strBackupWarning);
                    nWalletBackups = -1;
                    return false;
                }
            }
        }

        // Keep only the last 10 backups, including the new one of course
        typedef std::multimap<std::time_t, fs::path> folder_set_t;
        folder_set_t folder_set;
        fs::directory_iterator end_iter;
        backupsDir.make_preferred();
        // Build map of backup files for current(!) wallet sorted by last write time
        fs::path currentFile;
        for (fs::directory_iterator dir_iter(backupsDir); dir_iter != end_iter; ++dir_iter) {
            // Only check regular files
            if (fs::is_regular_file(dir_iter->status())) {
                currentFile = dir_iter->path().filename();
                // Only add the backups for the current wallet, e.g. wallet.dat.*
                if (dir_iter->path().stem().string() == strWalletFile) {
                    folder_set.insert(folder_set_t::value_type(fs::last_write_time(dir_iter->path()), *dir_iter));
                }
            }
        }

        // Loop backward through backup files and keep the N newest ones (1 <= N <= 10)
        int counter = 0;
        BOOST_REVERSE_FOREACH (PAIRTYPE(const std::time_t, fs::path) file, folder_set) {
            counter++;
            if (counter > nWalletBackups) {
                // More than nWalletBackups backups: delete oldest one(s)
                try {
                    fs::remove(file.second);
                    LogPrintf("Old backup deleted: %s\n", file.second);
                } catch (fs::filesystem_error& error) {
                    strBackupWarning = strprintf(_("Failed to delete backup, error: %s"), error.what());
                    LogPrintf("%s\n", strBackupWarning);
                    return false;
                }
            }
        }
        return true;
    }

    LogPrintf("Automatic wallet backups are disabled!\n");
    return false;
}

void CWallet::AutoCombineDust()
{
    LOCK2(cs_main, cs_wallet);
    const CBlockIndex* tip = chainActive.Tip();
    if (tip->nTime < (GetAdjustedTime() - 300) || IsLocked()) {
        return;
    }

    std::map<CDynamicAddress, std::vector<COutput> > mapCoinsByAddress = AvailableCoinsByAddress(true, nAutoCombineThreshold * COIN);

    //coins are sectioned by address. This combination code only wants to combine inputs that belong to the same address
    for (std::map<CDynamicAddress, std::vector<COutput> >::iterator it = mapCoinsByAddress.begin(); it != mapCoinsByAddress.end(); it++) {
        std::vector<COutput> vCoins, vRewardCoins;
        bool maxSize = false;
        vCoins = it->second;

        // We don't want the tx to be refused for being too large
        // we use 50 bytes as a base tx size (2 output: 2*34 + overhead: 10 -> 90 to be certain)
        unsigned int txSizeEstimate = 90;

        //find dynode rewards that need to be combined
        CCoinControl coinControl;
        CAmount nTotalRewardsValue = 0;
        for (const COutput& out : vCoins) {
            if (!out.fSpendable)
                continue;
            //no coins should get this far if they dont have proper maturity, this is double checking
            if (out.tx->IsCoinStake() && out.tx->GetDepthInMainChain() < COINBASE_MATURITY + 1)
                continue;

            COutPoint outpt(out.tx->GetHash(), out.i);
            coinControl.Select(outpt);
            vRewardCoins.push_back(out);
            nTotalRewardsValue += out.tx->tx->GetValueOut();

            // Combine to the threshold and not way above
            if (nTotalRewardsValue > nAutoCombineThreshold * COIN)
                break;

            // Around 180 bytes per input. We use 190 to be certain
            txSizeEstimate += 190;
            // TODO: remove for assets?
            if (txSizeEstimate >= MAX_STANDARD_TX_SIZE - 200) {
                maxSize = true;
                break;
            }
        }

        //if no inputs found then return
        if (!coinControl.HasSelected())
            continue;

        //we cannot combine one coin with itself
        if (vRewardCoins.size() <= 1)
            continue;

        std::vector<CRecipient> vecSend;
        CScript scriptPubKey = GetScriptForDestination(it->first.Get());
        CRecipient recipient = {scriptPubKey, nTotalRewardsValue, false};
        vecSend.push_back(recipient);

        //Send change to same address
        CTxDestination destMyAddress;
        if (!ExtractDestination(scriptPubKey, destMyAddress)) {
            LogPrintf("AutoCombineDust: failed to extract destination\n");
            continue;
        }
        coinControl.destChange = destMyAddress;

        // Create the transaction and commit it to the network
        CWalletTx wtx;
        CReserveKey keyChange(this); // this change address does not end up being used, because change is returned with coin control switch
        std::string strErr;
        CAmount nFeeRet = 0;
        int nChangePosInOut = 0;
        // 10% safety margin to avoid "Insufficient funds" errors
        vecSend[0].nAmount = nTotalRewardsValue - (nTotalRewardsValue / 10);

        if (!CreateTransaction(vecSend, wtx, keyChange, nFeeRet, nChangePosInOut, strErr, coinControl, true, ALL_COINS, false, false)) {
            LogPrintf("AutoCombineDust createtransaction failed, reason: %s\n", strErr);
            continue;
        }

        //we don't combine below the threshold unless the fees are 0 to avoid paying fees over fees over fees
        if (!maxSize && nTotalRewardsValue < nAutoCombineThreshold * COIN && nFeeRet > 0)
            continue;

        CValidationState state;
        if (!CommitTransaction(wtx, keyChange, g_connman.get(), state)) {
            LogPrintf("AutoCombineDust transaction commit failed\n");
            continue;
        }

        LogPrintf("AutoCombineDust sent transaction\n");

    }
}

bool CWallet::MultiSend()
{
    LOCK2(cs_main, cs_wallet);
    // Stop the old blocks from sending multisends
    const CBlockIndex* tip = chainActive.Tip();

    int chainTipHeight = tip->nHeight;
    if (tip->nTime < (GetAdjustedTime() - 300) || IsLocked()) {
        return false;
    }

    if (chainTipHeight <= nLastMultiSendHeight) {
        LogPrintf("Multisend: lastmultisendheight is higher than current best height\n");
        return false;
    }

    std::vector<COutput> vCoins;
    AvailableCoins(vCoins);
    bool stakeSent = false;
    bool mnSent = false;
    for (const COutput& out : vCoins) {

        //need output with precise confirm count - this is how we identify which is the output to send
        if (out.tx->GetDepthInMainChain() != COINBASE_MATURITY + 1)
            continue;

        COutPoint outpoint(out.tx->GetHash(), out.i);
        bool sendMSonDNReward = outpoint.IsDynodeReward(out.tx->tx.get());
        bool sendMSOnStake = fMultiSendStake && out.tx->tx->IsCoinStake() && !sendMSonDNReward; //output is either dnreward or stake reward, not both

        if (!(sendMSOnStake || sendMSonDNReward))
            continue;

        CTxDestination destMyAddress;
        if (!ExtractDestination(out.tx->tx->vout[out.i].scriptPubKey, destMyAddress)) {
            LogPrintf("Multisend: failed to extract destination\n");
            continue;
        }

        //Disabled Addresses won't send MultiSend transactions
        if (vDisabledAddresses.size() > 0) {
            for (unsigned int i = 0; i < vDisabledAddresses.size(); i++) {
                if (vDisabledAddresses[i] == CDynamicAddress(destMyAddress).ToString()) {
                    LogPrintf("Multisend: disabled address preventing multisend\n");
                    return false;
                }
            }
        }

        // create new coin control, populate it with the selected utxo, create sending vector
        CCoinControl coinControl;
        coinControl.Select(COutPoint(out.tx->GetHash(), out.i));
        coinControl.destChange = destMyAddress;

        CWalletTx wtx;
        CReserveKey keyChange(this); // this change address does not end up being used, because change is returned with coin control switch
        CAmount nFeeRet = 0;
        std::vector<CRecipient> vecSend;

        // loop through multisend vector and add amounts and addresses to the sending vector
        const isminefilter filter = ISMINE_SPENDABLE;
        CAmount nAmount = 0;
        for (unsigned int i = 0; i < vMultiSend.size(); i++) {
            // MultiSend vector is a pair of 1)Address as a std::string 2) Percent of stake to send as an int
            nAmount = ((out.tx->GetCredit(filter) - out.tx->GetDebit(filter)) * vMultiSend[i].second) / 100;
            CDynamicAddress strAddSend(vMultiSend[i].first);
            CScript scriptPubKey;
            scriptPubKey = GetScriptForDestination(strAddSend.Get());
            CRecipient recipient = { scriptPubKey, nAmount, false };
            vecSend.push_back(recipient);
        }

        //get the fee amount
        CWalletTx wtxdummy;
        std::string strErr;
        int nChangePosInOut = 0;
        CreateTransaction(vecSend, wtxdummy, keyChange, nFeeRet, nChangePosInOut, strErr, coinControl, true, ALL_COINS, false, false);
        CAmount nLastSendAmount = vecSend[vecSend.size() - 1].nAmount;
        if (nLastSendAmount < nFeeRet + 500) {
            LogPrintf("%s: fee of %d is too large to insert into last output\n", __func__, nFeeRet + 500);
            return false;
        }
        vecSend[vecSend.size() - 1].nAmount = nLastSendAmount - nFeeRet - 500;

        // Create the transaction and commit it to the network
        if (!CreateTransaction(vecSend, wtxdummy, keyChange, nFeeRet, nChangePosInOut, strErr, coinControl, true, ALL_COINS, false, false)) {
            LogPrintf("MultiSend createtransaction failed\n");
            return false;
        }
        CValidationState state;
        if (!CommitTransaction(wtx, keyChange, g_connman.get(), state)) {
            LogPrintf("MultiSend transaction commit failed\n");
            return false;
        } else
            fMultiSendNotify = true;

        //write nLastMultiSendHeight to DB
        CWalletDB walletdb(strWalletFile);
        nLastMultiSendHeight = chainActive.Tip()->nHeight;
        if (!walletdb.WriteMSettings(fMultiSendStake, fMultiSendDynodeReward, nLastMultiSendHeight))
            LogPrintf("Failed to write MultiSend setting to DB\n");

        LogPrintf("MultiSend successfully sent\n");

        //set which MultiSend triggered
        if (sendMSOnStake)
            stakeSent = true;
        else
            mnSent = true;

        //stop iterating if we have sent out all the MultiSend(s)
        if ((stakeSent && mnSent) || (stakeSent && !fMultiSendDynodeReward) || (mnSent && !fMultiSendStake))
            return true;
    }

    return true;
}

void CWallet::UpdateMyRestrictedAssets(std::string& address, std::string& asset_name, int type, uint32_t date)
{
    LOCK(cs_wallet);
    NotifyMyRestrictedAssetsChanged(this, address, asset_name, type, date);
}

CKeyPool::CKeyPool()
{
    nTime = GetTime();
    fInternal = false;
}

CKeyPool::CKeyPool(const CPubKey& vchPubKeyIn, bool fInternalIn)
{
    nTime = GetTime();
    vchPubKey = vchPubKeyIn;
    fInternal = fInternalIn;
}

CEdKeyPool::CEdKeyPool()
{
    nTime = GetTime();
    fInternal = false;
}

CEdKeyPool::CEdKeyPool(const std::vector<unsigned char>& edPubKeyIn, bool fInternalIn)
{
    nTime = GetTime();
    edPubKey = edPubKeyIn;
    fInternal = fInternalIn;
}

CWalletKey::CWalletKey(int64_t nExpires)
{
    nTimeCreated = (nExpires ? GetTime() : 0);
    nTimeExpires = nExpires;
}

void CMerkleTx::SetMerkleBranch(const CBlockIndex* pindex, int posInBlock)
{
    // Update the tx's hashBlock
    hashBlock = pindex->GetBlockHash();

    // set the position of the transaction in the block
    nIndex = posInBlock;
}

int CMerkleTx::GetDepthInMainChain(const CBlockIndex*& pindexRet) const
{
    int nResult;

    if (hashUnset())
        nResult = 0;
    else {
        AssertLockHeld(cs_main);

        // Find the block it claims to be in
        BlockMap::iterator mi = mapBlockIndex.find(hashBlock);
        if (mi == mapBlockIndex.end())
            nResult = 0;
        else {
            CBlockIndex* pindex = (*mi).second;
            if (!pindex || !chainActive.Contains(pindex))
                nResult = 0;
            else {
                pindexRet = pindex;
                nResult = ((nIndex == -1) ? (-1) : 1) * (chainActive.Height() - pindex->nHeight + 1);

                if (nResult == 0 && !mempool.exists(GetHash()))
                    return -1; // Not in chain, not in mempool
            }
        }
    }

    return nResult;
}

bool CMerkleTx::IsLockedByInstantSend() const
{
    return instantsend.IsLockedInstantSendTransaction(GetHash());
}

int CMerkleTx::GetBlocksToMaturity() const
{
    if (!(IsCoinBase() || IsCoinStake()))
        return 0;
    return std::max(0, (COINBASE_MATURITY + 1) - GetDepthInMainChain());
}


bool CMerkleTx::AcceptToMemoryPool(const CAmount& nAbsurdFee, CValidationState& state)
{
    return ::AcceptToMemoryPool(mempool, state, tx, true, nullptr, nullptr, false, nAbsurdFee);
}
