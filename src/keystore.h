// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_KEYSTORE_H
#define DYNAMIC_KEYSTORE_H

#include "dht/ed25519.h"
#include "hdchain.h"
#include "key.h"
#include "key/extkey.h"
#include "key/stealth.h"
#include "pubkey.h"
#include "script/script.h"
#include "script/standard.h"
#include "sync.h"

#include <boost/signals2/signal.hpp>
#include <boost/variant.hpp>

/** A virtual base class for key stores */
class CKeyStore
{
protected:
    mutable CCriticalSection cs_KeyStore;

public:
    virtual ~CKeyStore() {}

    //! Add a key to the store.
    virtual bool AddKeyPubKey(const CKey& key, const CPubKey& pubkey) = 0;
    virtual bool AddKey(const CKey& key);
    //! Check whether a key corresponding to a given address is present in the store.
    virtual isminetype IsMine(const CKeyID &address) const =0;
    //! Check whether a key corresponding to a given address is present in the store.
    virtual bool HaveKey(const CKeyID &address) const =0;
    virtual bool HaveDHTKey(const CKeyID &address) const =0;
    virtual bool GetKey(const CKeyID &address, CKey& keyOut) const =0;
    virtual void GetKeys(std::set<CKeyID> &setAddress) const =0;
    virtual bool GetPubKey(const CKeyID& address, CPubKey& vchPubKeyOut) const =0;
    virtual bool GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const =0;
    virtual bool GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const =0;
    virtual bool AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& vchPubKey) =0;
    
    //! Support for BIP 0013 : see https://github.com/bitcoin/bips/blob/master/bip-0013.mediawiki
    virtual bool AddCScript(const CScript& redeemScript) = 0;
    virtual bool HaveCScript(const CScriptID& hash) const = 0;
    virtual bool GetCScript(const CScriptID& hash, CScript& redeemScriptOut) const = 0;

    //! Support for Watch-only addresses
    virtual bool AddWatchOnly(const CScript& dest) = 0;
    virtual bool RemoveWatchOnly(const CScript& dest) = 0;
    virtual bool HaveWatchOnly(const CScript& dest) const = 0;
    virtual bool HaveWatchOnly() const = 0;
};

typedef std::map<CKeyID, CKey> KeyMap;
typedef std::map<CKeyID, CPubKey> WatchKeyMap;
typedef std::map<CScriptID, CScript> ScriptMap;
typedef std::set<CScript> WatchOnlySet;
typedef std::map<CKeyID, CKeyEd25519> DHTKeyMap;

/** Basic key store, that keeps keys in an address->secret map */
class CBasicKeyStore : public CKeyStore
{
protected:
    KeyMap mapKeys;
    WatchKeyMap mapWatchKeys;
    ScriptMap mapScripts;
    WatchOnlySet setWatchOnly;
    DHTKeyMap mapDHTKeys;
    /* the HD chain data model*/
    CHDChain hdChain;

public:
    bool AddKeyPubKey(const CKey& key, const CPubKey &pubkey) override;
    bool GetPubKey(const CKeyID& address, CPubKey& vchPubKeyOut) const override;
    bool HaveDHTKey(const CKeyID &address) const override
    {
        bool result;
        {
            LOCK(cs_KeyStore);
            result = (mapDHTKeys.count(address) > 0);
        }
        return result;
    }
    bool HaveKey(const CKeyID &address) const override
    {
        bool result;
        {
            LOCK(cs_KeyStore);
            result = (mapKeys.count(address) > 0);
        }
        return result;
    }
    void GetKeys(std::set<CKeyID>& setAddress) const override
    {
        setAddress.clear();
        {
            LOCK(cs_KeyStore);
            KeyMap::const_iterator mi = mapKeys.begin();
            while (mi != mapKeys.end()) {
                setAddress.insert((*mi).first);
                mi++;
            }
        }
    }
    bool GetKey(const CKeyID& address, CKey& keyOut) const override
    {
        {
            LOCK(cs_KeyStore);
            KeyMap::const_iterator mi = mapKeys.find(address);
            if (mi != mapKeys.end()) {
                keyOut = mi->second;
                return true;
            }
        }
        return false;
    }
    virtual bool AddCScript(const CScript& redeemScript) override;
    virtual bool HaveCScript(const CScriptID& hash) const override;
    virtual bool GetCScript(const CScriptID& hash, CScript& redeemScriptOut) const override;

    virtual bool AddWatchOnly(const CScript& dest) override;
    virtual bool RemoveWatchOnly(const CScript& dest) override;
    virtual bool HaveWatchOnly(const CScript& dest) const override;
    virtual bool HaveWatchOnly() const override;

    virtual bool GetHDChain(CHDChain& hdChainRet) const;

    bool GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const override;
    bool AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& vchPubKey) override;
    bool GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const override;

    //! Check whether a key corresponding to a given address is present in the store.
    isminetype IsMine(const CKeyID &address) const;

};

typedef std::vector<unsigned char, secure_allocator<unsigned char> > CKeyingMaterial;
typedef std::map<CKeyID, std::pair<CPubKey, std::vector<unsigned char> > > CryptedKeyMap;
typedef std::map<CKeyID, std::pair<std::vector<unsigned char>, std::vector<unsigned char> > > CryptedDHTKeyMap;

#endif // DYNAMIC_KEYSTORE_H
