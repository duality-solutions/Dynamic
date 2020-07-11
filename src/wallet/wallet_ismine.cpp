// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "wallet_ismine.h"

#include "bdap/stealth.h"
#include "key.h"
#include "keystore.h"
#include "script/script.h"
#include "script/sign.h"
#include "script/standard.h"
#include "util.h"
#include "validation.h"

typedef std::vector<unsigned char> valtype;

unsigned int HaveKeys(const std::vector<valtype>& pubkeys, const CKeyStore& keystore)
{
    unsigned int nResult = 0;
    for (const valtype& pubkey : pubkeys) {
        CKeyID keyID = CPubKey(pubkey).GetID();
        if (keystore.HaveKey(keyID))
            ++nResult;
    }
    return nResult;
}

isminetype IsMine(const CKeyStore& keystore, const CScript& scriptPubKey, SigVersion sigversion)
{
    bool isInvalid = false;
    return IsMine(keystore, scriptPubKey, isInvalid, sigversion);
}

isminetype IsMine(const CKeyStore& keystore, const CTxDestination& dest, SigVersion sigversion)
{
    bool isInvalid = false;
    return IsMine(keystore, dest, isInvalid, sigversion);
}

isminetype IsMine(const CKeyStore &keystore, const CTxDestination& dest, bool& isInvalid, SigVersion sigversion)
{
    CScript script = GetScriptForDestination(dest);
    return IsMine(keystore, script, isInvalid, sigversion);
}

isminetype IsMine(const CKeyStore &keystore, const CScript& scriptPubKey, bool& isInvalid, SigVersion sigversion)
{
    isInvalid = false;

    std::vector<valtype> vSolutions;
    txnouttype whichType;
    if (!Solver(scriptPubKey, whichType, vSolutions)) {
        if (keystore.HaveWatchOnly(scriptPubKey))
            return ISMINE_WATCH_UNSOLVABLE;
        return ISMINE_NO;
    }

    CKeyID keyID;
    switch (whichType) {
        case TX_NONSTANDARD:
        case TX_NULL_DATA:
            break;
        case TX_RESTRICTED_ASSET_DATA:
            break;
        case TX_PUBKEY:
            keyID = CPubKey(vSolutions[0]).GetID();
            if (sigversion != SIGVERSION_BASE && vSolutions[0].size() != 33) {
                isInvalid = true;
                return ISMINE_NO;
            }
            if (keystore.HaveKey(keyID))
                return ISMINE_SPENDABLE;
            break;
        case TX_PUBKEYHASH:
            keyID = CKeyID(uint160(vSolutions[0]));
            if (sigversion != SIGVERSION_BASE) {
                CPubKey pubkey;
                if (keystore.GetPubKey(keyID, pubkey) && !pubkey.IsCompressed()) {
                    isInvalid = true;
                    return ISMINE_NO;
                }
            }
            if (keystore.HaveKey(keyID))
                return ISMINE_SPENDABLE;
            break;
        case TX_SCRIPTHASH: {
            CScriptID scriptID = CScriptID(uint160(vSolutions[0]));
            CScript subscript;
            if (keystore.GetCScript(scriptID, subscript)) {
                isminetype ret = IsMine(keystore, subscript, isInvalid);
                if (ret == ISMINE_SPENDABLE || ret == ISMINE_WATCH_SOLVABLE || (ret == ISMINE_NO && isInvalid))
                    return ret;
            }
            break;
        }
        case TX_MULTISIG: {
            // Only consider transactions "mine" if we own ALL the
            // keys involved. Multi-signature transactions that are
            // partially owned (somebody else has a key that can spend
            // them) enable spend-out-from-under-you attacks, especially
            // in shared-wallet situations.
            std::vector<valtype> keys(vSolutions.begin() + 1, vSolutions.begin() + vSolutions.size() - 1);
            if (sigversion != SIGVERSION_BASE) {
                for (size_t i = 0; i < keys.size(); i++) {
                    if (keys[i].size() != 33) {
                        isInvalid = true;
                        return ISMINE_NO;
                    }
                }
            }
            if (HaveKeys(keys, keystore) == keys.size())
                return ISMINE_SPENDABLE;
            break;
        }
    /** ASSET START */
        case TX_NEW_ASSET: {
            if (!AreAssetsDeployed())
                return ISMINE_NO;
            keyID = CKeyID(uint160(vSolutions[0]));
            if (sigversion != SIGVERSION_BASE) {
                CPubKey pubkey;
                if (keystore.GetPubKey(keyID, pubkey) && !pubkey.IsCompressed()) {
                    isInvalid = true;
                    return ISMINE_NO;
                }
            }
            if (keystore.HaveKey(keyID))
                return ISMINE_SPENDABLE;
            break;

        }

        case TX_TRANSFER_ASSET: {
            if (!AreAssetsDeployed())
                return ISMINE_NO;
            keyID = CKeyID(uint160(vSolutions[0]));
            if (sigversion != SIGVERSION_BASE) {
                CPubKey pubkey;
                if (keystore.GetPubKey(keyID, pubkey) && !pubkey.IsCompressed()) {
                    isInvalid = true;
                    return ISMINE_NO;
                }
            }
            if (keystore.HaveKey(keyID))
                return ISMINE_SPENDABLE;
            break;
        }

        case TX_REISSUE_ASSET: {
            if (!AreAssetsDeployed())
                return ISMINE_NO;
            keyID = CKeyID(uint160(vSolutions[0]));
            if (sigversion != SIGVERSION_BASE) {
                CPubKey pubkey;
                if (keystore.GetPubKey(keyID, pubkey) && !pubkey.IsCompressed()) {
                    isInvalid = true;
                    return ISMINE_NO;
                }
            }
            if (keystore.HaveKey(keyID))
                return ISMINE_SPENDABLE;
            break;
        }
    /** ASSET END*/
    }

    if (keystore.HaveWatchOnly(scriptPubKey)) {
        // TODO: This could be optimized some by doing some work after the above solver
        SignatureData sigs;
        return ProduceSignature(DummySignatureCreator(&keystore), scriptPubKey, sigs) ? ISMINE_WATCH_SOLVABLE : ISMINE_WATCH_UNSOLVABLE;
    }
    return ISMINE_NO;
}
