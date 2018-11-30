// Copyright (c) 2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "operations.h"
#include "validation.h"

#include "key_io.h"
#include "wallet/wallet.h"
#include "wallet/walletdb.h"

#include <boost/algorithm/string.hpp>

#ifdef ENABLE_WALLET
extern CWallet* pwalletMain;
#endif //ENABLE_WALLET

/////////////////////////////////////////////////////////////
//
// String Manipulations and Operations
//
/////////////////////////////////////////////////////////////

std::string SignatureDelimiter = " ";
std::string PrimaryDelimiter = "@";
std::string SubDelimiter = "$";

void ScrubString(std::string& input, bool forInteger)
{
    input.erase(std::remove(input.begin(), input.end(), '@'), input.end());
    input.erase(std::remove(input.begin(), input.end(), '$'), input.end());
    if (forInteger)
        input.erase(std::remove(input.begin(), input.end(), ' '), input.end());
}

void SeparateString(std::string input, std::vector<std::string>& output, bool subDelimiter)
{
    if (subDelimiter)
        boost::split(output, input, boost::is_any_of(SubDelimiter));
    else
        boost::split(output, input, boost::is_any_of(PrimaryDelimiter));
}

void SeparateFluidOpString(std::string input, std::vector<std::string>& output)
{
    std::vector<std::string> firstSplit;
    SeparateString(input, firstSplit);
    if (firstSplit.size() > 1) {
        std::vector<std::string> secondSplit;
        SeparateString(firstSplit[0], secondSplit, true);
        for (const std::string& item : secondSplit) {
            output.push_back(item);
        }
        unsigned int n = 0;
        for (const std::string& item : firstSplit) {
            if (n != 0) {
                output.push_back(item);
            }
            n = +1;
        }
    }
}

std::string StitchString(std::string stringOne, std::string stringTwo, bool subDelimiter)
{
    if (subDelimiter)
        return stringOne + SubDelimiter + stringTwo;
    else
        return stringOne + PrimaryDelimiter + stringTwo;
}

std::string StitchString(std::string stringOne, std::string stringTwo, std::string stringThree, bool subDelimiter)
{
    if (subDelimiter)
        return stringOne + SubDelimiter + stringTwo + SubDelimiter + stringThree;
    else
        return stringOne + PrimaryDelimiter + stringTwo + PrimaryDelimiter + stringThree;
}

std::string GetRidOfScriptStatement(std::string input, int position)
{
    std::vector<std::string> output;
    boost::split(output, input, boost::is_any_of(" "));

    return output.at(position);
}

/////////////////////////////////////////////////////////////
//
// Cryptographic and Misc Operations
//
/////////////////////////////////////////////////////////////

bool COperations::VerifyAddressOwnership(CDynamicAddress dynamicAddress)
{
#ifdef ENABLE_WALLET
    LOCK2(cs_main, pwalletMain ? &pwalletMain->cs_wallet : NULL);
    CDynamicAddress address(dynamicAddress);

    if (address.IsValid()) {
        CTxDestination dest = address.Get();
        CScript scriptPubKey = GetScriptForDestination(dest);
        isminetype mine = pwalletMain ? IsMine(*pwalletMain, dest) : ISMINE_NO;

        return ((mine & ISMINE_SPENDABLE) ? true : false);
    }

    return false;
#else
    // Wallet cannot be accessed, cannot continue ahead!
    return false;
#endif //ENABLE_WALLET
}


bool COperations::SignTokenMessage(CDynamicAddress address, std::string unsignedMessage, std::string& stitchedMessage, bool stitch)
{
#ifdef ENABLE_WALLET
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << unsignedMessage;

    CDynamicAddress addr(address);

    CKeyID keyID;
    if (!addr.GetKeyID(keyID))
        return false;

    CKey key;
    if (!pwalletMain->GetKey(keyID, key))
        return false;

    std::vector<unsigned char> vchSig;
    if (!key.SignCompact(ss.GetHash(), vchSig))
        return false;
    else if (stitch)
        stitchedMessage = StitchString(unsignedMessage, EncodeBase64(&vchSig[0], vchSig.size()), false);
    else
        stitchedMessage = EncodeBase64(&vchSig[0], vchSig.size());

    return true;
#else
    return false;
#endif //ENABLE_WALLET
}

bool COperations::GenericSignMessage(const std::string message, std::string& signedString, CDynamicAddress signer)
{
    if (!SignTokenMessage(signer, message, signedString, true))
        return false;
    else
        ConvertToHex(signedString);

    return true;
}
