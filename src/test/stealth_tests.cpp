// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "key.h"

#include "base58.h"
#include "script/script.h"
#include "uint256.h"
#include "util.h"
#include "utilstrencodings.h"
#include "test/test_dynamic.h"
#include "bdap/stealth.h"
#include "wallet/wallet.h"

#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <boost/test/unit_test.hpp>

//stealth_tests

BOOST_FIXTURE_TEST_SUITE(stealth_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(stealth_test1)
{
    {
    CWallet* pwalletMain2;
    bool fFirstRun;
    pwalletMain2 = new CWallet("wallet_test.dat");
    pwalletMain2->LoadWallet(fFirstRun);

    // Sender only knows the public keys embedded in the string below.
    std::string stealth_address_str = "L3D2cdHCVd2d4wS6rP1NRZWk6TpRxEqTwFxTWN9iNcuGNi9BMMLBwWh6hbRG9GGPqAqYiscYmFUXTvvE3YsgQ6iicXdbJ6apUb41Vz";
    
    CTxDestination dest = DecodeDestination(stealth_address_str);

    BOOST_CHECK(IsValidDestination(dest));

    CStealthAddress sxAddr_sender = boost::get<CStealthAddress>(dest); //deriving a stealth address, just public key portion [sender]
    CScript scriptDest;
    std::vector<uint8_t> vStealthData;
    
    std::string sError;
    BOOST_CHECK(PrepareStealthOutput(sxAddr_sender, scriptDest, vStealthData, sError) == 0);

    CScript stealthScript;
    stealthScript << OP_RETURN << vStealthData;

    CTxDestination newDest;
    BOOST_CHECK(ExtractDestination(scriptDest, newDest));
    CDynamicAddress newAddress;
    newAddress.Set(newDest);
    BOOST_CHECK(newAddress.IsValid());
    //std::cout << "Derived " << newAddress.ToString() <<  " address from stealth\n";

    // Receiver has the new address the sender derived from their public stealth address along with raw public key
    std::string scan_private_key_str = "MktpUfR5RiRrFCkLWYxUEQ9portxgHADvpvByv8dD5aLNH4ZjVM5";
    std::string spend_private_key_str = "MmVtgDTFTUMe6b5mk166QFPFLpem1J1HtS8EhSMZNo5aGsZxZS96";
    
    CDynamicSecret scanSecret, spendSecret;

    scanSecret.SetString(scan_private_key_str);
    spendSecret.SetString(spend_private_key_str);

    BOOST_CHECK(scanSecret.IsValid());
    BOOST_CHECK(spendSecret.IsValid());

    CKey scanKey = scanSecret.GetKey();
    CKey spendKey = spendSecret.GetKey();

    CStealthAddress sxAddr_receiver(scanKey, spendKey); //destination (receiver, full stealth address)

    pwalletMain2->AddStealthAddress(sxAddr_receiver); //add stealthaddress to wallet
    pwalletMain2->AddKeyPubKey(scanKey, scanKey.GetPubKey()); //add to wallet
    pwalletMain2->AddKeyPubKey(spendKey, spendKey.GetPubKey()); //add to wallet

    BOOST_CHECK(IsDataScript(stealthScript)); 

    std::vector<uint8_t> vData;
    CKey sShared;
    std::vector<uint8_t> pkExtracted;

    CTxDestination address;

    BOOST_CHECK(GetDataFromScript(stealthScript, vData));

    std::vector<uint8_t> vchEphemPK;
    if (vData[0] == DO_STEALTH) {
        if (vData.size() < 34 ) {
            return; // error
        }
        if (vData.size() > 40) {
            return; // Stealth OP_RETURN data is always less then 40 bytes
        }

        vchEphemPK.resize(33);
        memcpy(&vchEphemPK[0], &vData[1], 33);

        uint32_t prefix = 0;
        bool fHavePrefix = false;
        if (vData.size() >= 34 + 5 && vData[34] == DO_STEALTH_PREFIX) {
            fHavePrefix = true;
            memcpy(&prefix, &vData[35], 4);
        }

        BOOST_CHECK(vData == vStealthData);

        BOOST_CHECK(ExtractDestination(scriptDest, address));

        BOOST_CHECK(address.type() == typeid(CKeyID)); 

        BOOST_CHECK(pwalletMain2->ProcessStealthOutput(address, vchEphemPK, prefix, fHavePrefix, sShared));

    }

    CKey derivedKey; //shared output key

    BOOST_CHECK(StealthSecret(sxAddr_receiver.scan_secret, vchEphemPK, sxAddr_receiver.spend_pubkey, derivedKey, pkExtracted) == 0);
    
    CPubKey pkE(pkExtracted);

    BOOST_CHECK(pkE.IsValid());

    CKey sSpendR;
    CKey sSpendRCompare;

    BOOST_CHECK(StealthSharedToSecretSpend(sShared, spendKey, sSpendR) == 0); //checks sender

    BOOST_CHECK(StealthSharedToSecretSpend(derivedKey, spendKey, sSpendRCompare) == 0); //checks receiver

    BOOST_CHECK(sSpendR == sSpendRCompare);

    CDynamicAddress addressCompare(sSpendR.GetPubKey().GetID());

    BOOST_CHECK(newAddress == addressCompare);

    //cleanup
    delete pwalletMain2;
    pwalletMain2 = NULL;
    std::cout << "Exit: stealth_test1\n";
    }
} //stealth_test1

BOOST_AUTO_TEST_CASE(stealth_test1_negative)
{
    {
    CWallet* pwalletMain2;
    bool fFirstRun;
    pwalletMain2 = new CWallet("wallet_test.dat");
    pwalletMain2->LoadWallet(fFirstRun);

    // Sender only knows the public keys embedded in the string below.
    std::string stealth_address_str = "L3D2cdHCVd2d4wS6rP1NRZWk6TpRxEqTwFxTWN9iNcuGNi9BMMLBwWh6hbRG9GGPqAqYiscYmFUXTvvE3YsgQ6iicXdbJ6apUb41Vz";
    
    CTxDestination dest = DecodeDestination(stealth_address_str);

    BOOST_CHECK(IsValidDestination(dest));

    CStealthAddress sxAddr_sender = boost::get<CStealthAddress>(dest); //deriving a stealth address, just public key portion [sender]
    CScript scriptDest;
    std::vector<uint8_t> vStealthData;
    
    std::string sError;
    BOOST_CHECK(PrepareStealthOutput(sxAddr_sender, scriptDest, vStealthData, sError) == 0);

    CScript stealthScript;
    stealthScript << OP_RETURN << vStealthData;

    CTxDestination newDest;
    BOOST_CHECK(ExtractDestination(scriptDest, newDest));
    CDynamicAddress newAddress;
    newAddress.Set(newDest);
    BOOST_CHECK(newAddress.IsValid());
    //std::cout << "Derived " << newAddress.ToString() <<  " address from stealth\n";

    //Receiver has random scanKey and spendKey to generate stealthkey
    CKey walletKey;
    CKey scanKey, spendKey;

    walletKey.MakeNewKey(true);
    walletKey.DeriveChildKey(spendKey);
    spendKey.DeriveChildKey(scanKey);

    CStealthAddress sxAddr_receiver(scanKey, spendKey); //destination (receiver, full stealth address)

    pwalletMain2->AddStealthAddress(sxAddr_receiver); //add stealthaddress to wallet
    pwalletMain2->AddKeyPubKey(scanKey, scanKey.GetPubKey()); //add to wallet
    pwalletMain2->AddKeyPubKey(spendKey, spendKey.GetPubKey()); //add to wallet

    BOOST_CHECK(IsDataScript(stealthScript)); 

    std::vector<uint8_t> vData;
    CKey sShared;
    std::vector<uint8_t> pkExtracted;

    CTxDestination address;

    BOOST_CHECK(GetDataFromScript(stealthScript, vData));

    std::vector<uint8_t> vchEphemPK;
    if (vData[0] == DO_STEALTH) {
        if (vData.size() < 34 ) {
            return; // error
        }
        if (vData.size() > 40) {
            return; // Stealth OP_RETURN data is always less then 40 bytes
        }

        vchEphemPK.resize(33);
        memcpy(&vchEphemPK[0], &vData[1], 33);

        uint32_t prefix = 0;
        bool fHavePrefix = false;
        if (vData.size() >= 34 + 5 && vData[34] == DO_STEALTH_PREFIX) {
            fHavePrefix = true;
            memcpy(&prefix, &vData[35], 4);
        }

        //added this check
        BOOST_CHECK(vData == vStealthData);

        BOOST_CHECK(ExtractDestination(scriptDest, address));

        BOOST_CHECK(address.type() == typeid(CKeyID)); 

        BOOST_CHECK(pwalletMain2->ProcessStealthOutput(address, vchEphemPK, prefix, fHavePrefix, sShared));

    }

    CKey derivedKey; //shared output key

    BOOST_CHECK(StealthSecret(sxAddr_receiver.scan_secret, vchEphemPK, sxAddr_receiver.spend_pubkey, derivedKey, pkExtracted) == 0);

    CKey sSpendR;
    CKey sSpendRCompare;

    BOOST_CHECK(StealthSharedToSecretSpend(sShared, spendKey, sSpendR) == 0);  //from sender side

    BOOST_CHECK(StealthSharedToSecretSpend(derivedKey, spendKey, sSpendRCompare) == 0); //from receiver side

    //these should not match
    BOOST_CHECK(!(sSpendR == sSpendRCompare));

    //cleanup
    delete pwalletMain2;
    pwalletMain2 = NULL;

    std::cout << "Exit: stealth_test1_negative\n";
    }
} //stealth_test1_negative



BOOST_AUTO_TEST_CASE(stealth_test2)
{
    {
    CWallet* pwalletMain2;
    CKey walletKey;
    CKey scanKey, spendKey;

    walletKey.MakeNewKey(true);
    walletKey.DeriveChildKey(spendKey);
    spendKey.DeriveChildKey(scanKey);

    CStealthAddress sxAddr(scanKey, spendKey); //destination

    bool fFirstRun;
    pwalletMain2 = new CWallet("wallet_test.dat");
    pwalletMain2->LoadWallet(fFirstRun);

    pwalletMain2->AddStealthAddress(sxAddr); //add stealthaddress to wallet
    pwalletMain2->AddKeyPubKey(scanKey, scanKey.GetPubKey()); //add to wallet
    pwalletMain2->AddKeyPubKey(spendKey, spendKey.GetPubKey()); //add to wallet

    CScript scriptDest;
    std::vector<uint8_t> vStealthData;
    std::vector<uint8_t> vchEphemPK;
    std::string sError;
 
    BOOST_CHECK(PrepareStealthOutput(sxAddr, scriptDest, vStealthData, sError) == 0);

    CScript stealthScript;
    stealthScript << OP_RETURN << vStealthData;

    BOOST_CHECK(IsDataScript(stealthScript)); 

    std::vector<uint8_t> vData;
    CKey sShared; //doesn't need initialization
    std::vector<uint8_t> pkExtracted;

    CTxDestination address;

    BOOST_CHECK(GetDataFromScript(stealthScript, vData)); 

    if (vData[0] == DO_STEALTH) {
        if (vData.size() < 34 ) {
            return; // error
        }
        if (vData.size() > 40) {
            return; // Stealth OP_RETURN data is always less then 40 bytes
        }
        vchEphemPK.resize(33);
        memcpy(&vchEphemPK[0], &vData[1], 33);

        uint32_t prefix = 0;
        bool fHavePrefix = false;
        if (vData.size() >= 34 + 5 && vData[34] == DO_STEALTH_PREFIX) {
            fHavePrefix = true;
            memcpy(&prefix, &vData[35], 4);
        }

        BOOST_CHECK(vData == vStealthData);

        BOOST_CHECK(ExtractDestination(scriptDest, address));

        BOOST_CHECK(address.type() == typeid(CKeyID)); 

        BOOST_CHECK(pwalletMain2->ProcessStealthOutput(address, vchEphemPK, prefix, fHavePrefix, sShared));
    }

    CKey derivedKey; //shared output key

    BOOST_CHECK(StealthSecret(sxAddr.scan_secret, vchEphemPK, sxAddr.spend_pubkey, derivedKey, pkExtracted) == 0);
    
    CPubKey pkE(pkExtracted);

    BOOST_CHECK(pkE.IsValid());

    CKey sSpendR;
    CKey sSpendRCompare;

    BOOST_CHECK(StealthSharedToSecretSpend(sShared, spendKey, sSpendR) == 0);

    BOOST_CHECK(StealthSharedToSecretSpend(derivedKey, spendKey, sSpendRCompare) == 0);

    BOOST_CHECK(sSpendR == sSpendRCompare);

    CDynamicAddress addressCompare(sSpendR.GetPubKey().GetID());

    BOOST_CHECK(address == addressCompare.Get());

    //cleanup
    delete pwalletMain2;
    pwalletMain2 = NULL;

    std::cout << "Exit: stealth_test2\n";
    }

} //stealth_test2

BOOST_AUTO_TEST_SUITE_END()