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
#include "bdap/linking.h"
#include "bdap/utils.h"
#include "dht/ed25519.h"
#include "dht/datarecord.h"
#include "dht/dataheader.h"
#include "dht/datachunk.h"

#include "bdap/bdap.h"
#include "bdap/vgpmessage.h"
#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/linkmanager.h"
#include "timedata.h"
#include "wallet/wallet.h"
#include "net.h"

#include "chain.h"
#include "chainparams.h"
#include "rpcserver.h"
#include "rpcclient.h"


#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <univalue.h>

#include <boost/test/unit_test.hpp>

//bdap_vgp_message_tests

BOOST_FIXTURE_TEST_SUITE(bdap_vgp_message_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(bdap_vgp_message_test1)
{

    //std::cout << Params().NetworkIDString() << "\n";

    CharString vchObjectID = vchFromValue("test1");
    CharString vchCommonName = vchFromValue("test user 1");

    //retrieve vchDefault* values from bdap.h

    CharString vchRequestorFQDN = vchFromValue("test1@public.bdap.io");
    CharString vchRecipientFQDN = vchFromValue("test2@public.bdap.io");
    std::string strLinkMessage = "link test from test1 to test2";

    //Domain Entry 1 (Sender)
    CDomainEntry senderDomainEntry;
    senderDomainEntry.RootOID = vchDefaultOIDPrefix;
    senderDomainEntry.DomainComponent = vchDefaultDomainName;
    senderDomainEntry.OrganizationalUnit = vchDefaultPublicOU;
    senderDomainEntry.CommonName = vchCommonName;
    senderDomainEntry.OrganizationName = vchDefaultOrganizationName;
    senderDomainEntry.ObjectID = vchObjectID;
    senderDomainEntry.fPublicObject = 1; // make entry public
    senderDomainEntry.nObjectType = 1; //BDAP_USER

    CharString vchObjectID2 = vchFromValue("test2");
    CharString vchCommonName2 = vchFromValue("test user 2");


    //Domain Entry 2 (Receiver)
    CDomainEntry receiverDomainEntry;
    receiverDomainEntry.RootOID = vchDefaultOIDPrefix;
    receiverDomainEntry.DomainComponent = vchDefaultDomainName;
    receiverDomainEntry.OrganizationalUnit = vchDefaultPublicOU;
    receiverDomainEntry.CommonName = vchCommonName2;
    receiverDomainEntry.OrganizationName = vchDefaultOrganizationName;
    receiverDomainEntry.ObjectID = vchObjectID2;
    receiverDomainEntry.fPublicObject = 1; // make entry public
    receiverDomainEntry.nObjectType = 1; //BDAP_USER


    //WalletKey for Sender
    CKey privWalletKey;
    privWalletKey.MakeNewKey(true);
    CPubKey pubWalletKey = privWalletKey.GetPubKey();
    CKeyID keyWalletID = pubWalletKey.GetID();
    CDynamicAddress walletAddress = CDynamicAddress(keyWalletID);
    CharString vchWalletAddress = vchFromString(walletAddress.ToString());
    senderDomainEntry.WalletAddress = vchWalletAddress;

    //WalletKey for Receiver
    CKey recprivWalletKey;
    recprivWalletKey.MakeNewKey(true);
    CPubKey recpubWalletKey = recprivWalletKey.GetPubKey();
    CKeyID reckeyWalletID = recpubWalletKey.GetID();
    CDynamicAddress recwalletAddress = CDynamicAddress(reckeyWalletID);
    CharString recvchWalletAddress = vchFromString(recwalletAddress.ToString());
    receiverDomainEntry.WalletAddress = recvchWalletAddress;


    //Private and Public Keys for Sender
    CKeyEd25519 privReqDHTKey;
    CharString vchDHTPubKey = privReqDHTKey.GetPubKey();
    senderDomainEntry.DHTPublicKey = vchDHTPubKey;

    //Private and Public Keys for Receiver
    CKeyEd25519 privReceiverDHTKey;
    CharString vchDHTReceiverPubKey = privReceiverDHTKey.GetPubKey();
    receiverDomainEntry.DHTPublicKey = vchDHTReceiverPubKey;

    //Private and Public Keys for Link Request
    CKeyEd25519 linkrequestKey;
    CharString linkrequestpubkey = linkrequestKey.GetPubKey();

    //Private and Public Keys for Link Accept
    CKeyEd25519 linkacceptKey;
    CharString linkacceptpubkey = linkacceptKey.GetPubKey();

    CLinkRequest linkRequest; 
    CLinkAccept linkAccept; 

    //Key exchange - Shared pub keys for linkrequest and link accept
    std::vector<unsigned char> linkRequestSharedPubKey = GetLinkSharedPubKey(linkrequestKey,receiverDomainEntry.DHTPublicKey);
    std::vector<unsigned char> linkAcceptSharedPubKey = GetLinkSharedPubKey(linkacceptKey, senderDomainEntry.DHTPublicKey);

    //LINK REQUEST
    //example: rpclinking.cpp line 153
    linkRequest.nVersion = 1; // version 1 = encrytped or a private link.
    linkRequest.RequestorFullObjectPath = vchRequestorFQDN;
    linkRequest.RecipientFullObjectPath = vchRecipientFQDN;
    linkRequest.LinkMessage = vchFromString(strLinkMessage);
    linkRequest.RequestorPubKey = linkrequestpubkey; //from above?
    linkRequest.SharedPubKey = linkRequestSharedPubKey;

    //LINK ACCEPT
    linkAccept.nVersion = 1; // version 1 = encrytped or a private link.
    linkAccept.RequestorFullObjectPath = vchRequestorFQDN;
    linkAccept.RecipientFullObjectPath = vchRecipientFQDN;
    linkAccept.RecipientPubKey = linkacceptpubkey;
    linkAccept.SharedPubKey = linkAcceptSharedPubKey;

    //CreateSignatureProof
    BOOST_CHECK(CreateSignatureProof(privWalletKey, linkRequest.RequestorFQDN(), linkRequest.SignatureProof));
    BOOST_CHECK(CreateSignatureProof(recprivWalletKey, linkAccept.RecipientFQDN(), linkAccept.SignatureProof));

    //SignatureProofIsValid
    BOOST_CHECK(SignatureProofIsValid(senderDomainEntry.GetWalletAddress(), linkRequest.RequestorFQDN(), linkRequest.SignatureProof));
    BOOST_CHECK(SignatureProofIsValid(receiverDomainEntry.GetWalletAddress(), linkAccept.RecipientFQDN(), linkAccept.SignatureProof));

    //ValidateValues
    BOOST_CHECK(linkRequest.ValidateValues(strLinkMessage));
    BOOST_CHECK(linkAccept.ValidateValues(strLinkMessage));

    //LINK ACCEPT AND CLINK
    uint256 linkID = GetLinkID(linkAccept);

    //MIMIC GetSecretSharedKey logic 
    CKeyEd25519 sharedKey1(linkRequestSharedPubKey);
    CKeyEd25519 sharedKey2(linkAcceptSharedPubKey);
    CKeyEd25519 sharedKey3;

    //Key Exchange, seed
    std::array<char, 32> seed1 = GetLinkSharedPrivateKey(sharedKey1, sharedKey2.GetPubKey());

    //Key Exchange, get reciprocal seed
    std::array<char, 32> seed2 = GetLinkSharedPrivateKey(sharedKey2, sharedKey1.GetPubKey());

    //these derived keys should be the same
    BOOST_CHECK(seed1 == seed2);

    //this derived key should be invalid
    std::array<char, 32> invalidseed = GetLinkSharedPrivateKey(sharedKey3, sharedKey2.GetPubKey()); // should be invalid key

    //CLINK
    CLink record;
    record.LinkID = linkID;
    record.fAcceptFromMe = true; //fIsLinkFromMe;
    record.nLinkState = 2;
    record.RequestorFullObjectPath = linkAccept.RequestorFullObjectPath;
    record.RecipientFullObjectPath = linkAccept.RecipientFullObjectPath;
    record.RecipientPubKey = linkAccept.RecipientPubKey;
    record.SharedAcceptPubKey = linkAccept.SharedPubKey;
    record.nHeightAccept = linkAccept.nHeight;
    record.nExpireTimeAccept = linkAccept.nExpireTime;
    record.txHashAccept = linkAccept.txHash;


    CKeyEd25519 linksharedkey(seed1);

    //std::cout << "Private key: " << linksharedkey.GetPrivKeyString() << "\n";

    //generate message data and pub keys based on linksharedkey
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    vvchPubKeys.push_back(linksharedkey.GetPubKeyBytes());
    uint256 subjectID = GetSubjectIDFromKey(linksharedkey);
    int64_t timestamp = GetAdjustedTime();
    int64_t stoptime = timestamp + 60; // stop relaying after 1 minute. (need timestamp for both)
    uint256 messageID = GetMessageID(linksharedkey, timestamp);

    std::vector<unsigned char> vchWalletPubKey(pubWalletKey.begin(), pubWalletKey.end());

    std::vector<unsigned char> vchMessageType = vchFromValue("testmessage");
    std::vector<unsigned char> vchMessage = vchFromValue("this is a test message");

    std::string strErrorMessage = "";

    //Broadcast wallet key
    CKey bcastprivWalletKey;
    bcastprivWalletKey.MakeNewKey(true);
    CPubKey bcastpubWalletKey = bcastprivWalletKey.GetPubKey();
    CKeyID bcastkeyWalletID = bcastpubWalletKey.GetID();
    CDynamicAddress bcastwalletAddress = CDynamicAddress(bcastkeyWalletID);
    CharString bcastvchWalletAddress = vchFromString(bcastwalletAddress.ToString());

    std::vector<unsigned char> bcastvchWalletPubKey(bcastpubWalletKey.begin(), bcastpubWalletKey.end());

    //create unsigned message
    CUnsignedVGPMessage unsignedMessage(subjectID, messageID, bcastvchWalletPubKey, timestamp, stoptime);

    //test that we can encrypt the message
    BOOST_CHECK(unsignedMessage.EncryptMessage(vchMessageType, vchMessage, vchRequestorFQDN, vvchPubKeys, false, strErrorMessage));

    //setup message for broadcast
    CVGPMessage vpgMessage(unsignedMessage);
    if (vpgMessage.vchMsg.size() > MAX_MESSAGE_SIZE)
    {
        throw std::runtime_error(strprintf("%s --Message size (%d) too large to send.\n", __func__, vpgMessage.vchMsg.size()));
    }

    //sign message
    vpgMessage.Sign(bcastprivWalletKey);

    //Check signature
    BOOST_CHECK(vpgMessage.CheckSignature(bcastvchWalletPubKey));

    std::string strErrorMessage2;

    std::vector<unsigned char> retrievedvchType, retrievedvchMessage, retrievedvchSenderFQDN;

    bool fKeepLast = false;
    //shouldn't decrypt with wrong seed
    BOOST_CHECK(unsignedMessage.DecryptMessage(invalidseed, retrievedvchType, retrievedvchMessage, retrievedvchSenderFQDN, fKeepLast, strErrorMessage2) == false);

    //decrypt with proper seed
    BOOST_CHECK(unsignedMessage.DecryptMessage(seed1, retrievedvchType, retrievedvchMessage, retrievedvchSenderFQDN, fKeepLast, strErrorMessage2));

    //message values should match (original vs decrypted retrieved value)
    BOOST_CHECK(vchMessage == retrievedvchMessage);

    std::cout << "Exit: bdap_vgp_message_tests\n";

}

BOOST_AUTO_TEST_SUITE_END()