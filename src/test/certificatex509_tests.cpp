// Copyright (c) 2016-2020 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2020 The Bitcoin Developers
// Copyright (c) 2009-2020 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "key.h"

#include "base58.h"
#include "script/script.h"
#include "uint256.h"
#include "util.h"
#include "utilstrencodings.h"
#include "test/test_dynamic.h"
#include "bdap/utils.h"
#include "bdap/x509certificate.h"
#include "dht/ed25519.h"

#include "primitives/transaction.h"

#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <libtorrent/ed25519.hpp>

#include <boost/test/unit_test.hpp>

class CTransaction;

//certificate_tests
BOOST_FIXTURE_TEST_SUITE(certificatex509_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(certificatex509_test1)
{
    {
        std::string strErrorMsg;

        std::string strLen32 = "12345678901234567890123456789012";
        std::string strLen33 = "123456789012345678901234567890123";

        std::string strLen191 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901";
        std::string strLen192 = "123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012";

        std::string strLen512 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012";
        std::string strLen513 = "123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123";

        std::string strLen100 = "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890";
        std::string strLen101 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901";

        std::string strLen3600 = "---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0";
        std::string strLen3601 = "---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0-";

        std::string strLen256 = "---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5------";
        std::string strLen257 = "---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5-------";

        CX509Certificate testCertificate;

        //test no subject, no SubjectSignature, no PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a PEM
        testCertificate.PEM = vchFromString(" ");

        //set a subject
        testCertificate.Subject = vchFromString("test@bdap.io");

        //test no SubjectSignature, PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a SubjectSignature
        testCertificate.SubjectSignature = vchFromString(strLen512);

        //test no PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a Public Key
        testCertificate.SubjectPublicKey = vchFromString(strLen512);

        //test populated subject, subjectsignature, publickey GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set a PEM Bad
        testCertificate.PEM = vchFromString(strLen3601);

        //test PEM BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a PEM Good
        testCertificate.PEM = vchFromString(strLen3600);

        //test PEM Good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set Subject bad
        testCertificate.Subject = vchFromString(strLen192);

        //test Subject BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set Subject good
        testCertificate.Subject = vchFromString(strLen191);

        //test Subject GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SubjectSignature bad
        testCertificate.SubjectSignature = vchFromString(strLen513);

        //test SubjectSignature BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SubjectSignature good
        testCertificate.SubjectSignature = vchFromString(strLen512);

        //test SubjectSignature GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set Issuer bad
        testCertificate.Issuer = vchFromString(strLen192);

        //test Issuer BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set Issuer good
        testCertificate.Issuer = vchFromString(strLen191);

        //test Issuer GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SubjectSignature bad
        testCertificate.SubjectSignature = vchFromString(strLen513);

        //test SubjectSignature BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SubjectSignature good
        testCertificate.SubjectSignature = vchFromString(strLen512);

        //test SubjectSignature GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set IssuerSignature bad
        testCertificate.IssuerSignature = vchFromString(strLen513);

        //test IssuerSignature BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SignatureValue good
        testCertificate.IssuerSignature = vchFromString(strLen512);

        //test IssuerSignature GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set PublicKey bad
        testCertificate.SubjectPublicKey = vchFromString(strLen513);

        //test PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set PublicKey good
        testCertificate.SubjectPublicKey = vchFromString(strLen512);

        //test PublicKey GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set ExternalVerificationFile BAD
        testCertificate.ExternalVerificationFile = vchFromString(strLen257);

        //test ExternalVerificationFile BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set ExternalVerificationFile GOOD
        testCertificate.ExternalVerificationFile = vchFromString(strLen256);

        //test ExternalVerificationFile GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        std::cout << "Exit: certificatex509_test1\n";
    }
} //certificate_test1

BOOST_AUTO_TEST_CASE(certificatex509_test2)
{
    {
        //signing test
        std::string strErrorMsg;

        std::string strLen512 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012";
        std::string strLen3600 = "---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0---------1---------2---------3---------4---------5---------6---------7---------8---------9---------0";

        CX509Certificate testCertificate;

        CKeyEd25519 privSubjectDHTKey;
        CKeyEd25519 privIssuerDHTKey;
        CKeyEd25519 privUnknownDHTKey;
        CKeyEd25519 privCertificateDHTKey;

        std::vector<unsigned char> SubjectSecretKey;
        std::vector<unsigned char> SubjectPublicKey;

        std::vector<unsigned char> IssuerSecretKey;
        std::vector<unsigned char> IssuerPublicKey;

        std::vector<unsigned char> UnknownSecretKey;
        std::vector<unsigned char> UnknownPublicKey;

        std::vector<unsigned char> CertificateSecretKey;
        std::vector<unsigned char> CertificatePublicKey;
        std::vector<unsigned char> CertificatePrivSeedBytes;

        SubjectSecretKey = privSubjectDHTKey.GetPrivKeyBytes();
        SubjectPublicKey = privSubjectDHTKey.GetPubKeyBytes();

        IssuerSecretKey = privIssuerDHTKey.GetPrivKeyBytes();
        IssuerPublicKey = privIssuerDHTKey.GetPubKeyBytes();

        UnknownSecretKey = privUnknownDHTKey.GetPrivKeyBytes();
        UnknownPublicKey = privUnknownDHTKey.GetPubKeyBytes();

        CertificateSecretKey = privCertificateDHTKey.GetPrivKeyBytes();
        CertificatePublicKey = privCertificateDHTKey.GetPubKeyBytes();
        CertificatePrivSeedBytes = privCertificateDHTKey.GetPrivSeedBytes();

        //set Certificate public key
        testCertificate.SubjectPublicKey = CertificatePublicKey;

        //set a serialnumber
        testCertificate.SerialNumber = GetTimeMicros();

        //set a subject
        testCertificate.Subject = vchFromString("test@bdap.io");

        //set a PEM
        testCertificate.PEM = vchFromString(strLen3600);

        //set a SubjectSignature
        testCertificate.SubjectSignature = vchFromString(strLen512);

        //set a Public Key
        //testCertificate.SubjectPublicKey = vchFromString(strLen512);

        //Subject Signs GOOD
        BOOST_CHECK(testCertificate.SignSubject(SubjectPublicKey, SubjectSecretKey) == true);

        //Validate after subject signs
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Check Subject Signature GOOD
        BOOST_CHECK(testCertificate.CheckSubjectSignature(SubjectPublicKey) == true);

        //Check Subject Signature using Unknown BAD
        BOOST_CHECK(!(testCertificate.CheckSubjectSignature(UnknownPublicKey) == true));

        //Check, certificate should NOT be approved yet
        BOOST_CHECK((testCertificate.IsApproved() == false));

        //set an issuer
        testCertificate.Issuer = vchFromString("test2@bdap.io");

        //Issuer Signs GOOD
        BOOST_CHECK(testCertificate.SignIssuer(IssuerPublicKey, IssuerSecretKey) == true);

        //Check, certificate should be approved
        BOOST_CHECK((testCertificate.IsApproved() == true));

        //Validate after issuer signs
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Check Issuer Signature GOOD
        BOOST_CHECK(testCertificate.CheckIssuerSignature(IssuerPublicKey) == true);

        //Check Issuer Signature using Unknown BAD
        BOOST_CHECK(!(testCertificate.CheckIssuerSignature(UnknownPublicKey) == true));

        //Check against Invalid PEM BAD
        BOOST_CHECK(!(testCertificate.ValidatePEM(strErrorMsg) == true));

        //Self sign test
        //Issuer = Subject
        testCertificate.Issuer = testCertificate.Subject;

        //Check X509 Root CA Sign GOOD
        BOOST_CHECK(testCertificate.X509RootCASign(CertificatePrivSeedBytes));

        //Check against PEM GOOD
        BOOST_CHECK(testCertificate.ValidatePEM(strErrorMsg) == true);

        //std::cout << "Error msg: " << strErrorMsg << "\n";

        //set issuer to a different value
        testCertificate.Issuer = vchFromString("test2@bdap.io");

        //Check against PEM BAD
        BOOST_CHECK(!(testCertificate.ValidatePEM(strErrorMsg) == true));

        //set issuer GOOD (self-sign)
        testCertificate.Issuer = testCertificate.Subject;

        //Check against PEM GOOD
        BOOST_CHECK(testCertificate.ValidatePEM(strErrorMsg) == true);

        //set certificate publickey BAD
        testCertificate.SubjectPublicKey = UnknownPublicKey;

        //Check against PEM BAD
        BOOST_CHECK(!(testCertificate.ValidatePEM(strErrorMsg) == true));


        std::cout << "Exit: certificatex509_test2\n";

    }
} //certificatex509_test2

BOOST_AUTO_TEST_CASE(certificatex509_test3)
{
    {
        //bad transaction test
        CMutableTransaction txMut;

        std::string strLen512 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012";

        CKeyEd25519 key;

        CharString data;

        data = vchFromString(strLen512);

        CScript scriptData;
        scriptData << OP_RETURN << data;

        CScript scriptPubKey;
        scriptPubKey << CScript::EncodeOP_N(OP_BDAP_MODIFY) << CScript::EncodeOP_N(OP_BDAP_CERTIFICATE) 
                 << 1 << key.GetPubKey() << OP_2DROP << OP_2DROP; 

        CTxOut txoutData(0, scriptData);
        CTxOut txoutPubKey(10, scriptPubKey);

        txMut.vout.push_back(txoutPubKey);
        txMut.vout.push_back(txoutData);

        CTransaction tx(txMut);

        CTransactionRef txRef = MakeTransactionRef(tx);

        CX509Certificate testCertificate;

        //check against bad txRef BAD
        BOOST_CHECK(!(testCertificate.UnserializeFromTx(txRef) == true));

        //create real certificate.. serialize it, then pass it to transaction.. instead of data, result of serialization.. check it should be true
        CKeyEd25519 privSubjectDHTKey;
        CKeyEd25519 privCertificateDHTKey;

        std::vector<unsigned char> SubjectSecretKey;
        std::vector<unsigned char> SubjectPublicKey;

        std::vector<unsigned char> CertificateSecretKey;
        std::vector<unsigned char> CertificatePublicKey;
        std::vector<unsigned char> CertificatePrivSeedBytes;

        SubjectSecretKey = privSubjectDHTKey.GetPrivKeyBytes();
        SubjectPublicKey = privSubjectDHTKey.GetPubKeyBytes();

        CertificateSecretKey = privCertificateDHTKey.GetPrivKeyBytes();
        CertificatePublicKey = privCertificateDHTKey.GetPubKeyBytes();
        CertificatePrivSeedBytes = privCertificateDHTKey.GetPrivSeedBytes();

        CX509Certificate goodCertificate;
        std::string strErrorMsg = "";

        //set Certificate public key
        goodCertificate.SubjectPublicKey = CertificatePublicKey;

        //set a serialnumber
        goodCertificate.SerialNumber = GetTimeMicros();

        //set a subject
        goodCertificate.Subject = vchFromString("test@bdap.io");

        //self sign certificate so subject = issuer, and sign both

        goodCertificate.Issuer = goodCertificate.Subject;

        //Subject Signs GOOD
        BOOST_CHECK(goodCertificate.SignSubject(SubjectPublicKey, SubjectSecretKey) == true);

        //Check Subject Signature GOOD
        BOOST_CHECK(goodCertificate.CheckSubjectSignature(SubjectPublicKey) == true);

        //Subject Signs GOOD (self sign root certificate so issuer = subject)
        BOOST_CHECK(goodCertificate.SignIssuer(SubjectPublicKey, SubjectSecretKey) == true);

        //Check Subject Signature GOOD
        BOOST_CHECK(goodCertificate.CheckIssuerSignature(SubjectPublicKey) == true);

        //Check X509 Root CA Sign GOOD
        BOOST_CHECK(goodCertificate.X509RootCASign(CertificatePrivSeedBytes));

        //Check against PEM GOOD
        BOOST_CHECK(goodCertificate.ValidatePEM(strErrorMsg) == true);

        //Validate Values GOOD
        BOOST_CHECK(goodCertificate.ValidateValues(strErrorMsg) == true);

        //std::cout << "Error msg: " << strErrorMsg << "\n";

        CMutableTransaction txMut2;
        CharString data2;

        goodCertificate.Serialize(data2);

        CScript scriptData2;
        scriptData2 << OP_RETURN << data2;

        CScript scriptPubKey2;
        scriptPubKey2 << CScript::EncodeOP_N(OP_BDAP_MODIFY) << CScript::EncodeOP_N(OP_BDAP_CERTIFICATE) 
                 << key.GetPubKey() << OP_2DROP << OP_DROP; 

        CTxOut txoutData2(0, scriptData2);
        CTxOut txoutPubKey2(10, scriptPubKey2);

        txMut2.vout.push_back(txoutPubKey2);
        txMut2.vout.push_back(txoutData2);

        CTransaction tx2(txMut2);

        CTransactionRef txRef2 = MakeTransactionRef(tx2);

        CX509Certificate testCertificate2;

        //check against good certificate, GOOD
        BOOST_CHECK((testCertificate2.UnserializeFromTx(txRef2) == true));

        std::cout << "Exit: certificatex509_test3\n";
    }
} //certificatex509_test3

BOOST_AUTO_TEST_SUITE_END()
