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
#include "bdap/certificate.h"
#include "dht/ed25519.h"

#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <libtorrent/ed25519.hpp>

#include <boost/test/unit_test.hpp>

//certificate_tests
BOOST_FIXTURE_TEST_SUITE(certificate_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(certificate_test1)
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

        CCertificate testCertificate;

        //test no subject, no SubjectSignature, no PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a subject
        testCertificate.Subject = vchFromString("test@bdap.io");

        //test no SubjectSignature, PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a SubjectSignature
        testCertificate.SubjectSignature = vchFromString(strLen512);

        //test no PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //set a Public Key
        testCertificate.PublicKey = vchFromString(strLen512);

        //test populated subject, subjectsignature, publickey GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SignatureAlgorithm bad
        testCertificate.SignatureAlgorithm = vchFromString(strLen33);

        //test SignatureAlgorithm BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //std::cout << std::to_string(testCertificate.SignatureAlgorithm.size()) << "\n";

        //Set SignatureAlgorithm good
        testCertificate.SignatureAlgorithm = vchFromString(strLen32);

        //test SignatureAlgorithm GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SignatureHashAlgorithm bad
        testCertificate.SignatureHashAlgorithm = vchFromString(strLen33);

        //test SignatureHashAlgorithm BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SignatureHashAlgorithm good
        testCertificate.SignatureHashAlgorithm = vchFromString(strLen32);

        //test SignatureHashAlgorithm GOOD
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

        //Set SignatureValue bad
        testCertificate.SignatureValue = vchFromString(strLen513);

        //test SignatureValue BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set SignatureValue good
        testCertificate.SignatureValue = vchFromString(strLen512);

        //test SignatureValue GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //Set PublicKey bad
        testCertificate.PublicKey = vchFromString(strLen513);

        //test PublicKey BAD
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //Set PublicKey good
        testCertificate.PublicKey = vchFromString(strLen512);

        //test PublicKey GOOD
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);


        //set KeyUsage length bad
        testCertificate.KeyUsage.push_back(vchFromString(strLen101));

        //test KeyUsage length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out KeyUsage
        testCertificate.KeyUsage.clear();

        //set KeyUsage length good
        testCertificate.KeyUsage.push_back(vchFromString(strLen100));

        //test KeyUsage length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out KeyUsage
        testCertificate.KeyUsage.clear();

        //set KeyUsage records bad
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record1
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record2
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record3
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record4
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record5
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record6
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record7
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record8
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record9
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record10
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record11 bad

        //test KeyUsage records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out KeyUsage
        testCertificate.KeyUsage.clear();

        //set KeyUsage records good
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record1
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record2
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record3
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record4
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record5
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record6
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record7
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record8
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record9
        testCertificate.KeyUsage.push_back(vchFromString(strLen100)); //record10

        //test KeyUsage records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set ExtendedKeyUsage length bad
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen101));

        //test ExtendedKeyUsage length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out ExtendedKeyUsage
        testCertificate.ExtendedKeyUsage.clear();

        //set ExtendedKeyUsage length good
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100));

        //test ExtendedKeyUsage length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out ExtendedKeyUsage
        testCertificate.ExtendedKeyUsage.clear();

        //set ExtendedKeyUsage records bad
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record1
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record2
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record3
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record4
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record5
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record6
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record7
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record8
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record9
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record10
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record11 bad

        //test ExtendedKeyUsage records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out ExtendedKeyUsage
        testCertificate.ExtendedKeyUsage.clear();

        //set ExtendedKeyUsage records good
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record1
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record2
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record3
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record4
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record5
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record6
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record7
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record8
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record9
        testCertificate.ExtendedKeyUsage.push_back(vchFromString(strLen100)); //record10

        //test ExtendedKeyUsage records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set AuthorityInformationAccess length bad
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen101));

        //test AuthorityInformationAccess length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out AuthorityInformationAccess
        testCertificate.AuthorityInformationAccess.clear();

        //set AuthorityInformationAccess length good
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100));

        //test AuthorityInformationAccess length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out AuthorityInformationAccess
        testCertificate.AuthorityInformationAccess.clear();

        //set AuthorityInformationAccess records bad
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record1
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record2
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record3
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record4
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record5
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record6
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record7
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record8
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record9
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record10
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record11 bad

        //test AuthorityInformationAccess records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out AuthorityInformationAccess
        testCertificate.AuthorityInformationAccess.clear();

        //set AuthorityInformationAccess records good
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record1
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record2
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record3
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record4
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record5
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record6
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record7
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record8
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record9
        testCertificate.AuthorityInformationAccess.push_back(vchFromString(strLen100)); //record10

        //test AuthorityInformationAccess records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set SubjectAlternativeName length bad
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen101));

        //test SubjectAlternativeName length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out SubjectAlternativeName
        testCertificate.SubjectAlternativeName.clear();

        //set SubjectAlternativeName length good
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100));

        //test SubjectAlternativeName length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out SubjectAlternativeName
        testCertificate.SubjectAlternativeName.clear();

        //set SubjectAlternativeName records bad
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record1
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record2
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record3
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record4
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record5
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record6
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record7
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record8
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record9
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record10
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record11 bad

        //test SubjectAlternativeName records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out SubjectAlternativeName
        testCertificate.SubjectAlternativeName.clear();

        //set SubjectAlternativeName records good
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record1
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record2
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record3
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record4
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record5
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record6
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record7
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record8
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record9
        testCertificate.SubjectAlternativeName.push_back(vchFromString(strLen100)); //record10

        //test SubjectAlternativeName records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set Policies length bad
        testCertificate.Policies.push_back(vchFromString(strLen101));

        //test Policies length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out Policies
        testCertificate.Policies.clear();

        //set Policies length good
        testCertificate.Policies.push_back(vchFromString(strLen100));

        //test Policies length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out Policies
        testCertificate.Policies.clear();

        //set Policies records bad
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record1
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record2
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record3
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record4
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record5
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record6
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record7
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record8
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record9
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record10
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record11 bad

        //test Policies records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out Policies
        testCertificate.Policies.clear();

        //set Policies records good
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record1
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record2
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record3
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record4
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record5
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record6
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record7
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record8
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record9
        testCertificate.Policies.push_back(vchFromString(strLen100)); //record10

        //test Policies records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set CRLDistributionPoints length bad
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen101));

        //test CRLDistributionPoints length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out CRLDistributionPoints
        testCertificate.CRLDistributionPoints.clear();

        //set CRLDistributionPoints length good
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100));

        //test CRLDistributionPoints length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out CRLDistributionPoints
        testCertificate.CRLDistributionPoints.clear();

        //set CRLDistributionPoints records bad
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record1
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record2
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record3
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record4
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record5
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record6
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record7
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record8
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record9
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record10
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record11 bad

        //test CRLDistributionPoints records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out CRLDistributionPoints
        testCertificate.CRLDistributionPoints.clear();

        //set CRLDistributionPoints records good
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record1
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record2
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record3
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record4
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record5
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record6
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record7
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record8
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record9
        testCertificate.CRLDistributionPoints.push_back(vchFromString(strLen100)); //record10

        //test CRLDistributionPoints records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //set SCTList length bad
        testCertificate.SCTList.push_back(vchFromString(strLen101));

        //test SCTList length bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out SCTList
        testCertificate.SCTList.clear();

        //set SCTList length good
        testCertificate.SCTList.push_back(vchFromString(strLen100));

        //test SCTList length good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out SCTList
        testCertificate.SCTList.clear();

        //set SCTList records bad
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record1
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record2
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record3
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record4
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record5
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record6
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record7
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record8
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record9
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record10
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record11 bad

        //test SCTList records bad
        BOOST_CHECK(!testCertificate.ValidateValues(strErrorMsg) == true);

        //clear out SCTList
        testCertificate.SCTList.clear();

        //set SCTList records good
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record1
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record2
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record3
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record4
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record5
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record6
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record7
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record8
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record9
        testCertificate.SCTList.push_back(vchFromString(strLen100)); //record10

        //test SCTList records good
        BOOST_CHECK(testCertificate.ValidateValues(strErrorMsg) == true);

        std::cout << "Exit: certificate_test1\n";
    }
} //certificate_test1


BOOST_AUTO_TEST_CASE(certificate_test2)
{
    {
        //signing test
        std::string strErrorMsg;

        std::string strLen512 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012";

        CCertificate testCertificate;

        CKeyEd25519 privSubjectDHTKey;
        CKeyEd25519 privIssuerDHTKey;
        CKeyEd25519 privUnknownDHTKey;

        std::vector<unsigned char> SubjectSecretKey;
        std::vector<unsigned char> SubjectPublicKey;

        std::vector<unsigned char> IssuerSecretKey;
        std::vector<unsigned char> IssuerPublicKey;

        std::vector<unsigned char> UnknownSecretKey;
        std::vector<unsigned char> UnknownPublicKey;

        SubjectSecretKey = privSubjectDHTKey.GetPrivKeyBytes();
        SubjectPublicKey = privSubjectDHTKey.GetPubKeyBytes();

        IssuerSecretKey = privIssuerDHTKey.GetPrivKeyBytes();
        IssuerPublicKey = privIssuerDHTKey.GetPubKeyBytes();

        UnknownSecretKey = privUnknownDHTKey.GetPrivKeyBytes();
        UnknownPublicKey = privUnknownDHTKey.GetPubKeyBytes();

        //set a subject
        testCertificate.Subject = vchFromString("test@bdap.io");

        //set a SubjectSignature
        testCertificate.SubjectSignature = vchFromString(strLen512);

        //set a Public Key
        testCertificate.PublicKey = vchFromString(strLen512);

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


        std::cout << "Exit: certificate_test2\n";

    }
} //certificate_test2

BOOST_AUTO_TEST_SUITE_END()