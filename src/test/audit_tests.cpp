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
#include "bdap/audit.h"

#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <boost/test/unit_test.hpp>

//audit_tests

BOOST_FIXTURE_TEST_SUITE(audit_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(audit_test1)
{
    {
        std::string strErrorMsg;

        std::string strLen191 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901";
        std::string strLen192 = "123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012";
        std::string strLen128 = "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678";
        std::string strLen129 = "123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789";
        std::string strLen32 = "12345678901234567890123456789012";
        std::string strLen33 = "123456789012345678901234567890123";
        std::string strLen64 = "1234567890123456789012345678901234567890123456789012345678901234";

        std::string strLen64a = "a234567890123456789012345678901234567890123456789012345678901234";
        std::string strLen64b = "b234567890123456789012345678901234567890123456789012345678901234";
        std::string strLen64c = "c234567890123456789012345678901234567890123456789012345678901234";
        std::string strLen64d = "d234567890123456789012345678901234567890123456789012345678901234";

        std::string strLen65 = "12345678901234567890123456789012345678901234567890123456789012345";
        std::string strLen66 = "123456789012345678901234567890123456789012345678901234567890123456";

        CAuditData goodauditData;
        CAuditData badauditData;
        CAuditData emptyData;
        CAuditData duplicateData;

        //good unique hashes
        goodauditData.vAuditData.push_back(vchFromString(strLen64d));
        goodauditData.vAuditData.push_back(vchFromString(strLen64c));
        goodauditData.vAuditData.push_back(vchFromString(strLen64b));
        goodauditData.vAuditData.push_back(vchFromString(strLen64a));

        //one bad hash
        badauditData.vAuditData.push_back(vchFromString(strLen64a));
        badauditData.vAuditData.push_back(vchFromString(strLen64b));
        badauditData.vAuditData.push_back(vchFromString(strLen65)); //the bad one
        badauditData.vAuditData.push_back(vchFromString(strLen64d));

        //duplicate hashes
        duplicateData.vAuditData.push_back(vchFromString(strLen64));
        duplicateData.vAuditData.push_back(vchFromString(strLen64));
    
        CAudit badaudit(badauditData);
        CAudit audit(goodauditData);
        CAudit emptyaudit(emptyData);
        CAudit duplicateaudit(duplicateData);

        //test no hashes BAD
        BOOST_CHECK(!emptyaudit.ValidateValues(strErrorMsg) == true);

        //test duplicate data BAD
        BOOST_CHECK(!duplicateaudit.ValidateValues(strErrorMsg) == true);

        //test hash length BAD
        BOOST_CHECK(!badaudit.ValidateValues(strErrorMsg) == true);

        //test hash length GOOD
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        //test Algorithm Type Length BAD
        audit.vchAlgorithmType = vchFromString(strLen33);
        BOOST_CHECK(!audit.ValidateValues(strErrorMsg) == true);

        //test Algorithm Type Length GOOD
        audit.vchAlgorithmType = vchFromString(strLen32);
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        //test Description Length BAD
        audit.vchDescription = vchFromString(strLen129);
        BOOST_CHECK(!audit.ValidateValues(strErrorMsg) == true);

        //test Description Length Good
        audit.vchDescription = vchFromString(strLen128);
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        //test IsSigned FALSE
        BOOST_CHECK(!audit.IsSigned() == true);

        //test Signature Length BAD
        audit.vchSignature = vchFromString(strLen66);
        BOOST_CHECK(!audit.ValidateValues(strErrorMsg) == true);

        //test Signature Length GOOD
        audit.vchSignature = vchFromString(strLen65);
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        //test IsSigned TRUE
        BOOST_CHECK(audit.IsSigned() == true);

        //test OwnerFullObjectPath Length BAD
        audit.vchOwnerFullObjectPath = vchFromString(strLen192);
        BOOST_CHECK(!audit.ValidateValues(strErrorMsg) == true);

        //test OwnerFullObjectPath Length GOOD
        audit.vchOwnerFullObjectPath = vchFromString(strLen191);
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        std::cout << "Exit: audit_test1\n";
    }
} //audit_test1


BOOST_AUTO_TEST_CASE(audit_test2)
{
    {
        //Test Signing
        
        std::string strErrorMsg;

        std::string strLen64 = "1234567890123456789012345678901234567890123456789012345678901234";
        std::string strLen64c = "c234567890123456789012345678901234567890123456789012345678901234";
        std::string strLen64d = "d234567890123456789012345678901234567890123456789012345678901234";

        CAuditData goodauditData;

        goodauditData.vAuditData.push_back(vchFromString(strLen64d));
        goodauditData.vAuditData.push_back(vchFromString(strLen64c));

        CAudit audit(goodauditData);

        //test hash data GOOD
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        //test IsSigned FALSE, not signed yet
        BOOST_CHECK(!audit.IsSigned() == true);

        CKey key;
        key.MakeNewKey(true);
        audit.Sign(key);

        //Validate after signed
        BOOST_CHECK(audit.ValidateValues(strErrorMsg) == true);

        CPubKey pubKey;
        pubKey = key.GetPubKey();

        //test IsSigned TRUE
        BOOST_CHECK(audit.IsSigned() == true);

        //test valid signature
        BOOST_CHECK(audit.CheckSignature(pubKey.Raw()));

        std::cout << "Exit: audit_test2\n";
    }
} //audit_test2

BOOST_AUTO_TEST_SUITE_END()
