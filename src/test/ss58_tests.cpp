// Copyright (c) 2021 - present Duality Blockchain Solutions Developers
// Copyright (c) 2009-2020 The Bitcoin Developers
// Copyright (c) 2009-2020 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "test_dynamic.h"

#include "swap/ss58.h"

#include <iostream>
#include <string>

#include <boost/test/unit_test.hpp>

//ss58_tests

BOOST_FIXTURE_TEST_SUITE(ss58_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(ss58_test1)
{
    {
        /*
            ./target/release/subkey inspect 5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eDs
                Public Key URI `5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eDs` is account:
                Network ID/version: substrate
                Public key (hex):   0x98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c8138
                Account ID:         0x98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c8138
                Public key (SS58):  5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eDs
                SS58 Address:       5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eDs
            ./dynmaic-cli ss58valid 5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eDs
                {
                  "address_bytes": 35,
                  "hex_address": "0x2a98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c81380556",
                  "address_type": 42,
                  "address_pubkey": "0x98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c8138",
                  "address_checksum": "0x0556",
                  "calculated_checksum": "0x0556",
                  "calculated_hash": "46eec7daea2c478c2ef86b969ffd6116c804b19ef6f6c304ae321d726f3115dfff4a9b60630873f1b6bfbe36ad4fdcbcb7ea93fa87ca2c2b42fd1eb17cee5605",
                  "valid_checksum": "true",
                  "valid": "true"
                }
        */

        std::string strValidSS58Address = "5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eDs";
        CSS58 validAddress(strValidSS58Address);

        //test address valid
        BOOST_CHECK(validAddress.Valid() == true);

        //test address length
        BOOST_CHECK(validAddress.nLength == 35);

        //test address hex
        BOOST_CHECK(validAddress.PublicKeyHex() == "0x98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c8138");

        //test address type
        BOOST_CHECK(validAddress.AddressType() == 42); // Substrate 

        //test address checksum
        BOOST_CHECK(validAddress.ValidChecksum() == true);

        //test address checksum
        BOOST_CHECK(validAddress.strError == "");

        std::cout << "Exit: ss58_test1\n";
    }
} //ss58_test1

BOOST_AUTO_TEST_CASE(ss58_test2)
{
    {
        // invalid Substrate address
        std::string strValidSS58Address = "5FX51QrBFqDD7p4p7QXDHBXUbNErxfTdBDNh68nCFSek2eD4"; // changed last hex char from 's' to '4'
        CSS58 validAddress(strValidSS58Address);

        //test address invalid with checksum
        BOOST_CHECK(validAddress.Valid() == false);

        //test address length
        BOOST_CHECK(validAddress.nLength == 35);

        //test address hex
        BOOST_CHECK(validAddress.PublicKeyHex() != "0x98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c0000");

        //test address type
        BOOST_CHECK(validAddress.AddressType() != 1); // Polkadot 

        //test address pubkey against subkey
        BOOST_CHECK(validAddress.PublicKeyHex() == "0x98d047d74178c6f45ed8ef01cfbdeb00f06448764393b54ac668de47d31c8138");

        //test address invalid checksum
        BOOST_CHECK(validAddress.ValidChecksum() == false);

        //test address checksum
        BOOST_CHECK(validAddress.strError != "");

        std::cout << "Exit: ss58_test2: " << validAddress.strError << "\n";
    }
} //ss58_test2

BOOST_AUTO_TEST_CASE(ss58_test3)
{
    {
        // invalid Base58 address. Decode should fail
        std::string strValidSS58Address = "1234567890OIUYTREWQPSLDFGHJKLMNNBVCXqwertyuioplkjhgfdsazxcvbnm"; // invalid Base58
        CSS58 validAddress(strValidSS58Address);

        //test address invalid base58
        BOOST_CHECK(validAddress.fValid == false);

        //test address invalid with checksum
        BOOST_CHECK(validAddress.Valid() == false);

        //test address checksum
        BOOST_CHECK(validAddress.strError != "");

        std::cout << "Exit: ss58_test3: " << validAddress.strError << "\n";
    }
} //ss58_test3

BOOST_AUTO_TEST_CASE(ss58_test4)
{
    {
        // invalid Base58 address. Decode should fail
        std::string strValidSS58Address = "dmz1FWyDq9wmCfEcRcPHxrHAP8c4nUmrFcJfUHz7TXKcJfEcx"; // Valid calamari address
        CSS58 validAddress(strValidSS58Address);

        //test address invalid
        BOOST_CHECK(validAddress.fValid == false);

        //test address invalid with checksum
        BOOST_CHECK(validAddress.Valid() == false);

        //test valid checksum
        BOOST_CHECK(validAddress.ValidChecksum() == true);

        //test address checksum
        BOOST_CHECK(validAddress.strError != "");

        std::cout << "Exit: ss58_test4: " << validAddress.strError << "\n";
    }
} //ss58_test4


BOOST_AUTO_TEST_CASE(ss58_test5)
{
    {
        // valid Base58 address. Checksums should match
        std::vector<std::string> vchAddresses {
            "VkEbUGFiS2Gy3pfjJ4MGYQa8jZUwtCwJSyYKBtqePrc5Ts2oE",
            "Vdt4fktKjs8KxYQoyeBiPsXKRDP6sVSaGbM9JHY5KLvo8tJEQ",
            "jHHm45rCwmecZVhyaKPJV1GgDyfXZHXzX6j2JeJmaHSbUqQpp",
            "dmz1FWyDq9wmCfEcRcPHxrHAP8c4nUmrFcJfUHz7TXKcJfEcx",
            "cTLTjC3Nuciej9dDretA5CZ76xfwYekDPq1f9UFbtXQ9PVBrj",
            "h4Y42oh5yHBFhmewfsbMjwLgygzSTf5UAqhsoWhktUqiLQv",
            "2hBP4PF3wemfVyT5exJziQY69mD3uC4q4ApoxfhgH4xBEMkw",
            "165L57PZtQxpWysQTB7RWk7g9Ue7jeJt1jmv9j1NdHn49Xr3",
            "Cb2QccEAM38pjwmcHHTTuTukUobTHwhakKH4kBo4k8Vur8o",
            "3CztBJtrsWZUcGb2hygxggCtoXBXbn8ehFTsKZyZjQCzhsiH"
        };

        for (const std::string& address : vchAddresses) {
            CSS58 validAddress(address);

            //test valid checksum
            BOOST_CHECK(validAddress.ValidChecksum() == true);
            std::cout << "testing checksum matches: " << address << " passed (" 
                        << validAddress.AddressChecksumHex() << " == " << validAddress.CalulatedChecksumHex() << ")\n";
        }

        std::cout << "Exit: ss58_test5: " << "\n";
    }
} //ss58_test5

BOOST_AUTO_TEST_SUITE_END()
