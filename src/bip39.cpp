// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/**
 * Copyright (c) 2013-2014 Tomas Dzetkulic
 * Copyright (c) 2013-2014 Pavol Rusnak
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

// Source:
// https://github.com/trezor/trezor-crypto

#include "bip39.h"
#include "bip39_english.h"
#include "bip39_chinese_simplified.h"
#include "bip39_chinese_traditional.h"
#include "bip39_french.h"
//#include "bip39_german.h"  //exclude for now, but include later
#include "bip39_italian.h"
#include "bip39_japanese.h"
#include "bip39_korean.h"
#include "bip39_russian.h"
#include "bip39_spanish.h"
#include "bip39_ukrainian.h"
#include "crypto/sha256.h"
#include "random.h"

#include "util.h"

#include <boost/algorithm/string.hpp>

#include <openssl/evp.h>


SecureString CMnemonic::Generate(int strength, Language selectLanguage)
{
    if (strength < 128 || strength > 256) {
        return SecureString();
    }
    SecureVector data(32);
    GetRandBytes(&data[0], 32);
    SecureString mnemonic = FromData(data, strength / 8,selectLanguage);
    return mnemonic;
}

// SecureString CMnemonic::FromData(const uint8_t *data, int len)
SecureString CMnemonic::FromData(const SecureVector& data, int len, Language selectLanguage)
{
    const char* const* refWordList = nullptr; //initialize

    getWordList(refWordList,selectLanguage);

    if (len % 4 || len < 16 || len > 32) {
        return SecureString();
    }

    SecureVector checksum(32);
    CSHA256().Write(&data[0], len).Finalize(&checksum[0]);

    // data
    SecureVector bits(len);
    memcpy(&bits[0], &data[0], len);
    // checksum
    bits.push_back(checksum[0]);

    int mlen = len * 3 / 4;
    SecureString mnemonic;

    int i, j, idx;
    for (i = 0; i < mlen; i++) {
        idx = 0;
        for (j = 0; j < 11; j++) {
            idx <<= 1;
            idx += (bits[(i * 11 + j) / 8] & (1 << (7 - ((i * 11 + j) % 8)))) > 0;
        }
        mnemonic.append(refWordList[idx]);
        if (i < mlen - 1) {
            mnemonic += ' ';
        }
    }

    refWordList = nullptr; //reset
    return mnemonic;
}


void CMnemonic::getWordList(const char* const* &input, Language selectLanguage) {

    switch (selectLanguage) {
        case Language::ENGLISH:
            input = wordlist;
            break;
        case Language::FRENCH:
            input = wordlist_french;
            break;
        case Language::CHINESE_SIMPLIFIED:
            input = wordlist_chinese_simplified;
            break;
        case Language::CHINESE_TRADITIONAL:
            input = wordlist_chinese_traditional;
            break;
        //exclude for now but include later    
        // case Language::GERMAN:
        //     input = wordlist_german;
        //     break;
        case Language::ITALIAN:
            input = wordlist_italian;
            break;
        case Language::JAPANESE:
            input = wordlist_japanese;
            break;
        case Language::KOREAN:
            input = wordlist_korean;
            break;
        case Language::RUSSIAN:
            input = wordlist_russian;
            break;
        case Language::SPANISH:
            input = wordlist_spanish;
            break;
        case Language::UKRAINIAN:
            input = wordlist_ukrainian;
            break;
        default:
            input = wordlist;
            break;

    } //switch

}

CMnemonic::Language CMnemonic::getLanguageEnumFromLabel(const std::string &input) {
    
    if (boost::algorithm::to_lower_copy(input) == "english") return CMnemonic::Language::ENGLISH;
    else if (boost::algorithm::to_lower_copy(input) == "french") return CMnemonic::Language::FRENCH;
    else if (boost::algorithm::to_lower_copy(input) == "chinesesimplified") return CMnemonic::Language::CHINESE_SIMPLIFIED;
    else if (boost::algorithm::to_lower_copy(input) == "chinesetraditional") return CMnemonic::Language::CHINESE_TRADITIONAL;
    //exclude for now but include later
    //else if (boost::algorithm::to_lower_copy(input) == "german") return CMnemonic::Language::GERMAN;
    else if (boost::algorithm::to_lower_copy(input) == "italian") return CMnemonic::Language::ITALIAN;
    else if (boost::algorithm::to_lower_copy(input) == "japanese") return CMnemonic::Language::JAPANESE;
    else if (boost::algorithm::to_lower_copy(input) == "korean") return CMnemonic::Language::KOREAN;
    else if (boost::algorithm::to_lower_copy(input) == "russian") return CMnemonic::Language::RUSSIAN;
    else if (boost::algorithm::to_lower_copy(input) == "spanish") return CMnemonic::Language::SPANISH;
    else if (boost::algorithm::to_lower_copy(input) == "ukrainian") return CMnemonic::Language::UKRAINIAN;
    else return CMnemonic::Language::ENGLISH;


} //getLanguageEnumFromLabel



bool CMnemonic::Check(SecureString mnemonic, Language selectLanguage)
{
    const char* const* refWordList = nullptr; //initialize


    getWordList(refWordList,selectLanguage);


    if (mnemonic.empty()) {
        return false;
    }

    uint32_t nWordCount{};

    for (size_t i = 0; i < mnemonic.size(); ++i) {
        if (mnemonic[i] == ' ') {
            nWordCount++;
        }
    }
    nWordCount++;
    // check number of words
    if (nWordCount != 12 && nWordCount != 18 && nWordCount != 24) {
        return false;
    }

    SecureString ssCurrentWord;
    SecureVector bits(32 + 1);

    uint32_t nWordIndex, ki, nBitsCount{};

    for (size_t i = 0; i < mnemonic.size(); ++i) {
        ssCurrentWord = "";
        while (i + ssCurrentWord.size() < mnemonic.size() && mnemonic[i + ssCurrentWord.size()] != ' ') {
            // if (ssCurrentWord.size() >= 26) { //was 9
            //     return false;
            // }
            ssCurrentWord += mnemonic[i + ssCurrentWord.size()];
        }
        //moved to get count AFTER space
        std::string teststring = "";
        teststring = ssCurrentWord.c_str();
        int characterSize = mbstowcs(NULL,teststring.c_str(), teststring.size());
        if (characterSize >= 25) { //was 9
            return false;
        }
        i += ssCurrentWord.size();
        nWordIndex = 0;
        for (;;) {
            if (!refWordList[nWordIndex]) { // word not found
                return false;
            }
            if (ssCurrentWord == refWordList[nWordIndex]) { // word found on index nWordIndex
                for (ki = 0; ki < 11; ki++) {
                    if (nWordIndex & (1 << (10 - ki))) {
                        bits[nBitsCount / 8] |= 1 << (7 - (nBitsCount % 8));
                    }
                    nBitsCount++;
                }
                break;
            }
            nWordIndex++;
        }
    }

    refWordList = nullptr; //reset

    if (nBitsCount != nWordCount * 11) {
        return false;
    }
    bits[32] = bits[nWordCount * 4 / 3];
    CSHA256().Write(&bits[0], nWordCount * 4 / 3).Finalize(&bits[0]);

    bool fResult = 0;
    if (nWordCount == 12) {
        fResult = (bits[0] & 0xF0) == (bits[32] & 0xF0); // compare first 4 bits
    } else if (nWordCount == 18) {
        fResult = (bits[0] & 0xFC) == (bits[32] & 0xFC); // compare first 6 bits
    } else if (nWordCount == 24) {
        fResult = bits[0] == bits[32]; // compare 8 bits
    }

    return fResult;
}

// passphrase must be at most 256 characters or code may crash
void CMnemonic::ToSeed(SecureString mnemonic, SecureString passphrase, SecureVector& seedRet)
{
    SecureString ssSalt = SecureString("mnemonic") + passphrase;
    SecureVector vchSalt(ssSalt.begin(), ssSalt.end());
    seedRet.resize(64);
    // int PKCS5_PBKDF2_HMAC(const char *pass, int passlen,
    //                    const unsigned char *salt, int saltlen, int iter,
    //                    const EVP_MD *digest,
    //                    int keylen, unsigned char *out);
    PKCS5_PBKDF2_HMAC(mnemonic.c_str(), mnemonic.size(), &vchSalt[0], vchSalt.size(), 2048, EVP_sha512(), 64, &seedRet[0]);
}
