// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MNEMONIC_H
#define DYNAMIC_MNEMONIC_H

#include <vector>
#include <string> 

#define SEED_KEY_SIZE 64
#define PBKDF2_ROUNDS 2048

class Mnemonic {
    private:
        std::vector<std::string> data;
    public:
        void setMnemonic(std::string& mnemoics);

        std::vector<std::string> getMnemonic() {return data;};
        std::vector<unsigned char> toVector();
        unsigned char *MnemonicToSeed();

        Mnemonic() {
        };

        Mnemonic(std::string mnemoics) {
            setMnemonic(mnemoics);
        };

        ~Mnemonic(){

        }
        
};

void SplitStringToVector(const std::string& source, std::vector<std::string>& v, const std::string& c);  

#endif // DYNAMIC_MNEMONIC_H