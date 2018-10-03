// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_MUTABLE_DATA_H
#define DYNAMIC_DHT_MUTABLE_DATA_H

#include "bdap/bdap.h"
#include "dbwrapper.h"
#include "serialize.h"
#include "sync.h"
#include "uint256.h"

static CCriticalSection cs_dht_entry;

class CMutableData {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    CharString InfoHash;  // key
    CharString PublicKey;
    CharString Signature;
    std::int64_t SequenceNumber;
    CharString Salt;
    CharString Value;

    CMutableData() {
        SetNull();
    }

    CMutableData(const CharString& infoHash, const CharString& publicKey, const CharString& signature, 
                    const std::int64_t& sequenceNumber, const CharString& salt, const CharString& value) :
                    InfoHash(infoHash), PublicKey(publicKey), Signature(signature), SequenceNumber(sequenceNumber), Salt(salt), Value(value){}

    inline void SetNull()
    {
        nVersion = CMutableData::CURRENT_VERSION;
        InfoHash.clear();
        PublicKey.clear();
        Signature.clear();
        SequenceNumber = 0;
        Salt.clear();
        Value.clear();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(InfoHash);
        READWRITE(PublicKey);
        READWRITE(Signature);
        READWRITE(VARINT(SequenceNumber));
        READWRITE(Salt);
        READWRITE(Value);
    }

    inline friend bool operator==(const CMutableData &a, const CMutableData &b) {
        return (a.InfoHash == b.InfoHash && a.PublicKey == b.PublicKey && a.Signature == b.Signature);
    }

    inline friend bool operator!=(const CMutableData &a, const CMutableData &b) {
        return !(a == b);
    }

    inline CMutableData operator=(const CMutableData &b) {
        nVersion = b.nVersion;
        InfoHash = b.InfoHash;
        PublicKey = b.PublicKey;
        Signature = b.Signature;
        SequenceNumber = b.SequenceNumber;
        Salt = b.Salt;
        Value = b.Value;
        return *this;
    }
 
    inline bool IsNull() const { return (Signature.empty()); }
    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char> &vchData, const std::vector<unsigned char> &vchHash);
};

class CMutableDataDB : public CDBWrapper {
public:
    CMutableDataDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "dht", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddMutableData(const CMutableData& data);
    bool UpdateMutableData(const CMutableData& data);
    bool ReadMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data);
    bool EraseMutableData(const std::vector<unsigned char>& vchInfoHash);
};

bool AddMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data);
bool UpdateMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data);
bool GetMutableData(const std::vector<unsigned char>& vchInfoHash, CMutableData& data);
bool PutMutableData(const std::vector<unsigned char>& vchInfoHash, const CMutableData& data);

extern CMutableDataDB* pMutableDataDB;

#endif // DYNAMIC_DHT_MUTABLE_DATA_H
