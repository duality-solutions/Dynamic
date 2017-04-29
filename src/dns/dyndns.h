// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2013-2017 Emercoin Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNDNS_H
#define DYNDNS_H

#include "netbase.h"
#include "pubkey.h"

#include <map>
#include <string>

#include <boost/thread.hpp>
#include <boost/xpressive/xpressive_dynamic.hpp>

#define DYNDNS_DAPSIZE     (8 * 1024)
#define DYNDNS_DAPTRESHOLD 3000 // 200K/min limit answer

#define VERMASK_NEW  -1
#define VERMASK_BLOCKED -2
#define VERMASK_NOSRL (1 << 24)

struct DNSHeader {
  static const uint32_t QR_MASK = 0x8000;
  static const uint32_t OPCODE_MASK = 0x7800; // shr 11
  static const uint32_t AA_MASK = 0x0400;
  static const uint32_t TC_MASK = 0x0200;
  static const uint32_t RD_MASK = 0x0100;
  static const uint32_t RA_MASK = 0x8000;
  static const uint32_t RCODE_MASK = 0x000F;

  uint16_t msgID;
  uint16_t Bits;
  uint16_t QDCount;
  uint16_t ANCount;
  uint16_t NSCount;
  uint16_t ARCount;

  inline void Transcode() {
    for(uint16_t *p = &msgID; p <= &ARCount; p++)
      *p = ntohs(*p);
  }
} __attribute__((packed)); // struct DNSHeader


struct DNSAP {    // DNS Amplifier Protector ExpDecay structure
  uint16_t timestamp; // Time in 64s ticks
  uint16_t ed_size; // ExpDecay output size in 64-byte units
} __attribute__((packed));

struct Verifier {
    Verifier() : mask(VERMASK_NEW) {}  // -1 == uninited, neg != -1 == cant fetch
    int32_t  mask;   // Signature Revocation List mask
    std::string   srl_tpl;   // Signature Revocation List template
    CKeyID   keyID;    // Key for verify message
}; // 72 bytes = 18 words

struct TollFree {
    TollFree(const char *re) :
  regex(boost::xpressive::sregex::compile(std::string(re))), regex_str(re)
    {}
    boost::xpressive::sregex  regex;
    std::string      regex_str;
    std::vector<std::string>    e2u;
};

class DynDns {
  public:
     DynDns(const char *bind_ip, uint16_t port_no,
     const char *gw_suffix, const char *allowed_suff,
     const char *local_fname, const char *enums, const char *tollfree, 
     uint8_t verbose);
    ~DynDns();

    void Run();

  private:
    static void StatRun(void *p);
    void HandlePacket();
    uint16_t HandleQuery();
    int  Search(uint8_t *key);
    int  LocalSearch(const uint8_t *key, uint8_t pos, uint8_t step);
    int  Tokenize(const char *key, const char *sep2, char **tokens, char *buf);
    void Answer_ALL(uint16_t qtype, char *buf);
    void Fill_RD_IP(char *ipddrtxt, int af);
    void Fill_RD_DName(char *txt, uint8_t mxsz, int8_t txtcor);
    int  TryMakeref(uint16_t label_ref);

    // Handle Special function - phone number in the E.164 format
    // to support ENUM service
    int SpfunENUM(uint8_t len, uint8_t **domain_start, uint8_t **domain_end);
    // Generate answewr for found EMUM NVS record
    void Answer_ENUM(const char *q_str);
    void HandleE2U(char *e2u);
    bool CheckEnumSig(const char *q_str, char *sig_str);
    void AddTF(char *tf_tok);

    // Returns x = hash index to update size; x==NULL = disable;
    DNSAP  *CheckDAP(uint32_t ip_addr);

    inline void Out2(uint16_t x) { x = htons(x); memcpy(m_snd, &x, 2); m_snd += 2; }
    inline void Out4(uint32_t x) { x = htonl(x); memcpy(m_snd, &x, 4); m_snd += 4; }
    void OutS(const char *p);

    DNSHeader *m_hdr; // 1st bzero element
    DNSAP    *m_dap_ht; // Hashtable for DAP; index is hash(IP)
    char     *m_value;
    const char *m_gw_suffix;
    uint8_t  *m_buf, *m_bufend, *m_snd, *m_rcv, *m_rcvend;
    SOCKET    m_sockfd;
    int       m_rcvlen;
    uint32_t  m_daprand;  // DAP random value for universal hashing
    uint32_t  m_ttl;
    uint16_t  m_label_ref;
    uint16_t  m_gw_suf_len;
    char     *m_allowed_base;
    char     *m_local_base;
    int16_t   m_ht_offset[0x100]; // Hashtable for allowed TLD-suffixes(>0) and local names(<0)
    struct sockaddr_in m_clientAddress;
    struct sockaddr_in m_address;
    socklen_t m_addrLen;
    uint8_t   m_gw_suf_dots;
    uint8_t   m_allowed_qty;
    uint8_t   m_verbose;  // LAST bzero element
        
    int8_t    m_status;
    boost::thread m_thread;
    std::map<std::string, Verifier> m_verifiers;
    std::vector<TollFree>      m_tollfree;
}; // class DynDns

#endif // DYNDNS_H

