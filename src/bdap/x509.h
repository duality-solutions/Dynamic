// Copyright (c) 2020 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_X509_H
#define DYNAMIC_BDAP_X509_H

#include <string>

class CCertificate;

bool ExportX509Certificate(const CCertificate& certificate, const std::string& filename = "x509.pem");


#endif // DYNAMIC_BDAP_CERTIFICATE_H