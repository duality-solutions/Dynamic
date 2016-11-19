// Copyright (c) 2009-2016 Satoshi Nakamoto
// Copyright (c) 2009-2016 The Bitcoin Developers
// Copyright (c) 2014-2016 The Dash Developers
// Copyright (c) 2015-2016 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef ENCODING_H
#define ENCODING_H
#include "argon2.h"

int encode_string(char *dst, size_t dst_len, argon2_context *ctx,
                  argon2_type type);

int decode_string(argon2_context *ctx, const char *str, argon2_type type);

#endif
