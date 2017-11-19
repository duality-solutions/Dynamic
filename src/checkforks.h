// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2009-2017 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_CHECKFORK_H
#define DYNAMIC_CHECKFORK_H

#include "arith_uint256.h"

#include "hash.h"
#include "util.h"
#include "uint256.h"
#include "chain.h"
#include "chainparams.h"
#include "validation.h"
#include "uint256.h"
#include "util.h"
#include "primitives/block.h"

#include <algorithm>

enum ForkID {
	DELTA_RETARGET = 1,
	PRE_DELTA_RETARGET = 2,
	START_DYNODE_PAYMENTS = 3,
	FORK_SLOT_3 = 4,
	FORK_SLOT_4 = 5
};

bool CheckForkIsTrue(ForkID identifier, const CBlockIndex* pindexLast, bool fTableFlip=false);

unsigned int LegacyRetargetBlock(const CBlockIndex* pindexLast, const CBlockHeader *pblock, const Consensus::Params&);

#endif // DYNAMIC_CHECKFORK_H