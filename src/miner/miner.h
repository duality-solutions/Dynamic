// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MINER_H
#define DYNAMIC_MINER_H

#include "miner-util.h" // IWYU pragma: keep

class CConnman;
class CChainParams;

class MinersController;

/** It's constructed and set in init.cpp */
extern std::unique_ptr<MinersController> gMiners;

/** Initializes miners controller */
void InitMiners(const CChainParams& chainparams, CConnman& connman);
/** Start all miner threads */
void StartMiners();
/** Shuts down all miner threads */
void ShutdownMiners();
/** Shuts down all CPU miner threads */
void ShutdownCPUMiners();
/** Shuts down all GPU miner threads */
void ShutdownGPUMiners();

/** Gets hash rate of GPU and CPU */
int64_t GetHashRate();
/** Gets hash rate of CPU */
int64_t GetCPUHashRate();
/** Gets hash rate of GPU */
int64_t GetGPUHashRate();

/** Sets amount of CPU miner threads */
void SetCPUMinerThreads(uint8_t target);
/** Sets amount of GPU miner threads */
void SetGPUMinerThreads(uint8_t target);

#endif // DYNAMIC_MINER_H
