
// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/impl/miner-gpu.h"
#include "util.h"

#ifdef ENABLE_GPU
#include "miner/internal/miner-context.h"
#include "primitives/block.h"


GPUMiner::GPUMiner(MinerContextRef ctx, std::size_t device_index)
    : MinerBase(ctx, device_index),
      _global(),
      _params((std::size_t)OUTPUT_BYTES, 2, 500, 8),
      _device(_global.getAllDevices()[device_index]),
      _context(&_global, {_device}, argon2gpu::ARGON2_D, argon2gpu::ARGON2_VERSION_10),
      _batch_size_target(((_device.getTotalMemory() / 0x13F332) / 16) * 16),
      _processing_unit(&_context, &_params, &_device, _batch_size_target, false, false) {}

int64_t GPUMiner::TryMineBlock(CBlock& block)
{
    static unsigned char pblank[1];
    const auto _begin = BEGIN(block.nVersion);
    const auto _end = END(block.nNonce);
    const void* input = (_begin == _end ? pblank : static_cast<const void*>(&_begin[0]));
    const std::uint64_t device_target = ArithToUint256(_hash_target).GetUint64(3);
    std::uint32_t start_nonce = block.nNonce;

    //Increase nNonce for the next batch
    block.nNonce += _batch_size_target;

    std::uint32_t result_nonce = _processing_unit.scanNonces(input, start_nonce, device_target);

    if ( result_nonce < std::numeric_limits<uint32_t>::max()){
        block.nNonce = result_nonce;
        uint256 cpuHash = block.GetHash();
         if (UintToArith256(cpuHash) <= _hash_target) {
             LogPrintf("Dynamic GPU Miner Found Nonce %u \n", block.nNonce);
             this->ProcessFoundSolution(block, cpuHash);
         }else{
             LogPrintf("Dynamic GPU Miner False Nonce %u \n", block.nNonce);
         }

    }
    return _batch_size_target;
}

#endif // ENABLE_GPU
