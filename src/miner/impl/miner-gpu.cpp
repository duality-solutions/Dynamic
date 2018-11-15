
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/impl/miner-gpu.h"

#ifdef ENABLE_GPU
#include "miner/internal/miner-context.h"
#include "primitives/block.h"


GPUMiner::GPUMiner(MinerContextRef ctx, std::size_t device_index)
    : MinerBase(ctx, device_index),
      _global(),
      _params((std::size_t)OUTPUT_BYTES, 2, 500, 8),
      _device(_global.getAllDevices()[device_index]),
      _context(&_global, {_device}, argon2gpu::ARGON2_D, argon2gpu::ARGON2_VERSION_10),
      _batch_size_target(_device.getTotalMemory() / 8192e3),
      _processing_unit(&_context, &_params, &_device, _batch_size_target, false, false) {}

int64_t GPUMiner::TryMineBlock(CBlock& block)
{
    uint32_t start_nonce = block.nNonce;
    int64_t hashes_done = 0;
    // current batch size
    std::size_t batch_size = _batch_size_target;
    // set batch input
    static unsigned char pblank[1];
    for (std::size_t i = 0; i < _batch_size_target; i++) {
        const auto _begin = BEGIN(block.nVersion);
        const auto _end = END(block.nNonce);
        const void* input = (_begin == _end ? pblank : static_cast<const void*>(&_begin[0]));
        // input is copied onto memory buffer
        _processing_unit.setInputAndSalt(i, input, INPUT_BYTES);
        // increment block nonce
        block.nNonce += 1;
        // increment hashes done
        hashes_done += 1;
        // TODO(crackcomm): is this only to count hashes?
        if ((block.nNonce & 0xFF) == 0) {
            batch_size = i + 1;
            break;
        }
    }
    // start GPU processing
    _processing_unit.beginProcessing();
    // wait for results
    _processing_unit.endProcessing();
    // check batch results
    uint256 hash;
    for (std::size_t i = 0; i < batch_size; i++) {
        _processing_unit.getHash(i, (uint8_t*)&hash);
        if (UintToArith256(hash) <= _hash_target) {
            block.nNonce = start_nonce + i;
            this->ProcessFoundSolution(block, hash);
            break;
        }
    }
    return hashes_done;
}

#endif // ENABLE_GPU
