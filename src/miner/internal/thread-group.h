// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_INTERNAL_THREAD_GROUP_H
#define DYNAMIC_INTERNAL_THREAD_GROUP_H

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/thread.hpp>

/**
 * Miner threads controller.
 * Separate object for CPU and GPU.
 */
template <class T, class Context>
class ThreadGroup
{
public:
    explicit ThreadGroup(Context ctx);

    // Starts set amount of target threads
    void Start() { SyncGroupTarget(); };

    // Shuts down all threads
    void Shutdown()
    {
        SetNumThreads(0);
        SyncGroupTarget();
    };

    // Sets amount of threads
    void SetNumThreads(uint8_t target)
    {
        boost::unique_lock<boost::shared_mutex> guard(_mutex);
        _target_threads = target;
    };

protected:
    // Starts or shutdowns threads to meet the target
    void SyncGroupTarget();

    Context _ctx;
    size_t _devices;
    uint8_t _target_threads = 0;
    std::vector<std::shared_ptr<boost::thread> > _threads;
    mutable boost::shared_mutex _mutex;
};

/** Miners device group class constructor */
template <class T, class Context>
ThreadGroup<T, Context>::ThreadGroup(Context ctx)
    : _ctx(ctx), _devices(T::TotalDevices()){};

template <class T, class Context>
void ThreadGroup<T, Context>::SyncGroupTarget()
{
    boost::unique_lock<boost::shared_mutex> guard(_mutex);

    size_t current;
    while ((current = _threads.size()) != _target_threads) {
        if (current < _target_threads) {
            auto miner = std::shared_ptr<T>(new T(_ctx->MakeChild(), current % _devices));
            _threads.push_back(std::make_shared<boost::thread>([miner] {
                (*miner)();
            }));
        } else {
            std::shared_ptr<boost::thread> thread = _threads.back();
            _threads.pop_back();
            thread->interrupt();
        }
    }
};

#endif // DYNAMIC_INTERNAL_THREAD_GROUP_H
