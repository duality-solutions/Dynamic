// Copyright (c) 2018 Duality Blockchain Solutions Developers
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
    ThreadGroup(Context ctx);

    // Starts all miner threads
    void Start() { SyncGroupTarget(); };

    // Shuts down all miner threads
    void Shutdown()
    {
        SetNumThreads(0);
        SyncGroupTarget();
    };

    // Sets amount of threads
    void SetNumThreads(int target)
    {
        boost::unique_lock<boost::shared_mutex> guard(_mutex);
        _target_threads = target;
    };

protected:
    // Starts or shutdowns threads to meet the target
    void SyncGroupTarget();

    Context _ctx;
    size_t _devices;
    size_t _target_threads = 0;
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
    size_t real_target = _target_threads * _devices;
    while ((current = _threads.size()) != real_target) {
        for (size_t device_index = 0; device_index < _devices; device_index++) {
            if (current < real_target) {
                auto miner = std::shared_ptr<T>(new T(_ctx->MakeChild(), device_index));
                _threads.push_back(std::make_shared<boost::thread>([miner] {
                    (*miner)();
                }));
            } else {
                std::shared_ptr<boost::thread> thread = _threads.back();
                _threads.pop_back();
                thread->interrupt();
            }
        }
    }
};

#endif // DYNAMIC_INTERNAL_THREAD_GROUP_H
