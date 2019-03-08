// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_PUTBUFFER_H
#define DYNAMIC_DHT_PUTBUFFER_H

#include <cstddef>
#include <memory>
#include <mutex>

template <class T>
class CPutBuffer
{
public:
    explicit CPutBuffer (size_t size) :
    buffer(std::unique_ptr<T[]>(new T[size])),
    max_size(size)
    {
        //empty constructor
    }

    void Reset()
    {
        std::lock_guard<std::mutex> lock(mutex);
        head = tail;
        full = false;
    }

    bool Empty() const
    {
        //if head and tail are equal, we are empty
        return (!full && (head == tail));
    }

    bool Full() const
    {
        //If tail is ahead the head by 1, we are full
        return full;
    }

    size_t Capacity() const
    {
        return max_size;
    }

    size_t Size() const
    {
        size_t size = max_size;
        if(!full)
        {
            if(head >= tail)
            {
                size = head - tail;
            }
            else
            {
                size = max_size + head - tail;
            }
        }
        return size;
    }

    void PushBack(T item)
    {
        std::lock_guard<std::mutex> lock(mutex);

        buffer[head] = item;

        if(full)
        {
            tail = (tail + 1) % max_size;
        }

        head = (head + 1) % max_size;

        full = head == tail;
    }

    T GetLast()
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(Empty())
        {
            return T();
        }
        //Read data and advance the tail (we now have a free space)
        auto val = buffer[tail];
        full = false;
        tail = (tail + 1) % max_size;

        return val;
    }

private:
    std::mutex mutex;
    std::unique_ptr<T[]> buffer;
    size_t head = 0;
    size_t tail = 0;
    const size_t max_size;
    bool full = 0;
};

#endif // DYNAMIC_DHT_PUTBUFFER_H
