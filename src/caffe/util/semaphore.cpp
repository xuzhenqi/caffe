#include <mutex>
#include <condition_variable>
#include <caffe/util/semaphore.hpp>

namespace caffe {

void Semaphore::wait(void) {
    std::unique_lock<std::mutex> lock(mutex);
    if(count==0) {
        condition.wait(lock);
    }
    count--;
}

void Semaphore::signal(void) {
    std::lock_guard<std::mutex> lock(mutex);
    count++;
    condition.notify_one();
}

}
