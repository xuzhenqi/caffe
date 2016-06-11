#ifndef CAFFE_SEMAPHONE_HPP_
#define CAFFE_SEMAPHONE_HPP_
#include <mutex>
#include <condition_variable>

namespace caffe {

class Semaphore {
 public:
    Semaphore(int value=0): count(value) {}
    void wait(void);
    void signal(void);

 private:
    int count;
    std::mutex mutex;
    std::condition_variable condition;
};

}

#endif //CAFFE_SEMAPHONE_HPP_
