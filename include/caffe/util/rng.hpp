#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"

#include "caffe/common.hpp"

namespace caffe {

typedef boost::mt19937 rng_t;

inline rng_t* caffe_rng() {
  return static_cast<caffe::rng_t*>(Caffe::rng_stream().generator());
}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                    RandomGenerator* gen, int num = 1) {
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
      difference_type;
  typedef typename boost::uniform_int<difference_type> dist_type;

  difference_type length = std::distance(begin, end);
  if (length <= 0) return;
  
  difference_type temp;
  for (difference_type i = length/num - 1; i > 0; --i) {
    dist_type dist(0, i);
    temp = dist(*gen);
    for (int j = 0; j < num; ++j){
      std::cout << i*num + j << " " << temp * num + j << std::endl;
      std::iter_swap(begin + i*num + j, begin + temp * num + j);
    }
  }
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end, int num = 1) {
  shuffle(begin, end, caffe_rng(), num);
}
}  // namespace caffe

#endif  // CAFFE_RNG_HPP_
