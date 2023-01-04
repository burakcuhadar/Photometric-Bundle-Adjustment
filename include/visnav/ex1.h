/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  T theta = xi.norm();

  if (theta == 0) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }

  Eigen::Matrix<T, 3, 3> skew;
  skew << 0, -xi(2, 0), xi(1, 0), xi(2, 0), 0, -xi(0, 0), -xi(1, 0), xi(0, 0),
      0;

  return Eigen::Matrix<T, 3, 3>::Identity() + sin(theta) / theta * skew +
         (1 - cos(theta)) / (theta * theta) * (skew * skew);
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  T theta = acos((mat.trace() - 1) / 2);
  if (theta == 0) {
    return Eigen::Matrix<T, 3, 1>::Zero();
  }
  Eigen::Matrix<T, 3, 1> w;
  w << (mat(2, 1) - mat(1, 2)), (mat(0, 2) - mat(2, 0)),
      (mat(1, 0) - mat(0, 1));
  w = w * theta / 2 / sin(theta);
  return w;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  Eigen::Matrix<T, 3, 1> w = xi.block(3, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> v = xi.block(0, 0, 3, 1);

  Eigen::Matrix<T, 3, 3> skew;
  skew << 0, -w(2, 0), w(1, 0), w(2, 0), 0, -w(0, 0), -w(1, 0), w(0, 0), 0;

  T theta = w.norm();

  Eigen::Matrix<T, 3, 3> J = Eigen::Matrix<T, 3, 3>::Identity();
  if (theta != 0) {
    J += (1 - cos(theta)) / (theta * theta) * skew +
         (theta - sin(theta)) / (theta * theta * theta) * skew * skew;
  }

  Eigen::Matrix<T, 4, 4> result;
  result.block(0, 0, 3, 3) = user_implemented_expmap(w);
  result.block(0, 3, 3, 1) = J * v;
  result.block(3, 0, 1, 3) = Eigen::Matrix<T, 1, 3>::Zero();
  result(3, 3) = (T)1;
  return result;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);

  Eigen::Matrix<T, 3, 1> w = user_implemented_logmap(R);
  T theta = w.norm();
  Eigen::Matrix<T, 3, 3> skew;
  skew << 0, -w(2, 0), w(1, 0), w(2, 0), 0, -w(0, 0), -w(1, 0), w(0, 0), 0;

  Eigen::Matrix<T, 3, 3> J_inv = Eigen::Matrix<T, 3, 3>::Identity();
  if (theta != 0) {
    J_inv += -0.5 * skew + (1 / (theta * theta) -
                            (1 + cos(theta)) / (2 * theta * sin(theta))) *
                               skew * skew;
  }

  Eigen::Matrix<T, 3, 1> v = J_inv * t;

  Eigen::Matrix<T, 6, 1> result;
  result.block(0, 0, 3, 1) = v;
  result.block(3, 0, 3, 1) = w;

  return result;
}

}  // namespace visnav