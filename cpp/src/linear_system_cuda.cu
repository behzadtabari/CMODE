#include "linear_system_detail.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

namespace odelab {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}
template <class T> struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::invalid_argument("CUDA allocation size overflow");
    check(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(n, std::size_t(1)) * sizeof(T)));
  }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
void upload(double* dst, const std::vector<double>& src) {
  if (!src.empty()) check(cudaMemcpy(dst, src.data(), src.size()*sizeof(double), cudaMemcpyHostToDevice));
}

// Parallel cyclic reduction. Coefficients are constant for every time step,
// so save each stage's two multipliers and the final reduced diagonal once.
__global__ void factor_stage(const double* a, const double* b, const double* c,
                             double* next_a, double* next_b, double* next_c,
                             double* weight_l, double* weight_r,
                             std::size_t n, std::size_t stride, int* error) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= n) return;
  double l = 0, r = 0, d = b[i], na = 0, nc = 0;
  if (i >= stride) {
    l = -a[i] / b[i-stride];
    d += l*c[i-stride];
    na = l*a[i-stride];
  }
  if (i+stride < n) {
    r = -c[i] / b[i+stride];
    d += r*a[i+stride];
    nc = r*c[i+stride];
  }
  if (!isfinite(d) || d <= 0 || !isfinite(l) || !isfinite(r)) atomicExch(error, 1);
  next_a[i] = na; next_b[i] = d; next_c[i] = nc;
  weight_l[i] = l; weight_r[i] = r;
}
__global__ void form_rhs(const double* lower, const double* diag, const double* upper,
                         const double* y, double* rhs, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= n) return;
  double value = diag[i]*y[i];
  if (i) value += lower[i-1]*y[i-1];
  if (i+1 < n) value += upper[i]*y[i+1];
  rhs[i] = value;
}
__global__ void reduce_rhs(const double* rhs, double* next,
                           const double* weight_l, const double* weight_r,
                           std::size_t n, std::size_t stride) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= n) return;
  double value = rhs[i];
  if (i >= stride) value += weight_l[i]*rhs[i-stride];
  if (i+stride < n) value += weight_r[i]*rhs[i+stride];
  next[i] = value;
}
__global__ void finish_step(const double* rhs, const double* diagonal,
                            double* y, std::size_t n, int* error) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = rhs[i]/diagonal[i];
  if (!isfinite(y[i])) atomicExch(error, 1);
}
}  // namespace

LinearSystemResult solve_tridiagonal_cn_cuda(
    const std::vector<double>& ll, const std::vector<double>& ld,
    const std::vector<double>& lu, const std::vector<double>& rl,
    const std::vector<double>& rd, const std::vector<double>& ru,
    const std::vector<double>& initial, std::size_t steps) {
  using namespace cn_detail;
  validate(ll, ld, lu, rl, rd, ru, initial, steps);
  check(cudaDeviceSynchronize());
  const auto start = Clock::now();
  const auto n = initial.size();
  std::size_t stages = 0;
  for (std::size_t stride = 1; stride < n; stride *= 2) ++stages;
  if (stages && n > std::numeric_limits<std::size_t>::max() / stages)
    throw std::invalid_argument("CUDA reduction dimensions overflow");
  DeviceBuffer<double> a0(n), a1(n), b0(n), b1(n), c0(n), c1(n);
  DeviceBuffer<double> wl(n*stages), wr(n*stages), lower(n-1), diag(n), upper(n-1);
  DeviceBuffer<double> y(n), rhs0(n), rhs1(n);
  DeviceBuffer<int> error(1);
  check(cudaMemset(error.ptr, 0, sizeof(int)));
  std::vector<double> padded_lower(n, 0), padded_upper(n, 0);
  std::copy(ll.begin(), ll.end(), padded_lower.begin()+1);
  std::copy(lu.begin(), lu.end(), padded_upper.begin());
  upload(a0.ptr, padded_lower); upload(b0.ptr, ld); upload(c0.ptr, padded_upper);
  upload(lower.ptr, rl); upload(diag.ptr, rd); upload(upper.ptr, ru); upload(y.ptr, initial);
  double *a = a0.ptr, *b = b0.ptr, *c = c0.ptr;
  double *na = a1.ptr, *nb = b1.ptr, *nc = c1.ptr;
  const auto blocks = static_cast<unsigned>((n+255)/256);
  for (std::size_t stage = 0, stride = 1; stage < stages; ++stage, stride *= 2) {
    factor_stage<<<blocks, 256>>>(a, b, c, na, nb, nc, wl.ptr+stage*n, wr.ptr+stage*n,
                                  n, stride, error.ptr);
    check(cudaGetLastError());
    std::swap(a, na); std::swap(b, nb); std::swap(c, nc);
  }
  int failed = 0;
  check(cudaMemcpy(&failed, error.ptr, sizeof(int), cudaMemcpyDeviceToHost));
  if (failed) throw std::runtime_error("invalid CUDA reduction pivot; matrix may be ill-conditioned");
  const auto ready = Clock::now();
  for (std::size_t step = 0; step < steps; ++step) {
    form_rhs<<<blocks, 256>>>(lower.ptr, diag.ptr, upper.ptr, y.ptr, rhs0.ptr, n);
    check(cudaGetLastError());
    double *rhs = rhs0.ptr, *next = rhs1.ptr;
    for (std::size_t stage = 0, stride = 1; stage < stages; ++stage, stride *= 2) {
      reduce_rhs<<<blocks, 256>>>(rhs, next, wl.ptr+stage*n, wr.ptr+stage*n, n, stride);
      check(cudaGetLastError());
      std::swap(rhs, next);
    }
    finish_step<<<blocks, 256>>>(rhs, b, y.ptr, n, error.ptr);
    check(cudaGetLastError());
  }
  check(cudaDeviceSynchronize());
  const auto done = Clock::now();
  check(cudaMemcpy(&failed, error.ptr, sizeof(int), cudaMemcpyDeviceToHost));
  if (failed) throw std::runtime_error("non-finite CUDA linear system solution");
  std::vector<double> result(n);
  check(cudaMemcpy(result.data(), y.ptr, n*sizeof(double), cudaMemcpyDeviceToHost));
  return {std::move(result), milliseconds(start, ready), milliseconds(ready, done)};
}
}  // namespace odelab
