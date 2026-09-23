#include "../include/odelab/ivp/first_order_solver.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace odelab {
namespace {

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

struct DeviceMemory {
  double* ptr = nullptr;
  explicit DeviceMemory(std::size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(double)));
  }
  ~DeviceMemory() { if (ptr) cudaFree(ptr); }
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
};

__global__ void solve_kernel(const double* a, const double* b, const double* y0,
                             double* out, std::size_t count, std::size_t steps,
                             double h, int method) {
  const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= count) return;
  const std::size_t offset = j * (steps + 1);
  double y = y0[j];
  out[offset] = y;
  const double w = method == 1 ? 1.0 : 0.5;
  const double factor = method == 0 ? 0.0 :
      (1.0 + (1.0 - w) * h * a[j]) / (1.0 - w * h * a[j]);
  const double forcing = method == 0 ? 0.0 : h * b[j] / (1.0 - w * h * a[j]);
  for (std::size_t i = 0; i < steps; ++i) {
    y = method == 0 ? y + h * (a[j] * y + b[j]) : factor * y + forcing;
    out[offset + i + 1] = y;
  }
}

}  // namespace

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<double> solve_linear_batch_cuda(
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& y0, double t0, double t1, std::size_t steps,
    Method method) {
  if (a.size() != b.size() || a.size() != y0.size())
    throw std::invalid_argument("a, b, and y0 must have equal lengths");
  if (!std::isfinite(t0) || !std::isfinite(t1) || !(t1 > t0) || steps == 0 ||
      steps == std::numeric_limits<std::size_t>::max() ||
      a.size() > std::numeric_limits<std::size_t>::max() / (steps + 1) ||
      a.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("invalid time range, steps or batch size");
  const double h = (t1 - t0) / static_cast<double>(steps);
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || !std::isfinite(y0[i]))
      throw std::invalid_argument("a, b and y0 must be finite");
    const double w = method == Method::backward_euler ? 1.0 : 0.5;
    if (method != Method::forward_euler &&
        std::abs(1.0 - w * h * a[i]) < 1e-14)
      throw std::runtime_error("implicit step has singular denominator");
  }
  std::vector<double> out(a.size() * (steps + 1));
  if (out.empty()) return out;
  DeviceMemory d_a(a.size()), d_b(b.size()), d_y0(y0.size()), d_out(out.size());
  check(cudaMemcpy(d_a.ptr, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_b.ptr, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_y0.ptr, y0.data(), y0.size() * sizeof(double), cudaMemcpyHostToDevice));
  const int code = method == Method::forward_euler ? 0 :
                   method == Method::backward_euler ? 1 : 2;
  solve_kernel<<<static_cast<unsigned>((a.size() + 255) / 256), 256>>>(
      d_a.ptr, d_b.ptr, d_y0.ptr, d_out.ptr, a.size(), steps, h, code);
  check(cudaGetLastError());
  check(cudaMemcpy(out.data(), d_out.ptr, out.size() * sizeof(double),
                   cudaMemcpyDeviceToHost));
  for (double x : out)
    if (!std::isfinite(x)) throw std::runtime_error("non-finite GPU solution");
  return out;
}

}  // namespace odelab