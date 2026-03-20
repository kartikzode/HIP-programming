#pragma once

// HIP API
#include <hip/hip_runtime.h> // hipGetErrorString

// STL
#include <cstdlib> // std::exit
#include <iostream> // std::cerr, std::endl

namespace hip
{
void check(hipError_t err, const char* name)
{
    if(err != hipSuccess)
    {
        std::cerr << name << "(" << hipGetErrorString(err) << ")" << std::endl;
        std::exit(err);
    }
}

template<class T, size_t Size>
class static_array
{
public:
    using value_type      = T;
    using size_type       = size_t;
    using reference       = T&;
    using const_reference = const T&;

    [[nodiscard]] __device__ __host__ constexpr size_type size() const noexcept
    {
        return Size;
    }
    [[nodiscard]] __device__ __host__ constexpr size_type max_size() const noexcept
    {
        return Size;
    }
    [[nodiscard]] __device__ __host__ constexpr bool empty() const noexcept
    {
        return false;
    }
    [[nodiscard]] __device__ __host__ constexpr reference front() noexcept
    {
        return _elems[0];
    }
    [[nodiscard]] __device__ __host__ constexpr const_reference front() const noexcept
    {
        return _elems[0];
    }
    [[nodiscard]] __device__ __host__ constexpr reference back() noexcept
    {
        return _elems[Size - 1];
    }
    [[nodiscard]] __device__ __host__ constexpr const_reference back() const noexcept
    {
        return _elems[Size - 1];
    }

    T _elems[Size];
};

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr T& get(static_array<T, Size>& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr const T& get(const static_array<T, Size>& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr T&& get(static_array<T, Size>&& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}

template<size_t I, class T, size_t Size>
[[nodiscard]] constexpr const T&& get(const static_array<T, Size>&& arr) noexcept
{
    static_assert(I < Size, "array index out of bounds");
    return arr._elems[I];
}
} // namespace hip

#define HIP_CHECK(call) ::hip::check((call), #call)