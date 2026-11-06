// Minimal self-contained test harness (no external deps). Test code only.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace nftest {
using Fn = void (*)();
struct Case { const char* name; Fn fn; };
inline Case* cases() { static Case c[256]; return c; }
inline int& count() { static int n = 0; return n; }
inline int& failures() { static int n = 0; return n; }
inline int& checks() { static int n = 0; return n; }
struct Reg { Reg(const char* n, Fn f) { cases()[count()++] = Case{n, f}; } };
inline int run_all() {
  int failed_cases = 0;
  for (int i = 0; i < count(); ++i) {
    int before = failures();
    cases()[i].fn();
    bool ok = failures() == before;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", cases()[i].name);
    if (!ok) ++failed_cases;
  }
  std::printf("%d cases, %d checks, %d failed checks, %d failed cases\n", count(), checks(), failures(), failed_cases);
  return failed_cases ? 1 : 0;
}
}  // namespace nftest

#define NF_CAT2(a, b) a##b
#define NF_CAT(a, b) NF_CAT2(a, b)
#define TEST(name)                                              \
  static void name();                                           \
  static nftest::Reg NF_CAT(reg_, name)(#name, &name);          \
  static void name()
#define CHECK(cond)                                                                         \
  do {                                                                                      \
    ++nftest::checks();                                                                     \
    if (!(cond)) {                                                                          \
      ++nftest::failures();                                                                 \
      std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);                \
    }                                                                                       \
  } while (0)
#define CHECK_NEAR(a, b, tol)                                                               \
  do {                                                                                      \
    ++nftest::checks();                                                                     \
    double va_ = static_cast<double>(a), vb_ = static_cast<double>(b);                      \
    if (!(std::fabs(va_ - vb_) <= static_cast<double>(tol))) {                              \
      ++nftest::failures();                                                                 \
      std::printf("  %s:%d: CHECK_NEAR(%s, %s) failed: %.9g vs %.9g (tol %.3g)\n", __FILE__, \
                  __LINE__, #a, #b, va_, vb_, static_cast<double>(tol));                    \
    }                                                                                       \
  } while (0)
#define CHECK_EQ(a, b)                                                                      \
  do {                                                                                      \
    ++nftest::checks();                                                                     \
    long long va_ = static_cast<long long>(a), vb_ = static_cast<long long>(b);             \
    if (va_ != vb_) {                                                                       \
      ++nftest::failures();                                                                 \
      std::printf("  %s:%d: CHECK_EQ(%s, %s) failed: %lld vs %lld\n", __FILE__, __LINE__,   \
                  #a, #b, va_, vb_);                                                        \
    }                                                                                       \
  } while (0)
#define NFTEST_MAIN() \
  int main() { return nftest::run_all(); }
