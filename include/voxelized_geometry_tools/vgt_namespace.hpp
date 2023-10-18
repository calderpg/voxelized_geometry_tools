#pragma once

/** Downstream projects can adjust these macros to tweak the project namespace.

When set, they should refer to a C++ inline namespace:
 https://en.cppreference.com/w/cpp/language/namespace#Inline_namespaces

The inline namespace provides symbol versioning to allow multiple copies of
common_robotics_utilities to be linked into the same image. In many cases,
the namespace will also be marked hidden so that linker symbols are private.

Example:
  #define VGT_NAMESPACE_BEGIN \
      inline namespace v1 __attribute__ ((visibility ("hidden"))) {
  #define VGT_NAMESPACE_END }
*/

#ifndef VGT_NAMESPACE_BEGIN
# define VGT_NAMESPACE_BEGIN inline namespace v1 {
#endif

#ifndef VGT_NAMESPACE_END
# define VGT_NAMESPACE_END }
#endif
