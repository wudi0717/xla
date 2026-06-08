/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/primitive_util.h"

#include <limits>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace primitive_util {
namespace {

const char* PrimitiveTypeNameNoReflection(PrimitiveType type) {
  switch (type) {
    case PRIMITIVE_TYPE_INVALID:
      return "primitive_type_invalid";
    case PRED:
      return "pred";
    case S4:
      return "s4";
    case S8:
      return "s8";
    case S16:
      return "s16";
    case S32:
      return "s32";
    case S64:
      return "s64";
    case U4:
      return "u4";
    case U8:
      return "u8";
    case U16:
      return "u16";
    case U32:
      return "u32";
    case U64:
      return "u64";
    case F16:
      return "f16";
    case F32:
      return "f32";
    case BF16:
      return "bf16";
    case F64:
      return "f64";
    case F8E5M2:
      return "f8e5m2";
    case F8E4M3FN:
      return "f8e4m3fn";
    case F8E4M3B11FNUZ:
      return "f8e4m3b11fnuz";
    case F8E5M2FNUZ:
      return "f8e5m2fnuz";
    case F8E4M3FNUZ:
      return "f8e4m3fnuz";
    case C64:
      return "c64";
    case C128:
      return "c128";
    case TUPLE:
      return "tuple";
    case OPAQUE_TYPE:
      return "opaque";
    case TOKEN:
      return "token";
    case PrimitiveType_INT_MIN_SENTINEL_DO_NOT_USE_:
    case PrimitiveType_INT_MAX_SENTINEL_DO_NOT_USE_:
      return "";
  }
  return "";
}

}  // namespace

int SignificandWidth(PrimitiveType type) {
  return PrimitiveTypeSwitch<int>(
      [&](auto constant_type) -> int {
        if constexpr (IsFloatingPointType(constant_type)) {
          return std::numeric_limits<NativeTypeOf<constant_type>>::digits;
        }
        LOG(FATAL) << "Not a floating data type " << type;
      },
      type);
}

int ExponentWidth(PrimitiveType type) {
  // Per the IEEE-754 standard: a floating point type is stored as a sign bit, a
  // biased exponent and a trailing significand field.
  int total_bit_width = BitWidth(type);
  // This field contains all bits in the significand other than the leading
  // digit which is implied by the exponent.
  int trailing_significand_field_width = SignificandWidth(type) - 1;
  // The sign is encoded with a single bit.
  int kSignBitWidth = 1;
  // The remaining bits are used for encoding the biased exponent.
  return total_bit_width - (trailing_significand_field_width + kSignBitWidth);
}

int UnderflowExponent(PrimitiveType type) {
  // |std::numeric_limits<float>::min_exponent| is defined as: "minimum negative
  // integer such that radix raised to the power one less than that integer is a
  // normalized floating-point number." as such it does not actually yield the
  // minimum exponent but one above the minimum exponent that a normalized
  // number can have.
  return PrimitiveTypeSwitch<int>(
      [&](auto constant_type) -> int {
        if constexpr (IsFloatingPointType(constant_type)) {
          return std::numeric_limits<NativeTypeOf<constant_type>>::min_exponent;
        }
        LOG(FATAL) << "Not a floating data type " << type;
      },
      type);
}

int OverflowExponent(PrimitiveType type) {
  // |std::numeric_limits<float>::max_exponent| is defined as: "Maximum positive
  // integer such that radix raised to the power one less than that integer is a
  // representable finite floating-point number." as such it does not actually
  // yield the maximum exponent but the exponent of the first integer which
  // overflows.
  return PrimitiveTypeSwitch<int>(
      [&](auto constant_type) -> int {
        if constexpr (IsFloatingPointType(constant_type)) {
          return std::numeric_limits<NativeTypeOf<constant_type>>::max_exponent;
        }
        LOG(FATAL) << "Not a floating data type " << type;
      },
      type);
}

int ExponentBias(PrimitiveType type) {
  return (1 - UnderflowExponent(type)) + 1;
}

bool HasInfinity(PrimitiveType type) {
  return PrimitiveTypeSwitch<bool>(
      [&](auto constant_type) -> bool {
        if constexpr (IsFloatingPointType(constant_type)) {
          return std::numeric_limits<NativeTypeOf<constant_type>>::has_infinity;
        }
        return false;
      },
      type);
}

xla::PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    case 4:
      return xla::S4;
    case 8:
      return xla::S8;
    case 16:
      return xla::S16;
    case 32:
      return xla::S32;
    case 64:
      return xla::S64;
    default:
      return xla::PRIMITIVE_TYPE_INVALID;
  }
}

// Class to memoize the lowercase name for all PrimitiveType values "p".
//
// xla::OPAQUE_TYPE canonically maps to the string "opaque" -- the only reason
// it's called OPAQUE_TYPE is to avoid clashing with a windows.h macro.
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (i == static_cast<int>(OPAQUE_TYPE)) {
        lowercase_name_[i] = "opaque";
      } else if (PrimitiveType_IsValid(i)) {
        lowercase_name_[i] =
            PrimitiveTypeNameNoReflection(static_cast<PrimitiveType>(i));
      }
    }
  }
  const std::string& LowercaseName(PrimitiveType t) {
    CHECK_LT(t, PrimitiveType_ARRAYSIZE);
    return lowercase_name_[static_cast<int>(t)];
  }

 private:
  std::string lowercase_name_[PrimitiveType_ARRAYSIZE];
};

const std::string& LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}

namespace {

// Returns a map from lower-case primitive type name to primitive type.
//
// Due to Postel's Law considerations, both "opaque" and "opaque_type" map to
// the xla::OPAQUE_TYPE enumerator.
const absl::flat_hash_map<std::string, PrimitiveType>&
GetPrimitiveTypeStringMap() {
  static absl::flat_hash_map<std::string, PrimitiveType>* name_to_type = [] {
    static auto* map = new absl::flat_hash_map<std::string, PrimitiveType>;
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (PrimitiveType_IsValid(i) && i != PRIMITIVE_TYPE_INVALID) {
        auto value = static_cast<PrimitiveType>(i);
        (*map)[LowercasePrimitiveTypeName(value)] = value;
      }
    }
    (*map)["opaque"] = OPAQUE_TYPE;
    return map;
  }();
  return *name_to_type;
}

}  // namespace

StatusOr<PrimitiveType> StringToPrimitiveType(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  if (found == map.end()) {
    return InvalidArgument("Invalid element type string: \"%s\".", name);
  }
  return found->second;
}

bool IsPrimitiveTypeName(absl::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  return found != map.end();
}

}  // namespace primitive_util
}  // namespace xla
