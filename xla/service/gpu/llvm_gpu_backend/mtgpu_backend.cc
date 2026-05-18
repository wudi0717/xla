/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/llvm_gpu_backend/mtgpu_backend.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Verifier.h"
#include "tsl/platform/env.h"
#include "tsl/platform/musa_musdl_path.h"
#include "tsl/platform/path.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace mtgpu {
namespace {

struct Rule {
  const char* old_name;
  const char* new_name;
  int type;
};

#include "musa_intrinsic.def"

struct MusaToolchainLayout {
  std::string musa_root;
  std::string musdl_root;
  std::string llvm_link;
  std::string opt;
  std::string llc;
  std::string ld_lld;
  std::string libdevice;
  std::string libdevice_mthg;
};

absl::Status MissingInstallArtifact(const char* artifact_name,
                                    const std::string& artifact_path) {
  return absl::FailedPreconditionError(absl::StrCat(
      "MTGPU backend code generation is not wired yet: missing ",
      artifact_name, " at ", artifact_path,
      ". Phase I expects the MUSA SDK layout from openxla/install.sh "
      "(MUSA_HOME=/usr/local/musa)."));
}

std::string QuoteForShell(const std::string& path) {
  return absl::StrCat("\"", path, "\"");
}

std::string GetConfiguredDumpDir() {
  const char* dump_dir = std::getenv("MTGPU_DUMP_DIR");
  return dump_dir == nullptr ? "" : dump_dir;
}

std::string Basename(const std::string& path) {
  const size_t pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string ReadSmallTextFile(const std::string& path) {
  std::string content;
  if (!tsl::ReadFileToString(tsl::Env::Default(), path, &content).ok()) {
    return "<unable to read log>";
  }
  constexpr size_t kMaxLogBytes = 4096;
  if (content.size() > kMaxLogBytes) {
    content.resize(kMaxLogBytes);
    content.append("\n... <truncated>");
  }
  return content.empty() ? "<empty>" : content;
}

absl::Status RunToolCommand(const std::string& command, const char* tool_name,
                            const std::string& log_base_path) {
  const std::string stdout_path = log_base_path + ".stdout.log";
  const std::string stderr_path = log_base_path + ".stderr.log";
  const std::string redirected_command =
      absl::StrCat(command, " > ", QuoteForShell(stdout_path), " 2> ",
                   QuoteForShell(stderr_path));
  if (std::system(redirected_command.c_str()) != 0) {
    return xla::Internal(
        "MTGPU backend failed while running %s.\nCommand: %s\nStdout log: "
        "%s\nStderr log: %s\nStderr:\n%s",
        tool_name, command, stdout_path, stderr_path,
        ReadSmallTextFile(stderr_path));
  }
  return absl::OkStatus();
}

std::string ChooseLibdevicePath(const std::string& musdl_root, tsl::Env* env) {
  for (const char* candidate : {"libdevice.bc", "libdevice.31.bc"}) {
    const std::string path = tsl::io::JoinPath(musdl_root, candidate);
    if (env->FileExists(path).ok()) {
      return path;
    }
  }
  return "";
}

absl::StatusOr<MusaToolchainLayout> ResolveMusaToolchainLayout() {
  MusaToolchainLayout layout;
  layout.musa_root = tsl::MusaRoot();
  layout.musdl_root = tsl::MusdlRoot();

  if (layout.musa_root.empty()) {
    return absl::FailedPreconditionError(
        "MTGPU backend code generation is not wired yet: MUSA root is empty. "
        "Set MUSA_PATH or MUSA_HOME; Phase I defaults to "
        "/usr/local/musa like openxla/install.sh.");
  }

  auto* env = tsl::Env::Default();
  if (!env->FileExists(layout.musa_root).ok()) {
    return MissingInstallArtifact("MUSA SDK root", layout.musa_root);
  }
  if (!env->FileExists(layout.musdl_root).ok()) {
    return MissingInstallArtifact("MUSA device-lib root", layout.musdl_root);
  }

  for (const auto& tool : std::array<std::pair<const char*, std::string*>, 4>{
           std::pair<const char*, std::string*>("llvm-link", &layout.llvm_link),
           std::pair<const char*, std::string*>("opt", &layout.opt),
           std::pair<const char*, std::string*>("llc", &layout.llc),
           std::pair<const char*, std::string*>("ld.lld", &layout.ld_lld)}) {
    *tool.second = tsl::io::JoinPath(layout.musa_root, "bin", tool.first);
    if (!env->FileExists(*tool.second).ok()) {
      return MissingInstallArtifact(tool.first, *tool.second);
    }
  }

  layout.libdevice = ChooseLibdevicePath(layout.musdl_root, env);
  if (layout.libdevice.empty()) {
    return MissingInstallArtifact("libdevice.bc",
                                  tsl::io::JoinPath(layout.musdl_root,
                                                    "libdevice.bc"));
  }
  layout.libdevice_mthg = tsl::io::JoinPath(layout.musdl_root, "libdevice.mthg.bc");
  if (!env->FileExists(layout.libdevice_mthg).ok()) {
    return MissingInstallArtifact("libdevice.mthg.bc", layout.libdevice_mthg);
  }

  return layout;
}

absl::StatusOr<std::string> MakeTempBasePath() {
  std::string base_path;
  if (!tsl::Env::Default()->LocalTempFilename(&base_path)) {
    return xla::Internal("Unable to create a temporary file path for MTGPU "
                         "backend code generation.");
  }
  const std::string dump_dir = GetConfiguredDumpDir();
  if (!dump_dir.empty() && tsl::Env::Default()->IsDirectory(dump_dir).ok()) {
    return tsl::io::JoinPath(dump_dir, Basename(base_path));
  }
  return base_path;
}

absl::StatusOr<std::vector<uint8_t>> ReadBinaryFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return xla::Internal("Unable to open generated MTGPU binary: %s", path);
  }
  std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0 && !input.read(reinterpret_cast<char*>(bytes.data()), size)) {
    return xla::Internal("Unable to read generated MTGPU binary: %s", path);
  }
  return bytes;
}

std::string MakeIrCompatibleWithMusaLlvm(std::string ir_text) {
  // Newer LLVM prints memory effects as memory(...). The MUSA LLVM tools used
  // by TF 2.15-era builds do not parse that spelling, and these attributes are
  // only optimizer hints for this backend path.
  return std::regex_replace(ir_text, std::regex(R"([[:space:]]+memory\([^)]*\))"),
                            "");
}

void PreserveGlobalVars(llvm::Module& module) {
  llvm::SmallPtrSet<llvm::Constant*, 16> keep;
  auto maybe_keep_constant = [&](llvm::Constant* constant) {
    if (auto* global_value = llvm::dyn_cast<llvm::GlobalValue>(constant)) {
      if (!global_value->hasName()) {
        return;
      }
    }
    keep.insert(constant);
  };

  if (llvm::GlobalVariable* llvm_used = module.getGlobalVariable("llvm.used")) {
    if (auto* constants =
            llvm::dyn_cast<llvm::ConstantArray>(llvm_used->getInitializer())) {
      for (llvm::Value* operand : constants->operands()) {
        llvm::Value* value =
            llvm::cast<llvm::Constant>(operand)->getOperand(0);
        maybe_keep_constant(llvm::cast<llvm::Constant>(value));
      }
    }
  }

  for (llvm::GlobalVariable& global : module.globals()) {
    if (!global.isDeclaration() && global.hasName()) {
      keep.insert(&global);
    }
  }

  if (keep.empty()) {
    return;
  }

  llvm::LLVMContext& context = module.getContext();
  llvm::Type* ptr_ty = llvm::PointerType::get(context, 0);
  llvm::ArrayType* array_ty = llvm::ArrayType::get(ptr_ty, keep.size());

  llvm::SmallVector<llvm::Constant*, 64> elements;
  for (llvm::Constant* constant : keep) {
    elements.push_back(llvm::ConstantExpr::getPointerCast(constant, ptr_ty));
  }

  llvm::Constant* initializer = llvm::ConstantArray::get(array_ty, elements);
  llvm::GlobalVariable* llvm_used = module.getGlobalVariable("llvm.used");
  if (!llvm_used) {
    llvm_used = new llvm::GlobalVariable(
        module, array_ty, false, llvm::GlobalValue::AppendingLinkage,
        initializer, "llvm.used");
    llvm_used->setSection("llvm.metadata");
  } else {
    llvm_used->setInitializer(initializer);
  }
}

void ConvertNvvmToMusaIntrinsics(llvm::Module& module) {
  struct RewriteRecord {
    llvm::CallInst* call;
    const Rule* rule;
  };
  std::vector<RewriteRecord> worklist;

  for (llvm::Function& function : module) {
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& instruction : block) {
        auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction);
        if (call == nullptr) {
          continue;
        }
        llvm::Function* callee = call->getCalledFunction();
        if (callee == nullptr) {
          continue;
        }
        for (const Rule& rule : rules) {
          if (callee->getName() == rule.old_name) {
            worklist.push_back({call, &rule});
            break;
          }
        }
      }
    }
  }

  for (const RewriteRecord& record : worklist) {
    llvm::CallInst* call = record.call;
    const Rule* rule = record.rule;
    llvm::Function* old_function = call->getCalledFunction();
    llvm::Function* new_function = module.getFunction(rule->new_name);
    if (new_function == nullptr) {
      switch (rule->type) {
        case 2:
        case 3: {
          llvm::LLVMContext& context = module.getContext();
          llvm::Type* int32_ty = llvm::Type::getInt32Ty(context);
          llvm::PointerType* ptr_as5_ty = llvm::PointerType::get(context, 5);
          llvm::FunctionType* function_type = llvm::FunctionType::get(
              int32_ty, {int32_ty, int32_ty, int32_ty, ptr_as5_ty}, false);
          new_function = llvm::Function::Create(
              function_type, llvm::GlobalValue::ExternalLinkage,
              rule->new_name, &module);

          llvm::AttributeList attributes;
          llvm::AttrBuilder function_attrs(context);
          function_attrs.addAttribute(llvm::Attribute::Convergent);
          function_attrs.addAttribute(llvm::Attribute::NoUnwind);
          function_attrs.addMemoryAttr(llvm::MemoryEffects::writeOnly());
          attributes = attributes.addFnAttributes(context, function_attrs);
          new_function->setAttributes(attributes);
          new_function->setCallingConv(old_function->getCallingConv());
          break;
        }
        case 1: {
          llvm::FunctionType* function_type =
              llvm::FunctionType::get(llvm::Type::getVoidTy(module.getContext()),
                                      /*isVarArg=*/false);
          new_function = llvm::Function::Create(
              function_type, llvm::GlobalValue::ExternalLinkage,
              rule->new_name, &module);
          new_function->setAttributes(old_function->getAttributes());
          new_function->setCallingConv(old_function->getCallingConv());
          break;
        }
        default: {
          new_function = llvm::Function::Create(
              old_function->getFunctionType(),
              llvm::GlobalValue::ExternalLinkage, rule->new_name, &module);
          new_function->setAttributes(old_function->getAttributes());
          new_function->setCallingConv(old_function->getCallingConv());
          break;
        }
      }
    }

    switch (rule->type) {
      case 3: {
        llvm::LLVMContext& context = module.getContext();
        llvm::Type* int32_ty = llvm::Type::getInt32Ty(context);
        llvm::Type* float_ty = llvm::Type::getFloatTy(context);
        llvm::PointerType* ptr_as5_ty = llvm::PointerType::get(context, 5);
        llvm::FunctionType* function_type = llvm::FunctionType::get(
            int32_ty, {int32_ty, int32_ty, int32_ty, ptr_as5_ty}, false);
        llvm::IRBuilder<> builder(call);
        llvm::Value* value = call->getArgOperand(1);
        llvm::Value* offset = call->getArgOperand(2);
        llvm::Value* mask = call->getArgOperand(3);
        llvm::Value* null_ptr = llvm::ConstantPointerNull::get(ptr_as5_ty);
        llvm::Value* value_as_int = builder.CreateBitCast(value, int32_ty);
        llvm::CallInst* new_call = llvm::CallInst::Create(
            function_type, new_function, {value_as_int, offset, mask, null_ptr},
            "", call);
        new_call->setCallingConv(call->getCallingConv());
        new_call->setAttributes(call->getAttributes());
        llvm::Value* result =
            builder.CreateBitCast(new_call, float_ty, call->getName());
        call->replaceAllUsesWith(result);
        call->eraseFromParent();
        break;
      }
      case 2: {
        llvm::LLVMContext& context = module.getContext();
        llvm::Type* int32_ty = llvm::Type::getInt32Ty(context);
        llvm::PointerType* ptr_as5_ty = llvm::PointerType::get(context, 5);
        llvm::FunctionType* function_type = llvm::FunctionType::get(
            int32_ty, {int32_ty, int32_ty, int32_ty, ptr_as5_ty}, false);
        llvm::IRBuilder<> builder(call);
        llvm::Value* value = call->getArgOperand(1);
        llvm::Value* offset = call->getArgOperand(2);
        llvm::Value* mask = call->getArgOperand(3);
        llvm::Value* null_ptr = llvm::ConstantPointerNull::get(ptr_as5_ty);
        llvm::CallInst* new_call = builder.CreateCall(
            function_type, new_function, {value, offset, mask, null_ptr});
        new_call->takeName(call);
        new_call->setCallingConv(call->getCallingConv());
        new_call->setAttributes(call->getAttributes());
        if (auto debug_loc = call->getDebugLoc()) {
          new_call->setDebugLoc(debug_loc);
        }
        call->replaceAllUsesWith(new_call);
        call->eraseFromParent();
        break;
      }
      case 1: {
        llvm::IRBuilder<> builder(call);
        llvm::CallInst* new_call = builder.CreateCall(new_function, {});
        new_call->takeName(call);
        new_call->setCallingConv(call->getCallingConv());
        new_call->setAttributes(call->getAttributes());
        if (auto debug_loc = call->getDebugLoc()) {
          new_call->setDebugLoc(debug_loc);
        }
        call->replaceAllUsesWith(new_call);
        call->eraseFromParent();
        break;
      }
      default:
        call->setCalledFunction(new_function);
        break;
    }
  }

  for (const Rule& rule : rules) {
    if (llvm::Function* function = module.getFunction(rule.old_name)) {
      if (function->use_empty()) {
        function->eraseFromParent();
      }
    }
  }

  CHECK(!llvm::verifyModule(module, &llvm::errs()));
}

absl::StatusOr<std::vector<uint8_t>> CompileLlvmIrToHsacoImpl(
    const std::string& ir_text, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(MusaToolchainLayout layout, ResolveMusaToolchainLayout());
  TF_ASSIGN_OR_RETURN(std::string temp_base, MakeTempBasePath());

  const std::string ll_path = temp_base + ".ll";
  const std::string linked_ll_path = temp_base + "_linked.ll";
  const std::string opt_ll_path = temp_base + "_opt.ll";
  const std::string obj_path = temp_base + ".o";
  const std::string mubin_path = temp_base + ".mubin";

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), ll_path, ir_text));

  TF_RETURN_IF_ERROR(RunToolCommand(
      absl::StrCat(QuoteForShell(layout.llvm_link), " -opaque-pointers ",
                   QuoteForShell(ll_path), " ",
                   QuoteForShell(layout.libdevice), " ",
                   QuoteForShell(layout.libdevice_mthg),
                   " --only-needed -S -o ", QuoteForShell(linked_ll_path)),
      "llvm-link", temp_base + ".llvm-link"));

  const char* opt_level = debug_options.xla_gpu_disable_gpuasm_optimizations()
                              ? "-O0"
                              : "-O2";
  TF_RETURN_IF_ERROR(RunToolCommand(
      absl::StrCat(QuoteForShell(layout.opt), " -opaque-pointers ", opt_level,
                   " ", QuoteForShell(linked_ll_path), " -S -o ",
                   QuoteForShell(opt_ll_path)),
      "opt", temp_base + ".opt"));

  TF_RETURN_IF_ERROR(RunToolCommand(
      absl::StrCat(QuoteForShell(layout.llc), " -opaque-pointers ",
                   QuoteForShell(opt_ll_path),
                   " -march=mtgpu -mcpu=mp_31 -filetype=obj -o ",
                   QuoteForShell(obj_path)),
      "llc", temp_base + ".llc"));

  TF_RETURN_IF_ERROR(RunToolCommand(
      absl::StrCat(QuoteForShell(layout.ld_lld), " -flavor gnu -shared ",
                   QuoteForShell(obj_path), " -o ", QuoteForShell(mubin_path)),
      "ld.lld", temp_base + ".ld.lld"));

  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> bytes, ReadBinaryFile(mubin_path));

  std::remove(ll_path.c_str());
  std::remove(linked_ll_path.c_str());
  std::remove(opt_ll_path.c_str());
  std::remove(obj_path.c_str());
  std::remove(mubin_path.c_str());

  return bytes;
}

}  // namespace

StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module,
    stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key) {
  (void)gpu_version;
  (void)module_config_cache_key;
  if (module->getTargetTriple().empty()) {
    module->setTargetTriple("mtgpu-mt-musa");
  }
  PreserveGlobalVars(*module);
  ConvertNvvmToMusaIntrinsics(*module);

  std::string ir_text;
  llvm::raw_string_ostream ir_stream(ir_text);
  module->print(ir_stream, nullptr);
  ir_stream.flush();
  ir_text = std::regex_replace(ir_text, std::regex("ptx_kernel"),
                               "mtgpu_kernel");
  ir_text = MakeIrCompatibleWithMusaLlvm(std::move(ir_text));
  return CompileLlvmIrToHsacoImpl(ir_text, debug_options);
}

StatusOr<std::vector<uint8_t>> CompileLlvmIrToHsacoForTest(
    const std::string& ir_text, const DebugOptions& debug_options) {
  return CompileLlvmIrToHsacoImpl(ir_text, debug_options);
}

}  // namespace mtgpu
}  // namespace gpu
}  // namespace xla
