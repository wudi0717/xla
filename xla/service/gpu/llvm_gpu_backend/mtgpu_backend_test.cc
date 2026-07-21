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

#include <memory>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "xla/test.h"

namespace xla {
namespace gpu {
namespace mtgpu {
namespace {

TEST(MTGPUBackendTest, CompileToHsacoProducesBinary) {
  DebugOptions debug_options;
  const std::string ir_text = R"(
target triple = "mtgpu-mt-musa"

define void @k() {
entry:
  ret void
}
)";
  auto result = CompileLlvmIrToHsacoForTest(ir_text, debug_options);

  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->empty());
}

TEST(MTGPUBackendTest, RewritesNearbyIntToMusaRoundEven) {
  llvm::LLVMContext context;
  llvm::Module module("nearbyint", context);
  llvm::IRBuilder<> builder(context);
  llvm::Function* function = llvm::Function::Create(
      llvm::FunctionType::get(builder.getVoidTy(), /*isVarArg=*/false),
      llvm::GlobalValue::ExternalLinkage, "kernel", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));

  llvm::Function* nearbyint_f32 = llvm::Intrinsic::getDeclaration(
      &module, llvm::Intrinsic::nearbyint, {builder.getFloatTy()});
  llvm::Function* nearbyint_f64 = llvm::Intrinsic::getDeclaration(
      &module, llvm::Intrinsic::nearbyint, {builder.getDoubleTy()});
  llvm::CallInst* call_f32 = builder.CreateCall(
      nearbyint_f32, {llvm::ConstantFP::get(builder.getFloatTy(), 1.5)});
  llvm::CallInst* call_f64 = builder.CreateCall(
      nearbyint_f64, {llvm::ConstantFP::get(builder.getDoubleTy(), 1.5)});
  builder.CreateRetVoid();

  ConvertNvvmToMusaIntrinsicsForTest(&module);

  EXPECT_EQ(call_f32->getCalledFunction()->getName(), "__mt_roundeven_f32");
  EXPECT_EQ(call_f64->getCalledFunction()->getName(), "__mt_roundeven_f64");
}

TEST(MTGPUBackendTest, DirectMusaPowRewriteIsOptIn) {
  auto build_module = [](llvm::LLVMContext& context) {
    auto module = std::make_unique<llvm::Module>("pow", context);
    llvm::IRBuilder<> builder(context);
    llvm::Function* function = llvm::Function::Create(
        llvm::FunctionType::get(builder.getVoidTy(), /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "kernel", *module);
    builder.SetInsertPoint(
        llvm::BasicBlock::Create(context, "entry", function));
    llvm::Function* pow_f32 = llvm::Function::Create(
        llvm::FunctionType::get(
            builder.getFloatTy(),
            {builder.getFloatTy(), builder.getFloatTy()},
            /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "__nv_powf", *module);
    builder.CreateCall(
        pow_f32, {llvm::ConstantFP::get(builder.getFloatTy(), 2.0),
                  llvm::ConstantFP::get(builder.getFloatTy(), 3.0)});
    builder.CreateRetVoid();
    return module;
  };

  llvm::LLVMContext disabled_context;
  std::unique_ptr<llvm::Module> disabled_module =
      build_module(disabled_context);
  ConvertNvvmToMusaIntrinsicsForTest(disabled_module.get(),
                                     /*enable_direct_mt_pow=*/false);
  EXPECT_NE(disabled_module->getFunction("__nv_powf"), nullptr);
  EXPECT_EQ(disabled_module->getFunction("__mt_pow_f32"), nullptr);

  llvm::LLVMContext enabled_context;
  std::unique_ptr<llvm::Module> enabled_module = build_module(enabled_context);
  ConvertNvvmToMusaIntrinsicsForTest(enabled_module.get(),
                                     /*enable_direct_mt_pow=*/true);
  EXPECT_NE(enabled_module->getFunction("__mt_pow_f32"), nullptr);
  llvm::CallInst* rewritten_call = nullptr;
  for (llvm::Instruction& instruction :
       enabled_module->getFunction("kernel")->getEntryBlock()) {
    if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
      rewritten_call = call;
      break;
    }
  }
  ASSERT_NE(rewritten_call, nullptr);
  EXPECT_EQ(rewritten_call->getCalledFunction()->getName(), "__mt_pow_f32");
}

}  // namespace
}  // namespace mtgpu
}  // namespace gpu
}  // namespace xla
