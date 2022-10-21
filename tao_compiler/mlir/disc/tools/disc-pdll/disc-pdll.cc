// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- mlir-pdll.cpp - MLIR PDLL frontend -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <set>
#include <string>
#include <unordered_map>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/CPPGen.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

const std::string kDefaultHelperFunctionDeclarations = R"pdll(
  Rewrite PackValue_1(tag : Attr, v0 : Value) -> ValueRange;
  Rewrite PackValue_2(tag : Attr, v0 : Value, v1 : Value) -> ValueRange;
  Rewrite UnpackValue_1(v : ValueRange) -> (Value);
  Rewrite UnpackValue_2(v : ValueRange) -> (Value, Value);
  Rewrite CreateCustomCall(tag : Attr, inputs : ValueRange, outputs : ValueRange) -> (op: Op, new_outputs : ValueRange);
  Rewrite SetAttr(op : Op, key : Attr, value : Attr);
  Rewrite SetCustomAttr(op : Op, key : Attr, value : Attr);
)pdll";

// Combines the `chunkBuffer` with some pre-defined helper function prototypes.
// The result is written to a newly allocated buffer which will be returned.
std::unique_ptr<llvm::MemoryBuffer> addPredefinedPrototypes(
    std::unique_ptr<llvm::MemoryBuffer>& chunkBuffer) {
  size_t bytes = kDefaultHelperFunctionDeclarations.size() +
                 chunkBuffer->getBufferSize() + 1;
  auto combinedBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    auto& c = combinedBuffer->getBufferStart()[i];
    if (i < kDefaultHelperFunctionDeclarations.size()) {
      c = kDefaultHelperFunctionDeclarations[i];
    } else if (i < bytes - 1) {
      c = chunkBuffer
              ->getBufferStart()[i - kDefaultHelperFunctionDeclarations.size()];
    } else {
      c = static_cast<char>(0);
    }
  }
  return combinedBuffer;
}

static OwningOpRef<ModuleOp> compilePDLL(
    MLIRContext& mlirContext, std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
    std::vector<std::string>& includeDirs) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(addPredefinedPrototypes(chunkBuffer), SMLoc());

  ods::Context odsContext;
  ast::Context astContext(odsContext);
  FailureOr<ast::Module*> module = parsePDLLAST(astContext, sourceMgr, false);
  if (failed(module)) return nullptr;

  auto pdlModule =
      codegenPDLLToMLIR(&mlirContext, astContext, sourceMgr, **module);
  if (pdlModule) {
    SmallVector<pdl::RewriteOp> ops;
    pdlModule->walk([&](pdl::RewriteOp op) { ops.push_back(op); });
    for (pdl::RewriteOp op : ops) {
      OpBuilder b(op);
      auto newOp =
          b.create<pdl::RewriteOp>(op.getLoc(), nullptr, nullptr, ValueRange{});
      newOp.getBodyRegion().getBlocks().splice(
          newOp.getBodyRegion().getBlocks().begin(),
          op.getBodyRegion().getBlocks());
      op->erase();
    }
  }
  return pdlModule;
}

void LoadDependentDialects(mlir::MLIRContext& context) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  registry.insert<mlir::TF::TensorFlowDialect>();

  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);
}

using PDLRewriteFunction =
    std::function<void(PatternRewriter&, PDLResultList&, ArrayRef<PDLValue>)>;

SmallVector<Value>& getThreadLocalValueRangeStorage(StringRef tag) {
  thread_local static auto valueRangeMap =
      new std::unordered_map<std::string, SmallVector<Value>>{};
  return (*valueRangeMap)[tag.str()];
}

template <int ExpectedNum>
static void packValues(PatternRewriter& rewriter, PDLResultList& results,
                       ArrayRef<PDLValue> values) {
  int numValueInputs = static_cast<int>(values.size()) - 1;
  if (numValueInputs != ExpectedNum) {
    llvm::errs() << "PackValue expects " << ExpectedNum << " values but got "
                 << numValueInputs << "\n";
    return;
  }

  if (values.size() <= 1) {
    results.push_back(ValueRange{});
    return;
  }

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = getThreadLocalValueRangeStorage(tag);
  vs.clear();
  for (auto& v : values.drop_front()) {
    vs.push_back(v.cast<Value>());
  }
  results.push_back(ValueRange{vs});
}

template <int ExpectedNum>
static void unpackValues(PatternRewriter& rewriter, PDLResultList& results,
                         ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  int numResults = 0;
  for (Value v : values[0].cast<ValueRange>()) {
    results.push_back(v);
    ++numResults;
  }

  if (numResults != ExpectedNum) {
    llvm::errs() << "PackValue expects " << ExpectedNum << " values but got "
                 << numResults << "\n";
    return;
  }
}

static void createCustomCall(PatternRewriter& rewriter, PDLResultList& results,
                             ArrayRef<PDLValue> values) {
  assert(values.size() == 3);

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = getThreadLocalValueRangeStorage(tag);
  vs.clear();
  auto inputs = values[1].cast<ValueRange>();
  auto outputs = values[2].cast<ValueRange>();

  SmallVector<Type> outputTypes;
  for (Value v : outputs) outputTypes.push_back(v.getType());
  assert(outputTypes.size() > 0);
  Operation* op = rewriter.create<mhlo_disc::CustomCallOp>(
      (*outputs.begin()).getLoc(), outputTypes, inputs, "", false,
      StringAttr::get((*outputs.begin()).getContext(), ""));

  for (Value out : op->getResults()) vs.push_back(out);

  results.push_back(op);
  results.push_back(ValueRange(vs));
}

static LogicalResult applyPatterns(OwningOpRef<ModuleOp> pdlModule,
                                   ModuleOp payloadModule) {
  PDLPatternModule pdlPatterns(std::move(pdlModule));
  pdlPatterns.registerRewriteFunction(
      "SetAttr", [](PatternRewriter& rewriter, Operation* op, Attribute keyAttr,
                    Attribute valueAttr) {
        StringRef key = keyAttr.cast<StringAttr>().getValue();
        op->setAttr(key, valueAttr);
      });
  pdlPatterns.registerRewriteFunction(
      "SetCustomAttr", [](PatternRewriter& rewriter, Operation* op,
                          Attribute keyAttr, Attribute valueAttr) {
        auto customAttrs = op->getAttrOfType<DictionaryAttr>("custom_attrs");
        if (!customAttrs) {
          customAttrs = DictionaryAttr::get(op->getContext(), {});
        }
        StringRef key = keyAttr.cast<StringAttr>().getValue();
        SmallVector<NamedAttribute> newAttrs;
        for (auto& attr : customAttrs) {
          if (attr.getName().getValue() == key) continue;
          newAttrs.push_back(attr);
        }
        newAttrs.push_back(
            NamedAttribute(keyAttr.cast<StringAttr>(), valueAttr));
        auto newCustomAttrs = DictionaryAttr::get(op->getContext(), newAttrs);
        op->setAttr("custom_attrs", newCustomAttrs);
      });
  pdlPatterns.registerRewriteFunction("CreateCustomCall", createCustomCall);
  pdlPatterns.registerRewriteFunction("PackValue_1", packValues<1>);
  pdlPatterns.registerRewriteFunction("PackValue_2", packValues<2>);
  pdlPatterns.registerRewriteFunction("UnpackValue_1", unpackValues<1>);
  pdlPatterns.registerRewriteFunction("UnpackValue_2", unpackValues<2>);

  RewritePatternSet patternList(payloadModule->getContext());
  patternList.add(std::move(pdlPatterns));
  return applyPatternsAndFoldGreedily(payloadModule.getBodyRegion(),
                                      std::move(patternList));
}

static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(StringRef inputFilename,
                                                        MLIRContext* context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return mlir::OwningOpRef<mlir::ModuleOp>(
      parseSourceFile<mlir::ModuleOp>(sourceMgr, context));
}

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> pdllInputFilename(
      "pdl-input", llvm::cl::desc("<input pdll file>"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> payloadInputFilename(
      "payload-input", llvm::cl::desc("<input payload file>"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "output", llvm::cl::desc("Output filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "DISC PDLL Frontend");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> pdllInputFile =
      openInputFile(pdllInputFilename, &errorMessage);
  if (!pdllInputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  MLIRContext mlirContext;
  LoadDependentDialects(mlirContext);
  OwningOpRef<ModuleOp> pdlModule =
      compilePDLL(mlirContext, std::move(pdllInputFile), includeDirs);
  if (!pdlModule) {
    llvm::errs() << "failed to compile the pdll file: " << pdllInputFilename
                 << "\n";
    return 1;
  }
  llvm::errs() << "/////// Parsed PDL module: \n" << pdlModule.get() << "\n\n";

  OwningOpRef<ModuleOp> payloadModule =
      parseMLIRInput(payloadInputFilename, &mlirContext);
  if (!payloadModule) {
    llvm::errs() << "failed to load payload from file: " << payloadInputFilename
                 << "\n";
    return 1;
  }
  llvm::errs() << "/////// Parsed Payload module: \n"
               << payloadModule.get() << "\n\n";

  if (failed(applyPatterns(std::move(pdlModule), payloadModule.get()))) {
    llvm::errs() << "failed to apply patterns\n";
    return 1;
  }
  llvm::errs() << "/////// Rewrited Payload module: \n"
               << payloadModule.get() << "\n\n";

  return 0;
}
