#ifndef HEIR_COYOTE_PYTHON_SCHEDULER_H
#define HEIR_COYOTE_PYTHON_SCHEDULER_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <vector>

#include "GraphUtils.h"
#include "GreedyAlign.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/EquivalenceClasses.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

//===----------------------------------------------------------------------===//
// Python Coyote Scheduler (fallback via subprocess)
//===----------------------------------------------------------------------===//

#define USE_PYTHON_COYOTE 1

#ifdef USE_PYTHON_COYOTE

namespace mlir {
namespace heir {

/// Serialize the MLIR circuit to three-address code, call Python Coyote,
/// and return the resulting Schedule.
///
/// Circuit serialization format (matches Python's Instr):
///   INPUT_GROUP <var1> <var2> ...
///   LOAD <dest_reg> <var_name>
///   OP <dest_reg> <lhs_reg> <rhs_reg> <op_symbol>
///
/// Python returns JSON with lanes[] and alignment[] arrays indexed by register.
///
/// The bridge script (run_coyote_from_circuit.py) must be findable via
/// COYOTE_PYTHON_PATH env var or in the current working directory.
std::optional<Schedule> runPythonScheduler(
    llvm::ArrayRef<Operation*> operations,
    llvm::ArrayRef<llvm::SmallVector<Operation*>> inputGroups) {
  // Locate the bridge script.
  std::string scriptPath;
  if (const char* envPath = std::getenv("COYOTE_PYTHON_PATH")) {
    scriptPath = std::string(envPath) + "/run_coyote_from_circuit.py";
  } else {
    for (const auto& candidate : {
             "run_coyote_from_circuit.py",
             "../run_coyote_from_circuit.py",
             "../../run_coyote_from_circuit.py",
         }) {
      if (std::filesystem::exists(candidate)) {
        scriptPath = candidate;
        break;
      }
    }
  }
  if (scriptPath.empty()) {
    llvm::errs()
        << "  [Python] Bridge script not found. "
        << "Set COYOTE_PYTHON_PATH or ensure run_coyote_from_circuit.py "
        << "is in the working directory.\n";
    return std::nullopt;
  }

  // The bridge script handles venv creation/activation internally,
  // so we just need any python3 to kick it off.
  std::string pythonBin = "python3";

  // --- Helpers ---
  auto isCircuitArith = [](Operation* op) {
    return isa<arith::MulIOp, arith::AddIOp, arith::SubIOp, arith::MulFOp,
               arith::AddFOp, arith::SubFOp>(op);
  };

  // Compute proper row-major flat index from all dimensions of a tensor
  // extract. E.g. tensor<2x3>: extract[1,2] -> flat = 1*3 + 2 = 5
  auto computeFlatIdx = [](tensor::ExtractOp extractOp) -> int64_t {
    auto indices = extractOp.getIndices();
    if (indices.empty()) return 0;
    auto tensorType = cast<RankedTensorType>(extractOp.getTensor().getType());
    auto shape = tensorType.getShape();
    int64_t flatIdx = 0;
    for (unsigned i = 0; i < indices.size(); ++i) {
      auto constOp = indices[i].getDefiningOp<arith::ConstantIndexOp>();
      if (!constOp) continue;
      int64_t stride = 1;
      for (unsigned j = i + 1; j < shape.size(); ++j) stride *= shape[j];
      flatIdx += constOp.value() * stride;
    }
    return flatIdx;
  };

  // --- Build variable names for extract ops (with correct flat index) ---
  llvm::DenseMap<Operation*, std::string> extractVarName;
  for (const auto& group : inputGroups) {
    BlockArgument blockArg;
    for (auto* op : group) {
      if (auto extractOp = dyn_cast<tensor::ExtractOp>(op))
        if (auto ba = dyn_cast<BlockArgument>(extractOp.getTensor())) {
          blockArg = ba;
          break;
        }
    }
    if (!blockArg) continue;
    std::string argName = "arg" + std::to_string(blockArg.getArgNumber());
    for (auto* op : group) {
      if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
        int64_t flatIdx = computeFlatIdx(extractOp);
        extractVarName[op] = argName + ":" + std::to_string(flatIdx);
      }
    }
  }

  // --- Assign registers WITH load replication ---
  // For each use of an extract op in a circuit arith op, create a separate
  // LOAD register.  This matches Python Coyote's allow_replicating='all':
  // each copy can be placed in a different lane, reducing rotations.
  llvm::DenseMap<Operation*, int64_t> opToReg;
  llvm::DenseMap<int64_t, Operation*> regToOp;

  // Track: arith_op -> { operand_index -> replicated load register }
  llvm::DenseMap<Operation*, llvm::SmallDenseMap<unsigned, int64_t, 2>>
      arithOperandReg;

  struct LoadEntry {
    int64_t reg;
    std::string varName;
  };
  llvm::SmallVector<LoadEntry> loadEntries;
  int64_t nextReg = 0;

  // Phase 1: Replicated LOAD registers (one per use of each extract in arith)
  for (auto* op : operations) {
    if (!isa<tensor::ExtractOp>(op) || !extractVarName.count(op)) continue;
    int64_t firstReg = -1;
    for (auto& use : op->getResult(0).getUses()) {
      Operation* user = use.getOwner();
      if (!isCircuitArith(user)) continue;
      int64_t reg = nextReg++;
      loadEntries.push_back({reg, extractVarName[op]});
      regToOp[reg] = op;
      arithOperandReg[user][use.getOperandNumber()] = reg;
      if (firstReg == -1) firstReg = reg;
    }
    if (firstReg != -1)
      opToReg[op] = firstReg;  // canonical register for this extract
  }

  // Phase 2: OP registers (one per arith op)
  for (auto* op : operations) {
    if (!isCircuitArith(op)) continue;
    opToReg[op] = nextReg;
    regToOp[nextReg] = op;
    ++nextReg;
  }

  // Create temp files.
  auto tmpDir = std::filesystem::temp_directory_path();
  auto circuitPath = tmpDir / "coyote_circuit.txt";
  auto resultPath = tmpDir / "coyote_python_result.json";

  // --- Serialize circuit ---
  {
    std::ofstream out(circuitPath);
    if (!out) {
      llvm::errs() << "  [Python] Cannot create temp file\n";
      return std::nullopt;
    }

    // INPUT_GROUP lines (unique variable names per group).
    for (const auto& group : inputGroups) {
      BlockArgument blockArg;
      for (auto* op : group) {
        if (auto extractOp = dyn_cast<tensor::ExtractOp>(op))
          if (auto ba = dyn_cast<BlockArgument>(extractOp.getTensor())) {
            blockArg = ba;
            break;
          }
      }
      if (!blockArg) continue;
      std::set<std::string> seen;
      out << "INPUT_GROUP";
      for (auto* op : group) {
        if (auto it = extractVarName.find(op); it != extractVarName.end())
          if (seen.insert(it->second).second) out << " " << it->second;
      }
      out << "\n";
    }

    // LOAD lines (with replication — one per use of each extract in arith).
    for (auto& entry : loadEntries)
      out << "LOAD " << entry.reg << " " << entry.varName << "\n";

    // OP lines.
    for (auto* op : operations) {
      if (!isCircuitArith(op)) continue;
      int64_t dest = opToReg[op];
      std::string opStr;
      if (isa<arith::MulIOp, arith::MulFOp>(op))
        opStr = "*";
      else if (isa<arith::AddIOp, arith::AddFOp>(op))
        opStr = "+";
      else
        opStr = "-";

      auto getOperandReg = [&](unsigned idx) -> std::string {
        // Replicated load register for extract operands.
        auto it = arithOperandReg.find(op);
        if (it != arithOperandReg.end()) {
          auto it2 = it->second.find(idx);
          if (it2 != it->second.end()) return std::to_string(it2->second);
        }
        // Another arith op's register.
        Operation* prod = op->getOperand(idx).getDefiningOp();
        if (prod && opToReg.count(prod)) return std::to_string(opToReg[prod]);
        return "const";
      };

      out << "OP " << dest << " " << getOperandReg(0) << " " << getOperandReg(1)
          << " " << opStr << "\n";
    }
  }

  // --- Call Python ---
  std::string cmd = pythonBin + " " + scriptPath + " " + circuitPath.string() +
                    " --output " + resultPath.string() + " 2>&1";
  llvm::errs() << "  [Python] Running: " << cmd << "\n";

  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    llvm::errs() << "  [Python] Failed to execute\n";
    return std::nullopt;
  }
  char buf[256];
  std::string pyOut;
  while (fgets(buf, sizeof(buf), pipe)) {
    llvm::errs() << "  [Python] " << buf;
    pyOut += buf;
  }
  int rc = pclose(pipe);
  if (rc != 0) {
    llvm::errs() << "  [Python] Failed (exit " << rc << ")\n";
    return std::nullopt;
  }

  // --- Read JSON result ---
  std::ifstream resultFile(resultPath);
  if (!resultFile) {
    llvm::errs() << "  [Python] Could not read result file\n";
    return std::nullopt;
  }
  std::stringstream ss;
  ss << resultFile.rdbuf();
  std::string json = ss.str();

  // Minimal JSON parser: extract integer arrays and scalars.
  auto extractInt = [&](const std::string& key) -> int {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1;
    pos = json.find(":", pos) + 1;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n')) pos++;
    return std::atoi(json.c_str() + pos);
  };

  auto extractIntArray = [&](const std::string& key) -> std::vector<int64_t> {
    std::vector<int64_t> result;
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    auto arrStart = json.find("[", pos);
    auto arrEnd = json.find("]", arrStart);
    if (arrStart == std::string::npos) return result;
    std::string arr = json.substr(arrStart + 1, arrEnd - arrStart - 1);
    std::istringstream is(arr);
    std::string token;
    while (std::getline(is, token, ',')) {
      // Trim whitespace
      auto s = token.find_first_not_of(" \n\r\t");
      if (s != std::string::npos) result.push_back(std::stoll(token.substr(s)));
    }
    return result;
  };

  int pyWarpSize = extractInt("warp_size");
  auto pyLanes = extractIntArray("lanes");
  auto pyAlignment = extractIntArray("alignment");

  if (pyLanes.empty() || pyAlignment.empty()) {
    llvm::errs() << "  [Python] Invalid schedule in result\n";
    return std::nullopt;
  }

  llvm::errs() << "  [Python] Schedule: warp=" << pyWarpSize
               << ", lanes=" << pyLanes.size() << ", depth="
               << (*std::max_element(pyAlignment.begin(), pyAlignment.end()) +
                   1)
               << "\n";

  // --- Build Schedule from Python result ---
  // Python's lanes/alignment arrays are indexed by register ID.
  // Only include operations that have been assigned a register.
  Schedule schedule;
  schedule.warpSize = pyWarpSize;

  llvm::SmallVector<Operation*> circuitOps;
  for (auto* op : operations) {
    if (!opToReg.count(op)) continue;
    if (!isa<tensor::ExtractOp>(op) && !isCircuitArith(op)) continue;
    circuitOps.push_back(op);
    int64_t reg = opToReg[op];
    if (reg < (int64_t)pyLanes.size()) {
      schedule.lanes[op] = pyLanes[reg];
      schedule.alignment[op] = pyAlignment[reg];
    }
  }
  schedule.instructions = std::move(circuitOps);

  // Cleanup.
  std::filesystem::remove(circuitPath);
  std::filesystem::remove(resultPath);

  return schedule;
}

}  // namespace heir
}  // namespace mlir

#endif  // USE_PYTHON_COYOTE

#endif  // HEIR_COYOTE_PYTHON_SCHEDULER_H
