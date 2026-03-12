//===- GreedyAlignment.cpp - Greedy instruction alignment ----------------===//
//
// Port of Python's fast_align function - greedy instruction scheduling
// that respects dependencies and groups by opcode type.
//
//===----------------------------------------------------------------------===//

#ifndef GREEDY_ALIGN_H
#define GREEDY_ALIGN_H

#include <algorithm>
#include <set>
#include <vector>

#include "GraphUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"     // from @llvm-project

namespace mlir {
namespace heir {

/// Complete schedule with lane and timing assignments.
/// Python equivalent: Schedule class in codegen.py
typedef struct Schedule {
  /// Lane assignment: operation -> lane number [0, warpSize)
  llvm::DenseMap<Operation *, int64_t> lanes;

  /// Time step assignment: operation -> cycle number.
  /// Operations with the same alignment execute in parallel (different lanes).
  llvm::DenseMap<Operation *, int64_t> alignment;

  /// Ordered list of all operations in the schedule (program order).
  llvm::ArrayRef<Operation *> instructions;

  /// Number of vector lanes (SIMD width).
  unsigned warpSize = 0;

  /// Get all operations executing at a specific time step.
  llvm::SmallVector<Operation *> getStep(int64_t step) const {
    llvm::SmallVector<Operation *> result;
    for (auto *op : instructions)
      if (alignment.lookup(op) == step) result.push_back(op);
    return result;
  }

  /// Get the highest time step in the schedule (schedule depth - 1).
  int64_t maxStep() const {
    int64_t max = 0;
    for (const auto &[op, step] : alignment)
      if (step > max) max = step;
    return max;
  }
} Schedule;

//===----------------------------------------------------------------------===//
// GreedyAligner: Fast greedy instruction scheduling
//===----------------------------------------------------------------------===//

class GreedyAligner {
 public:
  /// Align instructions greedily
  /// Python: synthesize_schedule.py:122-144 (fast_align)
  ///
  /// Algorithm:
  /// 1. Build dependency graph
  /// 2. While not all scheduled:
  ///    a. Find available instructions (dependencies satisfied)
  ///    b. Pick largest group of same-opcode instructions
  ///    c. Schedule one per lane (avoid conflicts)
  ///    d. Assign to current step
  std::vector<int64_t> align(
      llvm::ArrayRef<Operation *> program, unsigned warpSize,
      const llvm::DenseMap<Operation *, int64_t> &lanes) {
    size_t numInstr = program.size();
    std::vector<int64_t> alignment(numInstr, -1);
    std::set<size_t> scheduled;

    // Build lane-to-instructions mapping
    std::vector<std::set<size_t>> columns(warpSize);
    for (size_t i = 0; i < numInstr; ++i) {
      int64_t lane = lanes.lookup(program[i]);
      if (lane >= 0 && lane < (int64_t)warpSize) {
        columns[lane].insert(i);
      }
    }

    // Build dependency graph
    auto depGraph = buildDepGraph(program);

    // Split by operation type (add, mul, sub, etc.)
    auto typeGroups = splitByType(program);

    // Greedy scheduling loop
    while (scheduled.size() < numInstr) {
      // Find available instructions (all dependencies satisfied)
      std::set<size_t> available;
      for (size_t i = 0; i < numInstr; ++i) {
        if (scheduled.count(i)) continue;

        bool allDepsSatisfied = true;
        for (size_t dep : depGraph[i]) {
          if (!scheduled.count(dep)) {
            allDepsSatisfied = false;
            break;
          }
        }

        if (allDepsSatisfied) {
          available.insert(i);
        }
      }

      if (available.empty()) {
        // Should not happen if dependency graph is acyclic
        break;
      }

      // Pick largest group of same-type instructions
      std::set<size_t> toSchedule;
      size_t maxGroupSize = 0;

      for (const auto &typeGroup : typeGroups) {
        std::set<size_t> intersection;
        std::set_intersection(
            available.begin(), available.end(), typeGroup.begin(),
            typeGroup.end(), std::inserter(intersection, intersection.begin()));

        if (intersection.size() > maxGroupSize) {
          maxGroupSize = intersection.size();
          toSchedule = intersection;
        }
      }

      // Schedule one per lane (avoid lane conflicts)
      std::set<size_t> finalSchedule;
      for (const auto &column : columns) {
        std::set<size_t> intersection;
        std::set_intersection(
            toSchedule.begin(), toSchedule.end(), column.begin(), column.end(),
            std::inserter(intersection, intersection.begin()));

        if (!intersection.empty()) {
          // Pick first instruction from this lane
          finalSchedule.insert(*intersection.begin());
        }
      }

      // Assign current step
      int64_t currentStep = 0;
      for (int64_t a : alignment) {
        if (a > currentStep) currentStep = a;
      }
      currentStep++;

      for (size_t instrIdx : finalSchedule) {
        alignment[instrIdx] = currentStep;
        scheduled.insert(instrIdx);
      }
    }

    return alignment;
  }

 private:
  /// Build dependency graph: deps[i] = {indices of instructions that i depends
  /// on} Python: synthesize_schedule.py:25-46 (dependency_graph)
  std::vector<std::vector<size_t>> buildDepGraph(
      llvm::ArrayRef<Operation *> program) {
    std::vector<std::vector<size_t>> deps(program.size());

    // Build map: value -> producing instruction index
    llvm::DenseMap<Value, size_t> valueToInstr;
    for (size_t i = 0; i < program.size(); ++i) {
      for (Value result : program[i]->getResults()) {
        valueToInstr[result] = i;
      }
    }

    // For each instruction, find dependencies
    for (size_t i = 0; i < program.size(); ++i) {
      for (Value operand : program[i]->getOperands()) {
        if (valueToInstr.count(operand)) {
          deps[i].push_back(valueToInstr[operand]);
        }
      }

      // Remove duplicates
      std::sort(deps[i].begin(), deps[i].end());
      deps[i].erase(std::unique(deps[i].begin(), deps[i].end()), deps[i].end());
    }

    return deps;
  }

  /// Split instructions by operation type
  /// Python: synthesize_schedule.py:49-68 (split_types)
  std::vector<std::set<size_t>> splitByType(
      llvm::ArrayRef<Operation *> program) {
    std::map<llvm::StringRef, std::set<size_t>> typeMap;

    for (size_t i = 0; i < program.size(); ++i) {
      llvm::StringRef opName = program[i]->getName().getStringRef();
      typeMap[opName].insert(i);
    }

    // Convert map to vector
    std::vector<std::set<size_t>> result;
    for (auto &[name, indices] : typeMap) {
      result.push_back(std::move(indices));
    }

    return result;
  }
};

/// Standalone function wrapper
std::vector<int64_t> greedyAlign(
    llvm::ArrayRef<Operation *> program, unsigned warpSize,
    const llvm::DenseMap<Operation *, int64_t> &lanes) {
  GreedyAligner aligner;
  return aligner.align(program, warpSize, lanes);
}

}  // namespace heir
}  // namespace mlir

#endif  // GREEDY_ALIGN_H
