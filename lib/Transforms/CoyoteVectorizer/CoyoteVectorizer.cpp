//===- CoyoteVectorizer.cpp - Coyote Vectorization Pass ---------*- C++ -*-===//
//
// Implementation of the Coyote vectorization algorithm from ASPLOS '23:
// "Coyote: A Compiler for Vectorizing Encrypted Arithmetic Circuits"
// by Raghav Malik, Kabir Sheth, and Milind Kulkarni
//
// This is the integrated version using all ported components.
//
//===----------------------------------------------------------------------===//

#include "CoyoteVectorizer.h"
#include "lib/Utils/Graph/Graph.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/DenseSet.h"
#include "llvm/include/llvm/ADT/SetVector.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LogicalResult.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_COYOTEVECTORIZER
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

/// Node in the circuit quotient graph
/// Represents one or more operations assigned to the same (epoch, column)
struct SubCircuitNodeImpl {
  /// Operations in this subcircuit (merged together during quotient search)
  /// SetVector maintains insertion order + fast lookup
  llvm::SetVector<Operation*> operations;

  /// Scheduling attributes
  int64_t epoch = -1;   // Time step (topological height)
  int64_t column = -1;  // Lane assignment (SIMD lane number)

  /// Merge another node into this one (for edge contraction)
  void merge(const SubCircuitNodeImpl& other) {
    operations.insert(other.operations.begin(), other.operations.end());
    // Keep this node's epoch/column (don't update from other)
  }

  /// Comparison operators for use in sets/maps
  bool operator==(const SubCircuitNodeImpl& other) const {
    return operations == other.operations;
  }

  bool operator<(const SubCircuitNodeImpl& other) const {
    // Compare by size first, then lexicographically by operation pointers
    if (operations.size() != other.operations.size())
      return operations.size() < other.operations.size();

    auto it1 = operations.begin(), it2 = other.operations.begin();
    while (it1 != operations.end()) {
      if (*it1 != *it2) return *it1 < *it2;
      ++it1; ++it2;
    }
    return false;
  }
};
typedef std::shared_ptr<SubCircuitNodeImpl> SubCircuitNode;
typedef graph::Graph<SubCircuitNode> CircuitGraph;

CircuitGraph buildCircuitGraph(llvm::SmallVector<Operation*> operations) {
  CircuitGraph graph;

  // Map from MLIR operation to graph node (for edge construction)
  llvm::DenseMap<Operation*, SubCircuitNode> opToVertex;

  // Step 1: Create a singleton node for each operation
  // Initially, each node contains exactly one operation
  // Later, quotient search may merge nodes together
  for (auto* op : operations) {
    auto node = std::make_shared<SubCircuitNodeImpl>();
    node->operations.insert(op);

    // Initialize scheduling attributes
    node->epoch = -1;   // Will be set by gradeGraph()
    node->column = -1;  // Will be set by column assignment/quotient search

    graph.addVertex(node);
    opToVertex[op] = node;
  }

  // Step 2: Add edges based on data dependencies
  // For each operation, look at its operands and create edges from producers
  // This captures the use-def chain as explicit graph edges
  for (auto* op : operations) {
    SubCircuitNode consumerNode = opToVertex[op];

    // For each operand used by this operation
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();

      // Only add edge if producer is in our operations list
      // (filters out function arguments, constants from other regions)
      if (producer && opToVertex.count(producer)) {
        SubCircuitNode producerNode = opToVertex[producer];
        graph.addEdge(producerNode, consumerNode);
      }
    }
  }

  return graph;
}

/// Expand operation group transitively through use-def chains
/// Python equivalent: set.union(*(comp.loaded_regs[g] for g in group))
///
/// Given a seed set of operations, returns all operations that transitively
/// depend on them. This matches Python's loaded_regs expansion.
llvm::SmallVector<Operation*> expandGroupTransitively(
    llvm::ArrayRef<Operation*> seedOps,
    llvm::ArrayRef<Operation*> allOps) {

  llvm::DenseSet<Operation*> expanded;
  llvm::SmallVector<Operation*> worklist;

  // Initialize with seed operations
  for (auto* op : seedOps) {
    expanded.insert(op);
    worklist.push_back(op);
  }

  // BFS through use-def chain
  while (!worklist.empty()) {
    Operation* current = worklist.pop_back_val();

    // Add all users of this operation's results
    for (Value result : current->getResults()) {
      for (Operation* user : result.getUsers()) {
        // Only include operations in our analysis set
        if (llvm::find(allOps, user) != allOps.end() &&
            !expanded.count(user)) {
          expanded.insert(user);
          worklist.push_back(user);
        }
      }
    }
  }

  return llvm::SmallVector<Operation*>(expanded.begin(), expanded.end());
}

/// Identify and expand input/output operation groups
/// Python equivalent: Transforms comp.input_groups using comp.loaded_regs
///
/// Two-phase process matching Python:
/// 1. Identify seed operations (ones that directly load from BlockArguments)
/// 2. Expand transitively to include all dependent operations
///
/// This matches Python's:
///   input_groups = [set().union(*(comp.loaded_regs[g] for g in group))
///                   for group in comp.input_groups]
std::pair<llvm::SmallVector<llvm::SmallVector<Operation*>>,
          llvm::SmallVector<llvm::SmallVector<Operation*>>>
identifyIOGroups(func::FuncOp func, llvm::SmallVector<Operation*> allOps) {
  llvm::SmallVector<llvm::SmallVector<Operation*>> inputGroups;
  llvm::SmallVector<llvm::SmallVector<Operation*>> outputGroups;

  // Phase 1: Group operations by which BlockArgument they use
  // This creates the initial "variable name groups"
  llvm::DenseMap<BlockArgument, llvm::SmallVector<Operation*>> argToOps;

  func.walk([&](Operation* op) {
    if (isa<func::FuncOp, func::ReturnOp>(op)) return;

    // Find which BlockArgument(s) this operation uses
    for (Value operand : op->getOperands()) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        argToOps[arg].push_back(op);
        break; // Only count each operation once per argument
      }
    }
  });

  // Phase 2: Expand each group transitively
  // This is the set.union(*(comp.loaded_regs[g] for g in group)) step
  for (auto& [arg, seedOps] : argToOps) {
    auto expandedGroup = expandGroupTransitively(seedOps, allOps);
    if (!expandedGroup.empty()) {
      inputGroups.push_back(expandedGroup);
    }
  }

  // Handle outputs (simpler - just operations feeding into return)
  llvm::SmallVector<Operation*> outputOps;
  func.walk([&](Operation* op) {
    if (isa<func::FuncOp, func::ReturnOp>(op)) return;

    for (Operation* user : op->getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        outputOps.push_back(op);
        break;
      }
    }
  });

  if (!outputOps.empty()) {
    // Also expand output group transitively
    auto expandedOutputs = expandGroupTransitively(outputOps, allOps);
    outputGroups.push_back(expandedOutputs);
  }

  return {inputGroups, outputGroups};
}

struct CoyoteVectorizerPass
    : public impl::CoyoteVectorizerBase<CoyoteVectorizerPass> {
  using CoyoteVectorizerBase::CoyoteVectorizerBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::outs() << "\n=== Coyote Vectorizer Pass ===\n\n";

    // Step 1: Collect all operations
    llvm::SmallVector<Operation*> operations;
    func.walk([&](Operation* op) {
      if (isa<func::FuncOp>(op))
        return;
      
      if (isa<func::ReturnOp>(op)) 
        return;
      
      if (!OpTrait::hasElementwiseMappableTraits(op))
        return;

      operations.push_back(op);
    });

    if (operations.empty()) {
      llvm::errs() << "No operations to vectorize\n";
      return;
    }
    llvm::errs() << "Found " << operations.size() << " operations\n";

    // Step 2: Build initial circuit graph
    llvm::errs() << "\n[1/9] Building circuit graph...\n";
    CircuitGraph graph = buildCircuitGraph(operations);
    llvm::errs() << "  Graph has " << graph.getVertices().size() << " vertices\n";

    // Step 3: Identify input/output groups
    llvm::errs() << "\n[2/9] Identifying and expanding I/O groups...\n";

    auto [inputGroups, outputGroups] = identifyIOGroups(func, operations);

    // llvm::errs() << "  Input groups: " << inputGroups.size() << "\n";
    // for (size_t i = 0; i < inputGroups.size(); ++i) {
    //   llvm::errs() << "    Group " << i << ": " << inputGroups[i].size() << ": " << *inputGroups[i];
    //                << " operations (transitively expanded)\n";
    // }
    // llvm::errs() << "  Output groups: " << outputGroups.size() << "\n";

    // // Step 4: Assign epochs (topological grading)
    // llvm::errs() << "\n[3/9] Assigning epochs (grading graph)...\n";
    // auto [inputEpochs, outputEpochs] = gradeGraph(graph, inputGroups, outputGroups);
    // llvm::errs() << "  Assigned epochs to all nodes\n";

    // // Step 5: Initial column assignment (bipartite matching)
    // llvm::errs() << "\n[4/9] Assigning initial columns (bipartite matching)...\n";
    // llvm::DenseMap<Operation*, int64_t> forceLanes;  // Empty for now

    // // Note: ColumnAssigner is in Columnize.cpp but we need to expose it
    // // For now, we'll use a simple column assignment
    // int64_t nextColumn = 0;
    // for (auto& nodePtr : graph.getNodes()) {
    //   if (nodePtr->column == -1) {
    //     nodePtr->column = nextColumn++;
    //   }
    // }
    // llvm::errs() << "  Assigned " << nextColumn << " columns\n";

    // // Step 6: Quotient search (best-first with edge grouping)
    // llvm::errs() << "\n[5/9] Quotient search...\n";
    // unsigned searchRounds = 20;  // Reduced for demo
    // auto bestSchedule = searchQuotients(graph, inputGroups, outputGroups,
    //                                    forceLanes, searchRounds);
    // llvm::errs() << "  Best cost: " << bestSchedule.cost << "\n";
    // llvm::errs() << "  (Rotation: " << bestSchedule.rotationCost
    //             << ", Height: " << bestSchedule.heightCost << ")\n";

    // // Step 7: Build lane assignment map
    // llvm::errs() << "\n[6/9] Building lane assignments...\n";
    // llvm::DenseMap<Operation*, int64_t> lanes;
    // unsigned warpSize = 8;  // Default warp size

    // for (const auto& nodePtr : bestSchedule.graph.getNodes()) {
    //   for (auto* op : nodePtr->operations) {
    //     lanes[op] = nodePtr->column;
    //     if (nodePtr->column >= (int64_t)warpSize) {
    //       warpSize = nodePtr->column + 1;
    //     }
    //   }
    // }
    // llvm::errs() << "  Warp size: " << warpSize << "\n";

    // // Step 8: Instruction alignment
    // llvm::errs() << "\n[7/9] Aligning instructions (greedy)...\n";
    // auto alignment = greedyAlign(operations, warpSize, lanes);
    // llvm::errs() << "  Schedule height: " << (*std::max_element(alignment.begin(), alignment.end()) + 1) << "\n";

    // // Step 9: Build complete schedule
    // Schedule schedule;
    // schedule.lanes = lanes;
    // for (size_t i = 0; i < operations.size(); ++i) {
    //   schedule.alignment[operations[i]] = alignment[i];
    // }
    // schedule.instructions = operations;
    // schedule.warpSize = warpSize;

    // // Step 10: Code generation
    // llvm::errs() << "\n[8/9] Generating vector code...\n";
    // auto vecCode = generateVectorCode(schedule);
    // llvm::errs() << "  Generated " << vecCode.size() << " vector instructions\n";

    // // Step 11: Optimization passes
    // llvm::errs() << "\n[9/9] Running optimization passes...\n";
    // llvm::errs() << "  - Blend optimization...\n";
    // optimizeBlends(schedule, 100);  // Reduced rounds for demo
    // llvm::errs() << "  - Rotation optimization...\n";
    // optimizeRotations(vecCode, warpSize);

    // // Step 12: Lower to MLIR
    // llvm::errs() << "\nLowering to MLIR...\n";
    // lowerToMLIR(func, vecCode, schedule);

    // // Annotate operations with their assignments
    // for (auto* op : operations) {
    //   op->setAttr("coyote.lane",
    //              IntegerAttr::get(IntegerType::get(op->getContext(), 64),
    //                              lanes[op]));
    //   op->setAttr("coyote.alignment",
    //              IntegerAttr::get(IntegerType::get(op->getContext(), 64),
    //                              schedule.alignment.lookup(op)));
    // }

    // llvm::errs() << "\n=== Coyote Vectorizer Complete ===\n\n";
  }
};

}  // namespace heir
}  // namespace mlir
