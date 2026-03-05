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

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "GraphUtils.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/Graph/Graph.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/EquivalenceClasses.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_COYOTEVECTORIZER
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

/// Assigns topological epochs to graph nodes
/// Python reference: schedule_graph.py:56-112 (grade_nx_graph)
///
/// Algorithm:
/// 1. Clear existing epoch assignments
/// 2. Assign input_groups to epochs 0, 1, 2, ...
/// 3. Topologically visit nodes, setting epoch = max(predecessor epochs) + 1
/// 4. Assign output_groups to epochs (max_epoch + 1), (max_epoch + 2), ...
class EpochAssigner {
 public:
  using Node = SubCircuitNode;

  /// Assign epochs to all nodes in the graph
  /// Returns (input_epochs, output_epochs) sets
  std::pair<llvm::DenseSet<int64_t>, llvm::DenseSet<int64_t>> assignEpochs(
      CircuitGraph& graph,
      llvm::ArrayRef<llvm::SmallVector<Operation*>> inputGroups,
      llvm::ArrayRef<llvm::SmallVector<Operation*>> outputGroups) {
    llvm::DenseSet<int64_t> inputEpochs, outputEpochs;

    // Helper: find node containing a specific operation
    auto nodeOf = [&](Operation* op) -> Node {
      for (auto& nodePtr : graph.getVertices()) {
        if (nodePtr->operations.count(op)) {
          return nodePtr;
        }
      }
      llvm::outs() << "Warning: Operation " << *op
                   << " not found in any graph node\n";
      return nullptr;
    };

    // Step 1: Clear existing epoch assignments
    for (const Node& nodePtr : graph.getVertices()) {
      nodePtr->epoch = -1;
    }

    // Step 2: Assign input groups to epochs 0, 1, 2, ...
    for (size_t i = 0; i < inputGroups.size(); ++i) {
      for (auto* op : inputGroups[i]) {
        Node node = nodeOf(op);
        if (node) {
          node->epoch = i;
          inputEpochs.insert(i);
          llvm::outs() << "Op " << *op << " assigned to input epoch "
                       << node->epoch << "\n";
        }
      }
    }

    // Build output node set for fast lookup during epoch assignment
    std::unordered_set<Node> outputNodes;
    for (const auto& group : outputGroups) {
      for (auto* op : group) {
        if (Node node = nodeOf(op)) outputNodes.insert(node);
      }
    }

    // Step 3: Topologically sort all nodes
    auto topoOrder = graph.topologicalSort();
    assert(succeeded(topoOrder) && "CircuitGraph has a cycle");

    // Step 4: Iterate in topo order, assigning epoch = max(pred epochs) + 1
    // Skip nodes already pinned (input groups) and output nodes (assigned
    // later)
    for (Node node : *topoOrder) {
      if (node->epoch != -1) continue;        // already pinned (input group)
      if (outputNodes.count(node)) continue;  // will be pinned in step 5

      int64_t maxPredEpoch = -1;
      for (Node pred : graph.edgesInto(node)) {
        if (pred->epoch != -1)
          maxPredEpoch = std::max(maxPredEpoch, pred->epoch);
      }
      node->epoch = maxPredEpoch + 1;
      llvm::outs() << "Op " << *node->operations[0]
                   << " assigned to topoOrder epoch " << node->epoch << "\n";
    }

    // Step 5: Find max epoch and assign output groups
    int64_t maxEpoch = -1;
    for (auto& nodePtr : graph.getVertices()) {
      if (nodePtr->epoch > maxEpoch) {
        maxEpoch = nodePtr->epoch;
      }
    }

    for (size_t i = 0; i < outputGroups.size(); ++i) {
      for (auto* op : outputGroups[i]) {
        Node node = nodeOf(op);
        if (node) {
          node->epoch = maxEpoch + 1 + i;
          outputEpochs.insert(maxEpoch + 1 + i);
          llvm::outs() << "Op " << *op << " assigned to output epoch "
                       << node->epoch << "\n";
        }
      }
    }

    return {inputEpochs, outputEpochs};
  }
};

/// Standalone function wrapper for easier integration
std::pair<llvm::DenseSet<int64_t>, llvm::DenseSet<int64_t>> gradeGraph(
    CircuitGraph& graph,
    llvm::ArrayRef<llvm::SmallVector<Operation*>> inputGroups,
    llvm::ArrayRef<llvm::SmallVector<Operation*>> outputGroups) {
  EpochAssigner assigner;
  return assigner.assignEpochs(graph, inputGroups, outputGroups);
}

/// Identify input/output operation groups
///
/// Input group = all tensor.extract ops that load from the same source block
/// argument. Each unique block arg = one input group (one input vector).
/// No transitive expansion — the topo sort in EpochAssigner handles all
/// downstream ops correctly.
std::pair<llvm::SmallVector<llvm::SmallVector<Operation*>>,
          llvm::SmallVector<llvm::SmallVector<Operation*>>>
identifyIOGroups(func::FuncOp func) {
  llvm::SmallVector<llvm::SmallVector<Operation*>> inputGroups;
  llvm::SmallVector<llvm::SmallVector<Operation*>> outputGroups;

  // Group tensor.extract ops by their source block argument.
  llvm::DenseMap<BlockArgument, llvm::SmallVector<Operation*>> argToExtracts;

  func.walk([&](tensor::ExtractOp extractOp) {
    Value source = extractOp.getTensor();
    if (auto arg = dyn_cast<BlockArgument>(source)) {
      argToExtracts[arg].push_back(extractOp.getOperation());
    }
  });

  // Sort by block argument index for deterministic epoch assignment.
  llvm::SmallVector<BlockArgument> sortedArgs;
  for (auto& [arg, ops] : argToExtracts) sortedArgs.push_back(arg);
  llvm::sort(sortedArgs, [](BlockArgument a, BlockArgument b) {
    return a.getArgNumber() < b.getArgNumber();
  });
  for (auto arg : sortedArgs) {
    if (!argToExtracts[arg].empty()) inputGroups.push_back(argToExtracts[arg]);
  }

  // Output group: all tensor.insert ops, which define the output layout
  // (which lane's result goes into which output slot).
  llvm::SmallVector<Operation*> outputOps;
  func.walk([&](tensor::InsertOp insertOp) {
    outputOps.push_back(insertOp.getOperation());
  });
  if (!outputOps.empty()) {
    outputGroups.push_back(outputOps);
  }

  return {inputGroups, outputGroups};
}

/// Replaces Python's nx_columnize function
/// Python code: schedule_graph.py:118-248
///
/// Key algorithm:
/// 1. For each pair of adjacent epochs (i, j):
///    - Create bipartite graph: epoch j nodes vs their producers in epoch i
///    - Compute max-weight bipartite matching (weight = sum of degrees)
///    - Merge matched nodes into same column using Union-Find
/// 2. Limit total columns to maxLanes (210 in Python)
///
class ColumnAssigner {
 public:
  ColumnAssigner(CircuitGraph& graph,
                 llvm::DenseMap<Operation*, int64_t>& forceLanes,
                 unsigned maxLanes = 210)
      : graph(graph), forceLanes(forceLanes), maxLanes(maxLanes) {}

  /// Assign columns (lanes) to all nodes
  void assignColumns() {
    // Group nodes by epoch
    llvm::DenseMap<int64_t, llvm::SmallVector<SubCircuitNode>> epochs;
    for (auto& nodePtr : graph.getVertices()) {
      epochs[nodePtr->epoch].push_back(nodePtr);
    }

    int64_t numEpochs = epochs.size();

    // Pre-populate columns with forced lanes
    for (auto& nodePtr : graph.getVertices()) {
      for (auto* op : nodePtr->operations) {
        if (forceLanes.count(op)) {
          columns.insert(nodePtr.get());
          // Will be merged with other nodes on same forced lane later
        }
      }
    }

    // Process each pair of adjacent epochs
    for (int64_t i = 0; i < numEpochs; ++i) {
      for (int64_t j = i + 1; j < numEpochs; ++j) {
        processBipartiteMatch(epochs[i], epochs[j]);
      }
    }

    // Apply forced lane constraints
    applyForcedLanes();

    // Limit to maxLanes
    limitColumns();

    // Assign column numbers to nodes
    assignFinalColumns();
  }

 private:
  /// Process bipartite matching between two epochs using MCMF-based
  /// max-weight matching (replaces greedy BipartiteGraph::maxWeightMatching).
  /// Python: schedule_graph.py:138-230
  void processBipartiteMatch(llvm::ArrayRef<SubCircuitNode> epochI,
                             llvm::ArrayRef<SubCircuitNode> epochJ) {
    // leftNodes = epoch-j consumers; rightNodes = epoch-i producers of them.
    std::vector<SubCircuitNode> leftNodes(epochJ.begin(), epochJ.end());
    std::vector<SubCircuitNode> rightNodes;
    for (SubCircuitNode node : epochI) {
      for (SubCircuitNode jNode : epochJ) {
        if (graph.hasEdge(node, jNode)) {
          rightNodes.push_back(node);
          break;
        }
      }
    }

    auto matchableFn = [this](SubCircuitNode u, SubCircuitNode v) {
      return isMatchable(u, v);
    };

    auto matched =
        maxWeightBipartiteMatching(graph, leftNodes, rightNodes, matchableFn);

    for (auto& [u, v] : matched) columns.unionSets(u.get(), v.get());

    // Ensure all epoch-j nodes are registered in the column structure.
    for (SubCircuitNode node : epochJ) {
      if (!columns.contains(node.get())) columns.insert(node.get());
    }
  }

  /// Check if two nodes can be matched
  /// Python: schedule_graph.py:191-201
  bool isMatchable(SubCircuitNode u, SubCircuitNode v) {
    // Don't match if it would create epoch conflicts
    // i.e., if u's column already contains a node from v's epoch

    if (!columns.contains(u.get()) || !columns.contains(v.get())) {
      return true;  // At least one not yet in a column
    }

    if (columns.isEquivalent(u.get(), v.get())) {
      return false;  // Already in same column
    }

    // Get epochs in u's and v's columns
    std::set<int64_t> uEpochs, vEpochs;
    for (auto& nodePtr : graph.getVertices()) {
      if (columns.isEquivalent(nodePtr.get(), u.get()))
        uEpochs.insert(nodePtr->epoch);
      if (columns.isEquivalent(nodePtr.get(), v.get()))
        vEpochs.insert(nodePtr->epoch);
    }

    // Can't match if v's epoch is in u's column, or vice versa
    return !uEpochs.count(v->epoch) && !vEpochs.count(u->epoch);
  }

  /// Apply forced lane constraints
  /// Python: schedule_graph.py:172-177
  void applyForcedLanes() {
    // For each forced lane, unite all nodes that should be on that lane
    llvm::DenseMap<int64_t, llvm::SmallVector<SubCircuitNode>> laneGroups;

    for (auto& nodePtr : graph.getVertices()) {
      for (auto* op : nodePtr->operations) {
        if (forceLanes.count(op)) {
          laneGroups[forceLanes[op]].push_back(nodePtr);
        }
      }
    }

    for (auto& [lane, nodes] : laneGroups) {
      if (nodes.empty()) continue;
      SubCircuitNode rep = nodes[0];
      for (SubCircuitNode node : nodes) {
        columns.unionSets(rep.get(), node.get());
      }
    }
  }

  /// Extract all equivalence classes as a vector-of-vectors.
  std::vector<std::vector<SubCircuitNodeImpl*>> getColumnSets() {
    std::vector<std::vector<SubCircuitNodeImpl*>> sets;
    // Range-for over EquivalenceClasses yields const ECValue *.
    for (const auto* ecVal : columns) {
      if (!ecVal->isLeader()) continue;
      std::vector<SubCircuitNodeImpl*> members;
      for (auto mit = columns.member_begin(*ecVal); mit != columns.member_end();
           ++mit)
        members.push_back(*mit);
      sets.push_back(std::move(members));
    }
    return sets;
  }

  /// Limit number of columns to maxLanes
  /// Python: disjoint_set.py:28-88 (limit_classes)
  void limitColumns() {
    auto columnSets = getColumnSets();
    if (columnSets.size() <= maxLanes) {
      return;  // Already within limit
    }

    // Merge columns to reduce count
    // Simple strategy: merge smallest columns
    // TODO: Implement Python's sophisticated chunk allocation
    while (columnSets.size() > maxLanes) {
      // Find two smallest columns and merge them
      std::sort(
          columnSets.begin(), columnSets.end(),
          [](const auto& a, const auto& b) { return a.size() < b.size(); });

      auto& smallest = columnSets[0];
      auto& second = columnSets[1];

      // Merge smallest into second
      for (SubCircuitNodeImpl* node : smallest) {
        columns.unionSets(node, *second.begin());
      }

      // Remove smallest from list
      columnSets.erase(columnSets.begin());
    }
  }

  /// Assign final column numbers
  /// Python: schedule_graph.py:244-246
  void assignFinalColumns() {
    auto columnSets = getColumnSets();

    // Assign column index to each node
    for (size_t colIdx = 0; colIdx < columnSets.size(); ++colIdx) {
      for (SubCircuitNodeImpl* node : columnSets[colIdx]) {
        node->column = colIdx;
      }
    }
  }

 private:
  CircuitGraph& graph;
  llvm::DenseMap<Operation*, int64_t>& forceLanes;
  unsigned maxLanes;
  llvm::EquivalenceClasses<SubCircuitNodeImpl*> columns;
};

struct CoyoteVectorizerPass
    : public impl::CoyoteVectorizerBase<CoyoteVectorizerPass> {
  using CoyoteVectorizerBase::CoyoteVectorizerBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::outs() << "\n=== Coyote Vectorizer Pass ===\n\n";

    //==========================================================================
    // Step 1: Identify I/O groups and collect operations
    // Python equivalent: input_groups = [set().union(*(comp.loaded_regs[g]
    //                                     for g in group)) for group in
    //                                     comp.input_groups]
    //==========================================================================
    // identifyIOGroups is called BEFORE buildCircuitGraph to guarantee that
    // tensor.extract / tensor.insert ops are always present in the graph.
    // gradeGraph pins IO ops to their epochs via nodeOf(); if those ops are
    // absent, nodeOf returns nullptr and the schedule is silently broken.
    llvm::errs() << "\n[1/9] Identifying and expanding I/O groups...\n";

    auto [inputGroups, outputGroups] = identifyIOGroups(func);

    llvm::errs() << "  Input groups: " << inputGroups.size() << "\n";
    for (size_t i = 0; i < inputGroups.size(); ++i) {
      llvm::errs() << "    Group " << i << ": " << inputGroups[i].size()
                   << " operations (transitively expanded)\n";
    }
    llvm::errs() << "  Output groups: " << outputGroups.size() << "\n";

    // Collect all operations, then union in IO ops so gradeGraph always sees
    // them.
    llvm::DenseSet<Operation*> opSet;
    func.walk([&](Operation* op) {
      if (isa<func::FuncOp>(op)) return;

      if (isa<func::ReturnOp>(op)) return;

      if (!OpTrait::hasElementwiseMappableTraits(op)) return;

      opSet.insert(op);
    });

    // Debug: Print identified groups
    for (auto& group : inputGroups) {
      llvm::outs() << "Expanding input group with " << group.size()
                   << " seed ops...\n";
      for (auto* op : group) {
        llvm::outs() << "  Seed op: " << *op << "\n";
      }
    }
    for (auto& group : outputGroups) {
      llvm::outs() << "Expanding output group with " << group.size()
                   << " seed ops...\n";
      for (auto* op : group) {
        llvm::outs() << "  Seed op: " << *op << "\n";
      }
    }

    for (auto& group : inputGroups)
      for (auto* op : group) opSet.insert(op);
    for (auto& group : outputGroups)
      for (auto* op : group) opSet.insert(op);

    llvm::SmallVector<Operation*> operations(opSet.begin(), opSet.end());
    // Restore program (walk) order so dependency edges are correct.
    llvm::sort(operations, [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });

    if (operations.empty()) {
      llvm::errs() << "No operations to vectorize\n";
      return;
    }
    llvm::errs() << "Found " << operations.size() << " operations total\n";

    //==========================================================================
    // Step 2: Build initial circuit graph
    // Python equivalent: graph = instr_sequence_to_nx_graph(comp.code)
    //==========================================================================
    llvm::errs() << "\n[2/9] Building circuit graph...\n";
    CircuitGraph graph = buildCircuitGraph(operations);
    llvm::errs() << "  Graph has " << graph.getVertices().size()
                 << " vertices\n";

    //==========================================================================
    // Step 3: Assign epochs (topological grading)
    // Python equivalent: Done inside pq_relax_schedule()
    //==========================================================================
    // Epoch assignment is the first phase of scheduling. It assigns each
    // operation to a time step (epoch) based on the longest path from inputs.
    //
    // This is similar to ASAP (As Soon As Possible) scheduling in traditional
    // compilers. Operations are scheduled as early as their dependencies allow.
    //
    // In Python, this happens inside the protoschedule search, not as a
    // separate step. We do it explicitly here for clarity.
    llvm::errs() << "\n[3/9] Assigning epochs (grading graph)...\n";
    auto [inputEpochs, outputEpochs] =
        gradeGraph(graph, inputGroups, outputGroups);
    llvm::outs() << inputEpochs.size() << " input epochs, "
                 << outputEpochs.size() << " output epochs\n";
    llvm::errs() << "Assigned epochs to all nodes\n";

    //==========================================================================
    // Step 4: Build lane constraint map
    // Python equivalent: loaded_lanes = {next(iter(comp.loaded_regs[g])):
    //                                     comp.force_lanes[g] for g in
    //                                     comp.force_lanes}
    //==========================================================================
    // Force certain operations into specific lanes to satisfy alignment
    // requirements for certain SIMD operations.
    //
    // In Python, this converts group-based constraints (comp.force_lanes) to
    // register-based constraints (loaded_lanes) using comp.loaded_regs.
    //
    // TODO: Extract forceLanes from operation attributes or pass options
    // For now, empty (no forced constraints)
    llvm::errs() << "\n[4/9] Building lane constraints...\n";
    llvm::DenseMap<Operation*, int64_t> forceLanes;  // Empty for now
    // TODO: Check for heir.forced_lane attributes on operations
    // TODO: Add pass option for user-specified constraints
    llvm::errs() << "  No forced lane constraints\n";

    //==========================================================================
    // Step 4.5: Columnize (bipartite matching for initial lane assignment)
    // Python equivalent: nx_columnize(graph, ...)
    //==========================================================================
    // Seeds each node's column using max-weight bipartite matching across
    // adjacent epoch pairs. This gives the annealer a structured starting
    // point rather than a flat/random one, dramatically reducing the search
    // burden in searchQuotients.
    llvm::errs() << "\n[4.5/9] Columnizing (bipartite matching)...\n";
    ColumnAssigner columnizer(graph, forceLanes);
    columnizer.assignColumns();
    llvm::errs() << "  Initial lane assignment done\n";

    //==========================================================================
    // Step 5: Quotient search (best-first with edge grouping)
    // Python equivalent: protoschedule = pq_relax_schedule(graph, ...)
    //==========================================================================
    // This is the CORE of the Coyote algorithm. It explores the space of
    // possible schedules using a priority-queue based search over quotient
    // graphs.
    //
    // Key idea: Start with fine-grained graph (one op per node), progressively
    // merge nodes together (creating quotient graphs) to explore different
    // lane assignments. Use a cost function balancing rotation cost vs height.
    //
    // Python function signature:
    //   pq_relax_schedule(graph, input_groups, output_groups, loaded_lanes,
    //                     rounds=search_rounds)
    //
    // Search process:
    // 1. Start with initial graph (from grading)
    // 2. Generate candidate merges (quotient graphs)
    // 3. Evaluate cost (rotation + height)
    // 4. Explore best candidates first (priority queue)
    // 5. Return best schedule found within budget
    //
    // After this step, each node in the graph has:
    // - epoch: time step
    // - column: lane number

    // llvm::errs() << "\n[5/9] Quotient search...\n";
    // unsigned searchRounds = 200;  // Match Python default (was 20, too low)
    // auto bestSchedule = searchQuotients(graph, inputGroups, outputGroups,
    //                                    forceLanes, searchRounds);
    // llvm::errs() << "  Best cost: " << bestSchedule.cost << "\n";
    // llvm::errs() << "  (Rotation: " << bestSchedule.rotationCost
    //             << ", Height: " << bestSchedule.heightCost << ")\n";

    // // Step 7: Build lane assignment map
    // llvm::errs() << "\n[6/9] Building lane assignments...\n";
    // llvm::DenseMap<Operation*, int64_t> lanes;
    // unsigned warpSize = 8;  // Default warp size

    // for (const auto& nodePtr : bestSchedule.graph.getVertices()) {
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
    // llvm::errs() << "  Schedule height: " <<
    // (*std::max_element(alignment.begin(), alignment.end()) + 1) << "\n";

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
    // llvm::errs() << "  Generated " << vecCode.size() << " vector
    // instructions\n";

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
