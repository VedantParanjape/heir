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
#include "GreedyAlign.h"
#include "Optimizations.h"
#include "SimulatedAnnealing.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/Graph/Graph.h"
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
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/RegionUtils.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_COYOTEVECTORIZER
#include "lib/Transforms/CoyoteVectorizer/CoyoteVectorizer.h.inc"

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

  /// Print column assignments to llvm::errs() for debugging.
  void printColumns() const {
    // Collect nodes grouped by column, sorted by epoch within each column.
    std::map<int64_t, std::vector<SubCircuitNodeImpl*>> byColumn;
    for (auto& nodePtr : graph.getVertices())
      byColumn[nodePtr->column].push_back(nodePtr.get());
    llvm::errs() << "=== Column Assignments (" << byColumn.size()
                 << " columns) ===\n";
    for (auto& [col, nodes] : byColumn) {
      llvm::errs() << "  col " << col << ":\n";
      // Sort by epoch for a consistent print order.
      auto sorted = nodes;
      std::sort(sorted.begin(), sorted.end(),
                [](SubCircuitNodeImpl* a, SubCircuitNodeImpl* b) {
                  return a->epoch < b->epoch;
                });
      for (auto* node : sorted) {
        llvm::errs() << "    epoch=" << node->epoch
                     << "  ops=" << node->operations.size() << " [";
        bool first = true;
        for (auto* op : node->operations) {
          if (!first) llvm::errs() << ", ";
          op->print(llvm::errs());
          first = false;
        }
        llvm::errs() << "]\n";
      }
    }
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

  /// Limit number of columns to maxLanes using DSatur graph coloring.
  ///
  /// Epoch-disjoint constraint maps exactly to graph coloring:
  ///   - Nodes  = column indices (0..N-1)
  ///   - Edges  = conflict: columns i and j share a node from the same epoch
  ///   - Colors = bins (we want ≤ K colors)
  ///
  /// After DSatur we have M ≤ K color classes.  We then do a balanced split
  /// (peel one element off the largest class, repeat until K bins) to reach
  /// exactly K bins.  Splitting within a class is always safe: same-class
  /// nodes come from different epochs, so any sub-partition is epoch-disjoint.
  void limitColumns() {
    auto columnSets = getColumnSets();
    if (columnSets.size() <= maxLanes) return;

    size_t N = columnSets.size();
    size_t K = maxLanes;

    // Step 1: Build conflict graph.
    // Columns i and j conflict if they share a node from the same epoch —
    // merging them would put same-epoch nodes in the same lane (illegal).
    graph::UndirectedGraph<size_t> conflictGraph;
    for (size_t i = 0; i < N; ++i) conflictGraph.addVertex(i);

    // Map epoch → list of column indices containing nodes from that epoch.
    llvm::DenseMap<int64_t, llvm::SmallVector<size_t, 4>> epochToCols;
    for (size_t i = 0; i < N; ++i) {
      llvm::SmallSet<int64_t, 4> seen;
      for (auto* node : columnSets[i])
        if (seen.insert(node->epoch).second)
          epochToCols[node->epoch].push_back(i);
    }
    for (auto& [epoch, cols] : epochToCols)
      for (size_t a = 0; a < cols.size(); ++a)
        for (size_t b = a + 1; b < cols.size(); ++b)
          conflictGraph.addEdge(cols[a], cols[b]);

    // Step 2: DSatur graph coloring → epoch-disjoint color assignment.
    graph::GreedyGraphColoring<size_t> coloring;
    auto colorMap = coloring.color(conflictGraph);  // column_idx → color

    // Group column indices by color.
    std::map<int, std::vector<size_t>> colorClasses;
    for (size_t i = 0; i < N; ++i) colorClasses[colorMap[i]].push_back(i);

    // Step 3: Balanced split to reach exactly K bins.
    // colorClasses has M ≤ K classes.  Split the largest ones until K bins.
    // Splitting within a class is safe: same-class nodes have no epoch
    // conflicts, so any sub-partition remains epoch-disjoint.
    std::vector<std::vector<size_t>> bins;
    for (auto& [color, cols] : colorClasses) bins.push_back(cols);

    while (bins.size() < K) {
      auto it = std::max_element(
          bins.begin(), bins.end(),
          [](const auto& a, const auto& b) { return a.size() < b.size(); });
      if (it->size() <= 1) break;  // can't split singletons
      bins.push_back({it->back()});
      it->pop_back();
    }

    // Step 4: Merge all column-sets within the same bin.
    for (auto& bin : bins) {
      if (bin.empty()) continue;
      SubCircuitNodeImpl* rep = columnSets[bin[0]][0];
      for (size_t idx : bin)
        for (auto* node : columnSets[idx]) columns.unionSets(rep, node);
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
    columnizer.printColumns();
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

    llvm::errs() << "\n[5/9] Quotient search...\n";
    unsigned searchRounds = 200;  // Match Python default (was 20, too low)
    auto bestSchedule = searchQuotients(graph, inputGroups, outputGroups,
                                        forceLanes, searchRounds);
    llvm::errs() << "  Best cost: " << bestSchedule.cost << "\n";
    llvm::errs() << "  (Rotation: " << bestSchedule.rotationCost
                 << ", Height: " << bestSchedule.heightCost << ")\n";
    printGraph(bestSchedule.graph, "Best schedule graph after quotient search");

    //==========================================================================
    // Step 6: Normalize column assignments and extract metadata
    // Python equivalent: Lines 76-82 in vectorize_circuit.py
    //==========================================================================
    // After quotient search, normalize columns to start at 0 and extract
    // the lane assignments for each operation.
    //
    // Python code:
    //   min_column = min(protoschedule.nodes[node]['column'] ...)
    //   for node in protoschedule.nodes:
    //     protoschedule.nodes[node]['column'] -= min_column
    //   warp_size = max(protoschedule.nodes[node]['column'] ...) + 1
    //   lanes = get_lanes(protoschedule)
    //
    // Note: In Python, quotient nodes may contain multiple operations, so we
    // need to extract per-operation lane assignments.
    llvm::errs() << "\n[6/9] Normalizing and extracting lane assignments...\n";

    // Find minimum column (should already be 0 after quotient search, but
    // check)
    int64_t minColumn = INT64_MAX;
    for (const auto& nodePtr : bestSchedule.graph.getVertices()) {
      if (nodePtr->column < minColumn) {
        minColumn = nodePtr->column;
      }
    }

    // Normalize and build per-operation lane map
    llvm::DenseMap<Operation*, int64_t> lanes;
    unsigned warpSize = 0;

    for (const auto& nodePtr : bestSchedule.graph.getVertices()) {
      // Normalize column
      int64_t normalizedColumn = nodePtr->column - minColumn;

      // Assign all operations in this node to the same lane
      for (auto* op : nodePtr->operations) {
        lanes[op] = normalizedColumn;
      }

      // Track maximum for warp size
      if (normalizedColumn >= (int64_t)warpSize) {
        warpSize = normalizedColumn + 1;
      }
    }
    llvm::errs() << "  Warp size: " << warpSize << " lanes\n";

    //==========================================================================
    // Step 7: Fine-grained instruction alignment (epoch-by-epoch)
    // Python equivalent: Lines 211-228 in vectorize_circuit.py
    //==========================================================================
    // The quotient search assigns operations to epochs (coarse-grained timing).
    // This step computes fine-grained alignment within each epoch
    // independently, then concatenates epochs sequentially using a running
    // program_length offset.
    //
    // Python code:
    //   program_length = 0
    //   for stage in get_stages(protoschedule):
    //     stage_instrs = [comp.code[i] for i in stage]
    //     stage_align = fast_align(stage_instrs, warp_size, lanes)
    //     for s, i in zip(stage_align, stage_instrs):
    //       alignment[i.dest.val] = s + program_length
    //     program_length = max(alignment) + 1
    //
    // Each call to fast_align (greedyAlign) only sees the instructions in the
    // current epoch, so cross-epoch dependencies are implicitly satisfied
    // (they reference operations not present in the local dep graph, thus
    // treated as already-scheduled). Epochs are then stacked sequentially.
    llvm::errs()
        << "\n[7/9] Computing fine-grained alignment (epoch-by-epoch)...\n";

    // Build op -> epoch map from the best quotient graph.
    llvm::DenseMap<Operation*, int64_t> opToEpoch;
    for (const auto& nodePtr : bestSchedule.graph.getVertices())
      for (auto* op : nodePtr->operations) opToEpoch[op] = nodePtr->epoch;

    // Group operations by epoch, preserving program order within each epoch.
    std::map<int64_t, llvm::SmallVector<Operation*>> epochOps;
    for (auto* op : operations) {
      int64_t epoch =
          opToEpoch.lookup(op);  // 0 for ops not in graph (constants etc.)
      epochOps[epoch].push_back(op);
    }

    // Align per epoch, accumulating a global offset.
    llvm::DenseMap<Operation*, int64_t> alignmentMap;
    int64_t programLength = 0;

    for (auto& [epoch, stageOps] : epochOps) {
      auto stageAlign = greedyAlign(stageOps, warpSize, lanes);

      int64_t stageMax = 0;
      for (size_t i = 0; i < stageOps.size(); ++i) {
        int64_t t = stageAlign[i] + programLength;
        alignmentMap[stageOps[i]] = t;
        stageMax = std::max(stageMax, t);
      }
      programLength = stageMax + 1;
    }

    // Convert to index-parallel vector matching `operations` order.
    std::vector<int64_t> alignment(operations.size());
    for (size_t i = 0; i < operations.size(); ++i)
      alignment[i] = alignmentMap.lookup(operations[i]);

    llvm::errs() << "  Schedule height: " << programLength << " cycles\n";

    //==========================================================================
    // Step 8: Build complete Schedule object
    // Python equivalent: schedule = Schedule(lanes, alignment, comp.code)
    //==========================================================================
    // Package all scheduling information into a single Schedule structure.
    // This is the input to code generation and optimization passes.
    Schedule schedule;
    schedule.lanes = lanes;
    for (size_t i = 0; i < operations.size(); ++i) {
      schedule.alignment[operations[i]] = alignment[i];
    }
    schedule.instructions = operations;
    schedule.warpSize = warpSize;

    // //==========================================================================
    // // Step 9: Optimize blend operations
    // // Python equivalent: blend_relaxed_schedule = relax_blends(schedule)
    // //==========================================================================
    // // Blends are operations that combine values from different sources based
    // // on lane masks. They're used to implement cross-lane data movement.
    // //
    // // Python relax_blends() optimizes blend placement to reduce overhead:
    // // - Move blends closer to their use points
    // // - Merge redundant blends
    // // - Eliminate unnecessary blends
    // //
    // // This changes the schedule (updates alignment), so we do it before
    // // code generation.
    // //
    // // IMPORTANT: In Python, this step produces a NEW schedule. We need to
    // // update our schedule in-place or regenerate it.
    llvm::errs() << "\n[8/9] Optimizing blend operations...\n";
    optimizeBlends(schedule, 100);  // TODO: Match Python's iteration count
    llvm::errs() << "  Blend optimization complete\n";

    //==========================================================================
    // Step 10: Lower schedule to MLIR tensor operations
    // Python equivalent: codegen() + lowering
    //==========================================================================
    // Emit tensor_ext.rotate, element-wise arith ops on tensors, and
    // tensor.from_elements masks for blends — directly from the Schedule,
    // without going through the VecInstr intermediate representation.
    //
    // After lowering, scalar op results are replaced by tensor.extract from
    // the vectorized result tensors, and the original scalar ops are erased.
    llvm::errs() << "\n[9/9] Lowering schedule to MLIR tensor operations...\n";
    lowerToMLIR(func, schedule);
    llvm::errs() << "  Lowering complete\n";

    // Canonicalize + DCE to clean up dead ops left behind by lowering.
    {
      MLIRContext* ctx = func.getContext();
      RewritePatternSet patterns(ctx);
      for (auto* dialect : ctx->getLoadedDialects())
        dialect->getCanonicalizationPatterns(patterns);
      for (RegisteredOperationName op : ctx->getRegisteredOperations())
        op.getCanonicalizationPatterns(patterns, ctx);

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal();
      (void)applyPatternsGreedily(func, std::move(patterns), config);

      IRRewriter rewriter(ctx);
      (void)mlir::eraseUnreachableBlocks(rewriter,
                                         func.getOperation()->getRegions());
    }

    llvm::errs() << "\n=== Generated MLIR ===\n";
    func.print(llvm::errs());
    llvm::errs() << "\n=====================\n";

    llvm::errs() << "\n=== Coyote Vectorizer Complete ===\n";
    llvm::errs() << "Final schedule: " << warpSize << " lanes, "
                 << schedule.maxStep() + 1 << " cycles\n\n";
  }
};

}  // namespace heir
}  // namespace mlir
