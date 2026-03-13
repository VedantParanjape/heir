//===- MaxWeightBipartiteMatching.h - Bipartite matching for columnize ----===//
//
// Implements max-weight bipartite matching equivalent to:
//   nx.algorithms.max_weight_matching(G, maxcardinality=True)
//
// Used by the columnize phase of the Coyote vectorizer to seed initial lane
// assignments. For each (epoch_i, epoch_j) pair, finds the maximum-weight
// matching between epoch-i nodes and epoch-j nodes in the circuit graph, where
// edge weight = sum of total degrees of the two endpoints.
//
// Python equivalent: schedule_graph.py:209
//   nx.algorithms.max_weight_matching(matchable_graph, maxcardinality=True)
//
// Algorithm: min-cost max-flow via SPFA (successive shortest paths).
//   The flow network is built on graph::Graph<int> (heir's Graph utility).
//   Residual capacities and costs are tracked in std::maps alongside it.
//
//   maxcardinality=True is encoded by adding a cardinality bonus M to every
//   matching edge cost, making every augmenting path cheaper than not
//   augmenting. Weight then acts as a tiebreaker within equal cardinality.
//
// This file is standalone and not yet integrated into the main pass.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_COYOTEVECTORIZER_MAXWEIGHTBIPARTITEMATCHING_H_
#define LIB_TRANSFORMS_COYOTEVECTORIZER_MAXWEIGHTBIPARTITEMATCHING_H_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <vector>

#include "llvm/include/llvm/ADT/EquivalenceClasses.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"             // from @llvm-project
#include "ortools/graph/min_cost_flow.h"

// heir graph utility — provides Graph<V> with addVertex/addEdge/edgesOutOf/
// getInDegree/getOutDegree/hasEdge/getVertices.
#include "lib/Utils/Graph/Graph.h"

namespace mlir {
namespace heir {

/// Node in the circuit quotient graph
/// Represents one or more operations assigned to the same (epoch, column)
struct SubCircuitNodeImpl {
  /// Operations in this subcircuit (merged together during quotient search)
  /// SetVector maintains insertion order + fast lookup
  llvm::SetVector<Operation *> operations;

  /// Scheduling attributes
  int64_t epoch = -1;   // Time step (topological height)
  int64_t column = -1;  // Lane assignment (SIMD lane number)

  /// Merge another node into this one (for edge contraction)
  void merge(const SubCircuitNodeImpl &other) {
    operations.insert(other.operations.begin(), other.operations.end());
    // Keep this node's epoch/column (don't update from other)
  }

  /// Comparison operators for use in sets/maps
  bool operator==(const SubCircuitNodeImpl &other) const {
    return operations == other.operations;
  }

  bool operator<(const SubCircuitNodeImpl &other) const {
    // Compare by size first, then lexicographically by operation pointers
    if (operations.size() != other.operations.size())
      return operations.size() < other.operations.size();

    auto it1 = operations.begin(), it2 = other.operations.begin();
    while (it1 != operations.end()) {
      if (*it1 != *it2) return *it1 < *it2;
      ++it1;
      ++it2;
    }
    return false;
  }
};
typedef std::shared_ptr<SubCircuitNodeImpl> SubCircuitNode;
typedef graph::Graph<SubCircuitNode> CircuitGraph;

//===----------------------------------------------------------------------===//
// Deep copy
//===----------------------------------------------------------------------===//

/// Deep-copy a CircuitGraph, cloning all SubCircuitNodeImpl objects.
///
/// A plain copy of Graph<shared_ptr<T>> only copies the shared_ptr handles —
/// the underlying SubCircuitNodeImpl objects are shared between both copies.
/// That means mutations through one copy (epoch/column updates during SA,
/// merge() during edge contraction) silently affect the other, corrupting the
/// priority-queue search state.
///
/// This function mints a fresh SubCircuitNodeImpl for every vertex so the two
/// graphs are fully independent.  It uses only Graph<V>'s public interface
/// (getVertices, edgesOutOf) so Graph.h itself needs no changes.
///
/// Note: edgesOutOf() is non-const (it uses operator[] internally), so src
/// must be passed by non-const reference.
/// Clone a single node — SetVector has no copy constructor so we copy field
/// by field.
inline SubCircuitNode cloneNode(const SubCircuitNode &v) {
  auto n = std::make_shared<SubCircuitNodeImpl>();
  for (auto *op : v->operations) n->operations.insert(op);
  n->epoch = v->epoch;
  n->column = v->column;
  return n;
}

inline CircuitGraph deepCopyCircuitGraph(CircuitGraph &src) {
  // Build old-ptr → fresh-clone mapping.
  std::unordered_map<SubCircuitNode, SubCircuitNode> remap;
  for (const auto &v : src.getVertices()) remap[v] = cloneNode(v);

  // Add cloned vertices, then rebuild edges using remapped pointers.
  CircuitGraph result;
  for (auto &[oldV, newV] : remap) result.addVertex(newV);

  for (auto &[oldV, newV] : remap)
    for (const auto &oldSucc : src.edgesOutOf(oldV))
      result.addEdge(newV, remap.at(oldSucc));

  return result;
}

/// Replicate multi-use tensor.extract ops so each clone has a single arith
/// consumer.  This mirrors Python Coyote's allow_replicating='all' behavior:
/// each use of an input variable gets its own load register, making optimal
/// lane placement obvious and letting the scheduler converge faster.
///
/// Called between collecting operations and buildCircuitGraph.
void replicateMultiUseExtracts(
    llvm::SmallVector<Operation *> &operations,
    llvm::SmallVector<llvm::SmallVector<Operation *>> &inputGroups) {
  for (int i = 0; i < inputGroups.size(); i++) {
    SmallVector<Operation *> origInputGroup(inputGroups[i]);
    for (auto *op : origInputGroup) {
      size_t NumUses = op->getResult(0).getNumUses();
      SmallVector<OpOperand *> opUses;
      for (auto &use : op->getResult(0).getUses()) opUses.push_back(&use);

      SmallVector<Operation *> inputClones;
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);

      for (int i = 0; i < NumUses - 1; i++) {
        inputClones.push_back(builder.clone(*op));
      }

      for (int j = 1; j < NumUses; j++) {
        opUses[j]->set(inputClones[j - 1]->getResult(0));
        inputGroups[i].push_back(inputClones[j - 1]);
        operations.push_back(inputClones[j - 1]);
      }
    }
  }
}

CircuitGraph buildCircuitGraph(llvm::SmallVector<Operation *> operations) {
  CircuitGraph graph;

  // Map from MLIR operation to graph node (for edge construction)
  llvm::DenseMap<Operation *, SubCircuitNode> opToVertex;

  // Step 1: Create a singleton node for each operation
  // Initially, each node contains exactly one operation
  // Later, quotient search may merge nodes together
  for (auto *op : operations) {
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
  for (auto *op : operations) {
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

//===- MaxWeightBipartiteMatching.h - Bipartite matching for columnize ----===//
//
// Implements max-weight bipartite matching equivalent to:
//   nx.algorithms.max_weight_matching(G, maxcardinality=True)
//
// Used by the columnize phase of the Coyote vectorizer to seed initial lane
// assignments. For each (epoch_i, epoch_j) pair, finds the maximum-weight
// matching between epoch-i nodes and epoch-j nodes in the circuit graph, where
// edge weight = sum of total degrees of the two endpoints.
//
// Python equivalent: schedule_graph.py:209
//   nx.algorithms.max_weight_matching(matchable_graph, maxcardinality=True)
//
// Algorithm: min-cost max-flow via OR-Tools SimpleMinCostFlow.
//   The flow network has source (0), left nodes (1..L), right nodes (L+1..L+R),
//   and sink (L+R+1). Each arc has unit capacity.
//
//   maxcardinality=True is encoded by adding a cardinality bonus M to every
//   matching edge cost, making every augmenting path cheaper than not
//   augmenting. Weight then acts as a tiebreaker within equal cardinality.
/// Compute a maximum-weight bipartite matching between leftNodes (epoch i) and
/// rightNodes (epoch j).
///
/// An edge (u, v) exists iff the circuit graph has a directed edge in either
/// direction. Its weight is:
///   getInDegree(u) + getOutDegree(u) + getInDegree(v) + getOutDegree(v)
///
/// maxcardinality=True semantics (matching Python):
///   A cardinality bonus M = 2 * maxDeg * min(L,R) + 1 is added to each
///   matching edge cost so the solver always prefers augmenting. Weight breaks
///   ties within equal cardinality.
///
/// An optional `isMatchable` predicate can reject individual edges (used to
/// prevent epoch conflicts in the Union-Find column structure during
/// columnize).
///
/// Returns matched (leftNode, rightNode) pairs.
std::vector<std::pair<SubCircuitNode, SubCircuitNode>>
maxWeightBipartiteMatching(
    CircuitGraph &circuitGraph, const std::vector<SubCircuitNode> &leftNodes,
    const std::vector<SubCircuitNode> &rightNodes,
    std::function<bool(SubCircuitNode, SubCircuitNode)> isMatchable = nullptr) {
  if (leftNodes.empty() || rightNodes.empty()) return {};

  const int L = leftNodes.size();
  const int R = rightNodes.size();
  const int source = 0, sink = L + R + 1;

  operations_research::SimpleMinCostFlow mcmf;

  // source → left nodes (arc indices 0..L-1)
  for (int i = 0; i < L; ++i)
    mcmf.AddArcWithCapacityAndUnitCost(source, i + 1, 1, 0);

  // right nodes → sink (arc indices L..L+R-1)
  for (int j = 0; j < R; ++j)
    mcmf.AddArcWithCapacityAndUnitCost(L + 1 + j, sink, 1, 0);

  // Compute cardinality bonus: M > any single edge weight so that
  // maximising cardinality always dominates maximising weight.
  // O(L+R): 2 * maxDeg * min(L,R) + 1 exceeds the total weight of any
  // matching (at most min(L,R) edges, each with weight at most 2*maxDeg).
  int maxDeg = 0;
  for (auto &n : leftNodes)
    maxDeg = std::max(maxDeg, (int)(circuitGraph.getInDegree(n) +
                                    circuitGraph.getOutDegree(n)));
  for (auto &n : rightNodes)
    maxDeg = std::max(maxDeg, (int)(circuitGraph.getInDegree(n) +
                                    circuitGraph.getOutDegree(n)));
  const int cardinalityBonus = 2 * maxDeg * std::min(L, R) + 1;

  // left → right matching edges (arc indices starting at L+R).
  // Store (i, j) alongside each arc so we can recover node pairs after solve.
  std::vector<std::pair<int, int>> matchingArcNodes;
  for (int i = 0; i < L; ++i) {
    SubCircuitNode u = leftNodes[i];
    int degU = circuitGraph.getInDegree(u) + circuitGraph.getOutDegree(u);

    for (int j = 0; j < R; ++j) {
      SubCircuitNode v = rightNodes[j];

      if (!circuitGraph.hasEdge(u, v) && !circuitGraph.hasEdge(v, u)) continue;

      if (isMatchable && !isMatchable(u, v)) continue;

      int degV = circuitGraph.getInDegree(v) + circuitGraph.getOutDegree(v);
      int weight = degU + degV;
      // Negative cost so solver prefers this edge; cardinality bonus ensures
      // all augmenting paths are preferred over not augmenting.
      mcmf.AddArcWithCapacityAndUnitCost(i + 1, L + 1 + j, 1,
                                         -(weight + cardinalityBonus));
      matchingArcNodes.push_back({i, j});
    }
  }

  // Source supplies L units; SolveMaxFlowWithMinCost pushes as many as the
  // network allows (bounded by right→sink capacities, so at most min(L,R)).
  mcmf.SetNodeSupply(source, L);
  mcmf.SetNodeSupply(sink, -L);
  mcmf.SolveMaxFlowWithMinCost();

  // Extract matched pairs: matching arcs start at index L+R.
  std::vector<std::pair<SubCircuitNode, SubCircuitNode>> matching;
  const int arcBase = L + R;
  for (int k = 0; k < (int)matchingArcNodes.size(); ++k) {
    if (mcmf.Flow(arcBase + k) > 0) {
      auto [i, j] = matchingArcNodes[k];
      matching.push_back({leftNodes[i], rightNodes[j]});
    }
  }

  return matching;
}

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
      CircuitGraph &graph,
      llvm::ArrayRef<llvm::SmallVector<Operation *>> inputGroups,
      llvm::ArrayRef<llvm::SmallVector<Operation *>> outputGroups) {
    llvm::DenseSet<int64_t> inputEpochs, outputEpochs;

    // Helper: find node containing a specific operation
    auto nodeOf = [&](Operation *op) -> Node {
      for (auto &nodePtr : graph.getVertices()) {
        if (nodePtr->operations.count(op)) {
          return nodePtr;
        }
      }
      llvm::outs() << "Warning: Operation " << *op
                   << " not found in any graph node\n";
      return nullptr;
    };

    // Step 1: Clear existing epoch assignments
    for (const Node &nodePtr : graph.getVertices()) {
      nodePtr->epoch = -1;
    }

    // Step 2: Assign input groups to epochs 0, 1, 2, ...
    for (size_t i = 0; i < inputGroups.size(); ++i) {
      for (auto *op : inputGroups[i]) {
        Node node = nodeOf(op);
        if (node) {
          node->epoch = i;
          inputEpochs.insert(i);
          // llvm::outs() << "Op " << *op << " assigned to input epoch "
          //              << node->epoch << "\n";
        }
      }
    }

    // Build output node set for fast lookup during epoch assignment
    std::unordered_set<Node> outputNodes;
    for (const auto &group : outputGroups) {
      for (auto *op : group) {
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
      // llvm::outs() << "Op " << *node->operations[0]
      //              << " assigned to topoOrder epoch " << node->epoch << "\n";
    }

    // Step 5: Find max epoch and assign output groups
    int64_t maxEpoch = -1;
    for (auto &nodePtr : graph.getVertices()) {
      if (nodePtr->epoch > maxEpoch) {
        maxEpoch = nodePtr->epoch;
      }
    }

    for (size_t i = 0; i < outputGroups.size(); ++i) {
      for (auto *op : outputGroups[i]) {
        Node node = nodeOf(op);
        if (node) {
          node->epoch = maxEpoch + 1 + i;
          outputEpochs.insert(maxEpoch + 1 + i);
          // llvm::outs() << "Op " << *op << " assigned to output epoch "
          //              << node->epoch << "\n";
        }
      }
    }

    return {inputEpochs, outputEpochs};
  }
};

/// Standalone function wrapper for easier integration
std::pair<llvm::DenseSet<int64_t>, llvm::DenseSet<int64_t>> gradeGraph(
    CircuitGraph &graph,
    llvm::ArrayRef<llvm::SmallVector<Operation *>> inputGroups,
    llvm::ArrayRef<llvm::SmallVector<Operation *>> outputGroups) {
  EpochAssigner assigner;
  return assigner.assignEpochs(graph, inputGroups, outputGroups);
}

}  // namespace heir
}  // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::heir::SubCircuitNode> {
  using T = mlir::heir::SubCircuitNodeImpl;
  using Ptr = mlir::heir::SubCircuitNode;
  using PtrInfo = DenseMapInfo<T *>;

  static Ptr getEmptyKey() {
    return Ptr(PtrInfo::getEmptyKey(), [](T *) {});
  }
  static Ptr getTombstoneKey() {
    return Ptr(PtrInfo::getTombstoneKey(), [](T *) {});
  }
  static unsigned getHashValue(const Ptr &val) {
    return PtrInfo::getHashValue(val.get());
  }
  static bool isEqual(const Ptr &lhs, const Ptr &rhs) {
    return lhs.get() == rhs.get();
  }
};
}  // namespace llvm

#endif  // LIB_TRANSFORMS_COYOTEVECTORIZER_MAXWEIGHTBIPARTITEMATCHING_H_
