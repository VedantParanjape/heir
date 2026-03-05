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

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <vector>

#include "llvm/include/llvm/ADT/EquivalenceClasses.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SetVector.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"            // from @llvm-project

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

//===----------------------------------------------------------------------===//
// MCMF: min-cost max-flow using graph::Graph<int> as the flow network
//
// Node IDs are plain ints:
//   0          = source
//   1 .. L     = left nodes  (epoch i)
//   L+1 .. L+R = right nodes (epoch j)
//   L+R+1      = sink
//
// graph::Graph<int> holds the topology (including reverse edges so SPFA can
// traverse the residual graph). Two std::maps hold the residual capacity and
// cost for each directed (u, v) pair, updated in-place during augmentation.
//===----------------------------------------------------------------------===//

class MCMF {
 public:
  static constexpr int INF = std::numeric_limits<int>::max() / 2;

  explicit MCMF(int n) {
    for (int i = 0; i < n; ++i) flowGraph.addVertex(i);
  }

  /// Add a directed edge u→v with given capacity and cost, plus the reverse
  /// edge v→u with capacity 0 and negated cost (for the residual graph).
  void addEdge(int u, int v, int cap, int cost) {
    resCap[{u, v}] += cap;
    if (!resCap.count({v, u})) resCap[{v, u}] = 0;
    edgeCost[{u, v}] = cost;
    edgeCost[{v, u}] = -cost;
    // Add both directions to the topology so edgesOutOf covers residual edges.
    if (!flowGraph.hasEdge(u, v)) flowGraph.addEdge(u, v);
    if (!flowGraph.hasEdge(v, u)) flowGraph.addEdge(v, u);
  }

  /// Run min-cost max-flow from s to t. Returns {total_flow, total_cost}.
  std::pair<int, int> minCostMaxFlow(int s, int t) {
    int totalFlow = 0, totalCost = 0;

    while (true) {
      // SPFA: shortest (min-cost) path from s to t in the residual graph.
      // Only traverse edges where resCap > 0.
      std::map<int, int> dist;
      std::map<int, bool> inQueue;
      std::map<int, int> prev;

      for (int v : flowGraph.getVertices()) {
        dist[v] = INF;
        inQueue[v] = false;
      }
      dist[s] = 0;

      std::queue<int> q;
      q.push(s);
      inQueue[s] = true;

      while (!q.empty()) {
        int u = q.front();
        q.pop();
        inQueue[u] = false;

        for (int v : flowGraph.edgesOutOf(u)) {
          if (resCap[{u, v}] > 0 && dist[u] != INF &&
              dist[u] + edgeCost[{u, v}] < dist[v]) {
            dist[v] = dist[u] + edgeCost[{u, v}];
            prev[v] = u;
            if (!inQueue[v]) {
              q.push(v);
              inQueue[v] = true;
            }
          }
        }
      }

      if (dist[t] == INF) break;  // no augmenting path — optimal

      // Augment one unit of flow along the shortest path.
      ++totalFlow;
      totalCost += dist[t];
      for (int v = t; v != s;) {
        int u = prev[v];
        resCap[{u, v}]--;
        resCap[{v, u}]++;
        v = u;
      }
    }

    return {totalFlow, totalCost};
  }

  // Residual capacity and cost for each directed (u, v) pair.
  std::map<std::pair<int, int>, int> resCap;

 private:
  graph::Graph<int> flowGraph;  // topology including reverse edges
  std::map<std::pair<int, int>, int> edgeCost;
};

//===----------------------------------------------------------------------===//
// maxWeightBipartiteMatching
//===----------------------------------------------------------------------===//

/// Compute a maximum-weight bipartite matching between leftNodes (epoch i) and
/// rightNodes (epoch j).
///
/// An edge (u, v) exists iff the circuit graph has a directed edge in either
/// direction. Its weight is:
///   getInDegree(u) + getOutDegree(u) + getInDegree(v) + getOutDegree(v)
///
/// maxcardinality=True semantics (matching Python):
///   A cardinality bonus M = sum_of_all_weights + 1 is added to each matching
///   edge cost so SPFA always prefers augmenting. Weight breaks ties.
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

  MCMF mcmf(L + R + 2);

  // source → left nodes
  for (int i = 0; i < L; ++i) mcmf.addEdge(source, i + 1, 1, 0);

  // right nodes → sink
  for (int j = 0; j < R; ++j) mcmf.addEdge(L + 1 + j, sink, 1, 0);

  // Compute cardinality bonus: M > any single edge weight so that
  // maximising cardinality always dominates maximising weight.
  // O(L+R) bound: 2 * maxDeg * min(L,R) + 1 exceeds the total weight of any
  // matching (at most min(L,R) edges, each with weight at most 2*maxDeg).
  int maxDeg = 0;
  for (auto &n : leftNodes)
    maxDeg = std::max(maxDeg, (int)(circuitGraph.getInDegree(n) +
                                    circuitGraph.getOutDegree(n)));
  for (auto &n : rightNodes)
    maxDeg = std::max(maxDeg, (int)(circuitGraph.getInDegree(n) +
                                    circuitGraph.getOutDegree(n)));
  const int cardinalityBonus = 2 * maxDeg * std::min(L, R) + 1;

  // left → right edges
  for (int i = 0; i < L; ++i) {
    const SubCircuitNode &u = leftNodes[i];
    int degU = circuitGraph.getInDegree(u) + circuitGraph.getOutDegree(u);

    for (int j = 0; j < R; ++j) {
      const SubCircuitNode &v = rightNodes[j];

      if (!circuitGraph.hasEdge(u, v) && !circuitGraph.hasEdge(v, u)) continue;

      if (isMatchable && !isMatchable(u, v)) continue;

      int degV = circuitGraph.getInDegree(v) + circuitGraph.getOutDegree(v);
      int weight = degU + degV;
      // Negative cost so SPFA prefers this edge; cardinality bonus ensures
      // all augmenting paths are preferred over not augmenting.
      mcmf.addEdge(i + 1, L + 1 + j, 1, -(weight + cardinalityBonus));
    }
  }

  mcmf.minCostMaxFlow(source, sink);

  // Extract matched pairs: iterate only edges that were actually added,
  // selecting left→right entries where residual capacity == 0 (flow sent).
  std::vector<std::pair<SubCircuitNode, SubCircuitNode>> matching;
  for (auto &[edge, cap] : mcmf.resCap) {
    int u = edge.first, v = edge.second;
    if (u >= 1 && u <= L && v >= L + 1 && v <= L + R && cap == 0)
      matching.push_back({leftNodes[u - 1], rightNodes[v - L - 1]});
  }

  return matching;
}

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_COYOTEVECTORIZER_MAXWEIGHTBIPARTITEMATCHING_H_
