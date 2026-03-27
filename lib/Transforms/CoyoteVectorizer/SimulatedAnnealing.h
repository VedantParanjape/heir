//===- QuotientSearch.cpp - Circuit quotient search with edge grouping ---===//
//
// Port of Python's pq_relax_schedule function - best-first search over
// circuit quotients with edge grouping by rotation amount.
//
//===----------------------------------------------------------------------===//

#ifndef SIMULATED_ANNEALING_H
#define SIMULATED_ANNEALING_H

#include <map>
#include <optional>
#include <queue>
#include <random>
#include <set>

#include "GraphUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// Cost Functions and Lane Placement
//===----------------------------------------------------------------------===//

/// Compute rotation cost for a graph
/// Python equivalent: rotation_cost() in search/protoschedule.py
///
/// Rotations are expensive operations in HE that implement cross-lane data
/// movement. Each unique rotation amount requires a separate rotate
/// instruction.
///
/// Cost model:
/// - Group edges by their epoch (time step)
/// - For each epoch, count unique rotation amounts needed
/// - Rotation amount = difference in column (lane) numbers between
/// producer/consumer
///
/// Example: If epoch 0 has edges with spans [0, 1, 1, 2], rotation cost = 2
///          (need rotations by 1 and 2, but not 0 since that's same-lane)
double computeRotationCost(const CircuitGraph& graph) {
  // Map: epoch -> set of unique rotation amounts needed in that epoch
  static std::unordered_map<int64_t, std::unordered_set<int64_t>>
      rotationsPerEpoch;
  rotationsPerEpoch.clear();

  for (auto& [epoch, spans] : rotationsPerEpoch) spans.clear();

  // For each edge (producer -> consumer), compute the "span" (lane difference)
  for (const auto& [u, v] : graph.getEdges()) {
    int64_t span = v->column - u->column;

    // Only count non-zero spans (same-lane communication is free)
    if (span != 0) {
      rotationsPerEpoch[u->epoch].insert(span);
    }
  }

  // Total cost = sum of unique rotation counts across all epochs
  double cost = 0;
  for (const auto& [epoch, rotations] : rotationsPerEpoch) {
    cost += rotations.size();
  }

  return cost;
}

/// Compute schedule height weighted by operation type.
/// Python equivalent: schedule_height() in search/protoschedule.py
///
///   COSTS_PER_ROTATE = {'+': 0.1, '*': 1, '-': 0.1}
///
///   For each epoch, build a Counter of op types per column, then take the
///   union (element-wise max) across all columns within that epoch.
///   Sum weighted costs from the per-epoch maxes.
///
/// This matches Python's `reduce(lambda x, y: x | y, counters)` which uses
/// Counter.__or__ (element-wise max), NOT Counter.__add__ (sum).
double computeScheduleHeight(const CircuitGraph& graph) {
  enum class OpKind : uint8_t { Mul, AddSub, Other };

  auto classify = [](Operation* op) -> OpKind {
    if (isa<arith::MulIOp, arith::MulFOp>(op)) return OpKind::Mul;
    if (isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp>(op))
      return OpKind::AddSub;
    return OpKind::Other;
  };

  auto kindWeight = [](OpKind k) -> double {
    if (k == OpKind::Mul) return 1.0;
    if (k == OpKind::AddSub) return 0.1;
    return 0.0;
  };

  // cells[epoch][column] → Counter of op kinds (OpKind → count)
  std::map<int64_t, std::map<int64_t, std::map<OpKind, int>>> cells;
  for (auto& nodePtr : graph.getVertices())
    for (auto* op : nodePtr->operations) {
      OpKind k = classify(op);
      if (k != OpKind::Other) cells[nodePtr->epoch][nodePtr->column][k]++;
    }

  // For each epoch, take element-wise max across columns, then sum weighted
  // costs
  double cost = 0;
  for (auto& [epoch, cols] : cells) {
    std::map<OpKind, int> epochMax;
    for (auto& [col, counter] : cols)
      for (auto& [kind, count] : counter)
        epochMax[kind] = std::max(epochMax[kind], count);
    for (auto& [kind, maxCount] : epochMax) cost += kindWeight(kind) * maxCount;
  }
  return cost;
}

/// Generate a random lane permutation candidate.
/// Python equivalent: permute() in search/lane_placement.py
///
/// Defines the SA neighborhood: pick two distinct lanes n1 and n2, pick a
/// random subset of epochs (each epoch flips a fair coin), then collect the
/// nodes currently assigned to n1 (-> s1) and n2 (-> s2) within those epochs.
///
/// Restricting the swap to a subset of epochs — rather than swapping all of
/// n1 and n2 globally — is the key insight. Nodes in the same lane may need
/// to stay together in some time steps but not others; partial swaps let the
/// SA discover those separations without breaking the whole schedule at once.
///
/// Returns true with s1/s2/n1/n2 populated on success. Returns false after
/// 100 failed attempts (degenerate graph where no valid swap exists).
static bool generatePermutation(CircuitGraph& graph, int64_t numCols,
                                int64_t numEpochs,
                                llvm::SmallVector<SubCircuitNodeImpl*>& s1,
                                llvm::SmallVector<SubCircuitNodeImpl*>& s2,
                                int64_t& n1, int64_t& n2, std::mt19937& rng) {
  std::uniform_int_distribution<int64_t> colDist(0, numCols - 1);
  std::uniform_real_distribution<double> realDist(0.0, 1.0);

  for (int attempt = 0; attempt < 100; ++attempt) {
    // Pick two distinct lanes to swap between.
    n1 = colDist(rng);
    n2 = colDist(rng);
    if (n1 == n2) continue;

    // Each epoch is included in the swap with probability 0.5.
    // This random subset is what makes each SA move fine-grained.
    std::set<int64_t> epochSubset;
    for (int64_t e = 0; e < numEpochs; ++e)
      if (realDist(rng) < 0.5) epochSubset.insert(e);

    // s1: nodes currently in lane n1 within the epoch subset (will move to n2)
    // s2: nodes currently in lane n2 within the epoch subset (will move to n1)
    s1.clear();
    s2.clear();
    for (auto& nodePtr : graph.getVertices()) {
      if (!epochSubset.count(nodePtr->epoch)) continue;
      if (nodePtr->column == n1) s1.push_back(nodePtr.get());
      if (nodePtr->column == n2) s2.push_back(nodePtr.get());
    }

    // Require at least one non-empty set so the move actually changes
    // something.
    if (!s1.empty() || !s2.empty()) return true;
  }
  return false;  // degenerate graph — no valid permutation found
}

/// Run simulated annealing for lane placement optimization.
/// Python equivalent: lane_placement() in search/lane_placement.py
///
/// GOAL: Minimize rotation cost — the number of unique cross-lane rotation
/// amounts needed per epoch. Each edge u->v with v->column != u->column
/// requires a vector rotate by (v->column - u->column), which is expensive
/// in HE. We want to reassign columns so that producer-consumer pairs share
/// a lane as often as possible.
///
/// TEMPERATURE SCHEDULE: t = t / (1 + t * beta) each round.
/// This is a fast-cooling adaptive schedule: when t is large it drops quickly
/// (broad exploration of the search space); as t approaches 0 it barely moves
/// (fine-tuning near a local optimum).
///
/// METROPOLIS CRITERION: accept a move if it improves cost, or with probability
/// exp((currentCost - newCost) / t) if it worsens cost. At high temperature
/// many uphill moves are accepted, allowing escape from local minima.
/// At low temperature almost only improvements are accepted.
///
/// BEST TRACKING: the graph's column fields are mutated in-place. We snapshot
/// the best-ever assignment and restore it at the end, so the graph reflects
/// the globally optimal solution found, not just the last accepted move.
double runLanePlacement(CircuitGraph& graph,
                        const llvm::DenseMap<Operation*, int64_t>& forceLanes,
                        double temp, double beta, unsigned rounds) {
  double currentCost = computeRotationCost(graph);

  // numCols  = number of distinct lanes (max column index + 1)
  // numEpochs = number of time steps (max epoch index + 1)
  int64_t numCols = 0, numEpochs = 0;
  for (auto& nodePtr : graph.getVertices()) {
    numCols = std::max(numCols, nodePtr->column + 1);
    numEpochs = std::max(numEpochs, nodePtr->epoch + 1);
  }

  // With a single lane all spans are 0 — nothing to optimize.
  if (numCols <= 1) return currentCost;

  // Pre-build a flat set of pinned operations for O(1) lookup.
  // A node is excluded from any swap if any of its operations are pinned,
  // because forceLanes fixes those operations to specific lanes (e.g. I/O).
  llvm::DenseSet<Operation*> pinnedOps;
  for (auto& [op, _] : forceLanes) pinnedOps.insert(op);

  // Snapshot/restore helpers: capture node->column for every node.
  // Used to remember the globally best assignment across all SA rounds.
  auto snapshotColumns = [&]() {
    llvm::DenseMap<SubCircuitNodeImpl*, int64_t> snap;
    for (auto& nodePtr : graph.getVertices())
      snap[nodePtr.get()] = nodePtr->column;
    return snap;
  };
  auto restoreColumns =
      [&](const llvm::DenseMap<SubCircuitNodeImpl*, int64_t>& snap) {
        for (auto& nodePtr : graph.getVertices())
          nodePtr->column = snap.lookup(nodePtr.get());
      };

  double bestCost = currentCost;
  auto bestColumns = snapshotColumns();

  std::mt19937 rng(1);  // fixed seed for reproducibility
  std::uniform_real_distribution<double> realDist(0.0, 1.0);

  for (unsigned i = 0; i < rounds; ++i) {
    // Cool the temperature: fast early, slow late.
    temp /= (1.0 + temp * beta);

    // Propose a swap: s1 (in lane n1) <-> s2 (in lane n2), restricted to a
    // random epoch subset. Skip the round if no valid swap exists.
    llvm::SmallVector<SubCircuitNodeImpl*> s1, s2;
    int64_t n1, n2;
    if (!generatePermutation(graph, numCols, numEpochs, s1, s2, n1, n2, rng))
      continue;

    // Filter out nodes that contain pinned operations — their lane is fixed.
    auto removePinned = [&](llvm::SmallVector<SubCircuitNodeImpl*>& s) {
      s.erase(std::remove_if(s.begin(), s.end(),
                             [&](SubCircuitNodeImpl* node) {
                               for (auto* op : node->operations)
                                 if (pinnedOps.count(op)) return true;
                               return false;
                             }),
              s.end());
    };
    removePinned(s1);
    removePinned(s2);

    // Apply the swap in-place: s1 moves to n2, s2 moves to n1.
    for (auto* node : s1) node->column = n2;
    for (auto* node : s2) node->column = n1;

    double newCost = computeRotationCost(graph);

    // Metropolis accept/reject.
    // Accept unconditionally if cost improved; otherwise accept with
    // probability exp((current - new) / t) to escape local minima.
    if (newCost < currentCost ||
        realDist(rng) < std::exp((currentCost - newCost) / temp)) {
      currentCost = newCost;
    } else {
      // Rejected — revert the swap.
      for (auto* node : s1) node->column = n1;
      for (auto* node : s2) node->column = n2;
    }

    // Track the globally best assignment seen so far.
    if (currentCost < bestCost) {
      bestCost = currentCost;
      bestColumns = snapshotColumns();
    }
  }

  // Restore the best-ever column assignment into the graph.
  restoreColumns(bestColumns);
  return bestCost;
}

//===----------------------------------------------------------------------===//
// QuotientSchedule: Represents a circuit quotient with its cost
//===----------------------------------------------------------------------===//

struct QuotientSchedule {
  double cost;
  double rotationCost;
  double heightCost;
  CircuitGraph graph;

  // Edge groups to explore (grouped by (epoch, span))
  // Each group is a set of edges that can be contracted together.
  // nullopt = not yet generated; empty vector = generated but exhausted.
  // Python: cur.edges is None vs cur.edges == []
  using EdgePair = std::pair<SubCircuitNode, SubCircuitNode>;
  std::optional<std::vector<std::set<EdgePair>>> edgeGroups;

  // For priority queue (min-heap)
  bool operator>(const QuotientSchedule& other) const {
    return cost > other.cost;
  }
};

static void printGraph(const CircuitGraph& g, llvm::StringRef label = "") {
  if (!label.empty()) llvm::errs() << "=== " << label << " ===\n";

  for (const SubCircuitNode& node :
       const_cast<CircuitGraph&>(g).getVertices()) {
    llvm::errs() << "Node(epoch=" << node->epoch << ", col=" << node->column
                 << ", ops=[";
    bool first = true;
    for (auto* op : node->operations) {
      if (!first) llvm::errs() << ", ";
      llvm::errs() << op->getName().getStringRef();
      first = false;
    }
    llvm::errs() << "])\n";

    for (const SubCircuitNode& succ :
         const_cast<CircuitGraph&>(g).edgesOutOf(node)) {
      llvm::errs() << "  -> Node(epoch=" << succ->epoch
                   << ", col=" << succ->column << ")\n";
    }
  }
  llvm::errs() << "\n";
}

//===----------------------------------------------------------------------===//
// QuotientSearcher: Best-first search over circuit quotients
//===----------------------------------------------------------------------===//

class QuotientSearcher {
 public:
  using Node = SubCircuitNode;

  /// Search for best circuit quotient
  /// Python: protoschedule.py:49-247 (pq_relax_schedule)
  QuotientSchedule search(
      CircuitGraph& initialGraph,
      llvm::ArrayRef<llvm::SmallVector<Operation*>> inputGroups,
      llvm::ArrayRef<llvm::SmallVector<Operation*>> outputGroups,
      llvm::DenseMap<Operation*, int64_t>& forceLanes, unsigned rounds = 200) {
    // Build sets for fast lookup
    llvm::DenseSet<Operation*> inputOps, outputOps;
    for (const auto& group : inputGroups) {
      inputOps.insert(group.begin(), group.end());
    }
    for (const auto& group : outputGroups) {
      outputOps.insert(group.begin(), group.end());
    }

    // Step 1: Grade graph (assign epochs)
    gradeGraph(initialGraph, inputGroups, outputGroups);

    // Step 2: Columnize (bipartite matching for initial lanes)
    // NOTE: This should be done externally via ColumnAssigner before calling
    // search

    // Helper: pre-SA column swap.
    // Python: protoschedule.py:79-85
    //   for reg, lane in force_lanes.items():
    //       col_num = graph.nodes[reg,]["column"]
    //       new_data.update({n: lane for n, d in ... if d["column"] ==
    //       col_num})
    //   nx.set_node_attributes(graph, new_data, name="column")
    //
    // Before SA we remap entire columns: find the current column of each forced
    // op and reassign ALL nodes in that column to the forced lane.  This
    // ensures SA starts from a state that already respects force-lane
    // constraints, so it optimizes the remaining nodes relative to the correct
    // forced positions.
    auto applyForceLaneColumnSwap = [&](CircuitGraph& g) {
      // Build: current_column_of_forced_op -> forced_lane
      llvm::DenseMap<int64_t, int64_t> columnToForcedLane;
      for (auto& nodePtr : g.getVertices())
        for (auto* op : nodePtr->operations) {
          auto it = forceLanes.find(op);
          if (it != forceLanes.end())
            columnToForcedLane[nodePtr->column] = it->second;
        }
      // Reassign every node whose current column maps to a forced lane.
      for (auto& nodePtr : g.getVertices()) {
        auto it = columnToForcedLane.find(nodePtr->column);
        if (it != columnToForcedLane.end()) nodePtr->column = it->second;
      }
    };

    // Helper: re-apply forced lanes after SA.
    // Python: protoschedule.py:102-107
    //   for reg, lane in force_lanes.items():
    //       graph.nodes[reg,]["column"] = lane
    auto reapplyForceLanes = [&](CircuitGraph& g) {
      for (auto& nodePtr : g.getVertices())
        for (auto* op : nodePtr->operations)
          if (forceLanes.count(op)) nodePtr->column = forceLanes.lookup(op);
    };

    // Step 3: Run lane placement (simulated annealing)
    applyForceLaneColumnSwap(initialGraph);  // pre-SA: swap entire columns
    double rotCost =
        runLanePlacement(initialGraph, forceLanes, 50, 0.001, 20000);
    reapplyForceLanes(initialGraph);  // post-SA: re-pin individual forced ops
    double heightCost = computeScheduleHeight(initialGraph);

    // Initialize best schedule
    QuotientSchedule best;
    best.cost = rotCost + heightCost;
    best.rotationCost = rotCost;
    best.heightCost = heightCost;
    best.graph = initialGraph;  // deep copy via new copy ctor

    // Priority queue for best-first search
    std::priority_queue<QuotientSchedule, std::vector<QuotientSchedule>,
                        std::greater<QuotientSchedule>>
        pqueue;

    QuotientSchedule initial;
    initial.cost = best.cost;
    initial.rotationCost = best.rotationCost;
    initial.heightCost = best.heightCost;
    initial.graph = initialGraph;
    initial.edgeGroups = std::nullopt;  // Gap 5: nullopt = not yet generated

    pqueue.push(std::move(initial));

    // Best-first search
    unsigned saCallCount = 0;
    for (unsigned r = 0; r < rounds && !pqueue.empty(); ++r) {
      auto popStart = std::chrono::steady_clock::now();
      QuotientSchedule cur = pqueue.top();
      pqueue.pop();

      // Check if this is a new best
      if (cur.cost < best.cost) best = cur;  // deep copy via copy ctor

      // Generate edge groups lazily on first pop.
      // Gap 5: nullopt means "not yet generated" (Python: cur.edges is None)
      if (!cur.edgeGroups.has_value()) {
        cur.edgeGroups = groupCrossEdges(cur.graph, inputOps, outputOps);
        llvm::errs() << "    [pop " << r << "] generated "
                     << cur.edgeGroups->size() << " edge groups\n";
        for (size_t gi = 0; gi < cur.edgeGroups->size(); ++gi) {
          llvm::errs() << "      group " << gi << ": "
                       << (*cur.edgeGroups)[gi].size() << " edges\n";
          for (const auto& [eu, ev] : (*cur.edgeGroups)[gi]) {
            llvm::errs() << "        epoch=" << eu->epoch
                         << " col=" << eu->column << " -> epoch=" << ev->epoch
                         << " col=" << ev->column
                         << " (span=" << ev->column - eu->column << ")\n";
          }
        }
      }
      if (cur.edgeGroups->empty()) continue;  // No more edges to contract

      // Pick an edge group to contract
      std::set<QuotientSchedule::EdgePair> edgesToContract;
      while (!cur.edgeGroups->empty()) {
        edgesToContract = cur.edgeGroups->back();
        cur.edgeGroups->pop_back();

        if (respectsForceLanes(edgesToContract, forceLanes)) break;

        edgesToContract.clear();
      }

      if (edgesToContract.empty()) continue;

      // Contract edges and condense.
      // Gap 2: pass forceLanes so column restoration is correct.
      CircuitGraph contracted =
          contractAndCondense(cur.graph, edgesToContract, forceLanes);

      // Regrade after contraction
      gradeGraph(contracted, inputGroups, outputGroups);

      // After re-grading, column assignments may be stale: two nodes that were
      // at different epochs (and thus shared a column legally) may now be at
      // the same epoch after merging.  Such (epoch, column) conflicts are
      // invisible to SA (span=0 edges are skipped) and cause multiple ops at
      // the same lane, which leads to duplicate vector instructions in the
      // lowering. Fix: assign any conflicting node a fresh column beyond the
      // current max.
      {
        std::map<std::pair<int64_t, int64_t>, SubCircuitNode> seen;
        int64_t maxCol = 0;
        for (const SubCircuitNode& node : contracted.getVertices())
          maxCol = std::max(maxCol, node->column);
        for (const SubCircuitNode& node : contracted.getVertices()) {
          auto key = std::make_pair(node->epoch, node->column);
          auto [it, inserted] = seen.emplace(key, node);
          if (!inserted) {
            // Collision: give this node a fresh column.
            node->column = ++maxCol;
          }
        }
      }

      // Run lane placement on contracted graph
      applyForceLaneColumnSwap(contracted);  // pre-SA: swap entire columns
      auto saStart = std::chrono::steady_clock::now();
      double newRotCost =
          runLanePlacement(contracted, forceLanes, 50, 0.001, 20000);
      auto saMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - saStart)
                      .count();
      reapplyForceLanes(contracted);  // post-SA: re-pin individual forced ops
      double newHeightCost = computeScheduleHeight(contracted);
      double newCost = newRotCost + newHeightCost;

      ++saCallCount;
      auto popMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - popStart)
                       .count();
      llvm::errs() << "  [pop " << r << "] SA=" << saMs << "ms, total=" << popMs
                   << "ms, cost=" << newCost << " (rot=" << newRotCost
                   << " ht=" << newHeightCost
                   << "), queue=" << pqueue.size() + 1 << "\n";

      // Add contracted graph to queue
      QuotientSchedule newSchedule;
      newSchedule.cost = newCost;
      newSchedule.rotationCost = newRotCost;
      newSchedule.heightCost = newHeightCost;
      newSchedule.graph = contracted;
      newSchedule.edgeGroups = std::nullopt;  // will be generated on pop

      pqueue.push(std::move(newSchedule));

      // If there are more edge groups to explore, re-add current with
      // remaining groups (Python: heappush remaining edges variant).
      if (!cur.edgeGroups->empty()) pqueue.push(std::move(cur));
    }
    llvm::errs() << "  Quotient search: " << saCallCount << " SA calls in "
                 << rounds << " rounds\n";

    return best;
  }

 private:
  /// Group cross-lane edges by (source_epoch, rotation_span)
  /// Python: protoschedule.py:139-160
  std::vector<std::set<QuotientSchedule::EdgePair>> groupCrossEdges(
      const CircuitGraph& graph, const llvm::DenseSet<Operation*>& inputOps,
      const llvm::DenseSet<Operation*>& outputOps) {
    std::map<std::pair<int64_t, int64_t>, std::set<QuotientSchedule::EdgePair>>
        crossEdges;

    for (const auto& [u, v] : graph.getEdges()) {
      // Get source epoch and rotation span
      int64_t srcEpoch = u->epoch;
      int64_t span = v->column - u->column;

      // Skip same-lane edges (no rotation needed)
      if (span == 0) continue;

      // Skip edges between input nodes
      bool uIsInput = false, vIsInput = false;
      for (auto* op : u->operations) {
        if (inputOps.count(op)) {
          uIsInput = true;
          break;
        }
      }
      for (auto* op : v->operations) {
        if (inputOps.count(op)) {
          vIsInput = true;
          break;
        }
      }
      if (uIsInput && vIsInput) continue;

      // Skip edges between output nodes
      bool uIsOutput = false, vIsOutput = false;
      for (auto* op : u->operations) {
        if (outputOps.count(op)) {
          uIsOutput = true;
          break;
        }
      }
      for (auto* op : v->operations) {
        if (outputOps.count(op)) {
          vIsOutput = true;
          break;
        }
      }
      if (uIsOutput && vIsOutput) continue;

      // Group by (epoch, span)
      crossEdges[{srcEpoch, span}].insert(std::make_pair(u, v));
    }

    // Convert map to vector
    std::vector<std::set<QuotientSchedule::EdgePair>> result;
    for (auto& [key, edges] : crossEdges) {
      result.push_back(std::move(edges));
    }

    return result;
  }

  /// Contract edge group and condense cycles, restoring correct column
  /// assignments on the condensed nodes.
  ///
  /// Python: protoschedule.py:186-205
  ///   contracted = nx.condensation(raw_contracted)
  ///   for node in contracted:
  ///     fixed = members.intersection(force_lanes.keys())
  ///     if fixed: col = force_lanes[next(iter(fixed))]
  ///     else:     col = raw_contracted.nodes[first_member]["column"]
  /// Contract edge group and condense cycles, restoring correct column
  /// assignments on the condensed nodes.
  ///
  /// Python: protoschedule.py:186-205
  CircuitGraph contractAndCondense(
      CircuitGraph& inputGraph,
      const std::set<QuotientSchedule::EdgePair>& edges,
      const llvm::DenseMap<Operation*, int64_t>& forceLanes) {
    CircuitGraph graph = deepCopyCircuitGraph(inputGraph);

    // Map old nodes → new nodes by first Operation* (stable across deep copy)
    llvm::DenseMap<Operation*, SubCircuitNode> opToNode;
    for (const SubCircuitNode& node : graph.getVertices())
      opToNode[node->operations[0]] = node;

    // Snapshot op→column BEFORE any mutation so we can restore correct
    // column values onto the freshly-created condensed nodes afterwards.
    llvm::DenseMap<Operation*, int64_t> opToColumn;
    for (const SubCircuitNode& node : graph.getVertices())
      for (auto* op : node->operations) opToColumn[op] = node->column;

    // Merge functor: merge operations and fix column.
    auto mergeFn = [&](SubCircuitNode& a, const SubCircuitNode& b) {
      a->merge(*b);
      // Prefer a forced lane from either node.
      for (auto* op : a->operations)
        if (auto it = forceLanes.find(op); it != forceLanes.end()) {
          a->column = it->second;
          return;
        }
      // Fall back to pre-contraction column.
      if (a->column == -1)
        for (auto* op : a->operations)
          if (auto it = opToColumn.find(op); it != opToColumn.end()) {
            a->column = it->second;
            return;
          }
    };

    // Contract all edges in the group, remapping to deep copy's nodes.
    for (const auto& [oldU, oldV] : edges) {
      SubCircuitNode u = opToNode[oldU->operations[0]];
      SubCircuitNode v = opToNode[oldV->operations[0]];
      if (u == v || !graph.hasEdge(u, v)) continue;
      graph.contractEdge(u, v, mergeFn);
    }

    // Condense SCCs.
    graph.condenseGraph(mergeFn);

    return graph;
  }

  /// Check if edges respect force lane constraints
  /// Python: protoschedule.py:171-179
  bool respectsForceLanes(
      const std::set<QuotientSchedule::EdgePair>& edges,
      const llvm::DenseMap<Operation*, int64_t>& forceLanes) {
    std::set<int64_t> leftFixedLanes, rightFixedLanes;

    for (const auto& [u, v] : edges) {
      // Check if u or v contain forced operations
      for (auto* op : u->operations) {
        if (forceLanes.count(op)) {
          leftFixedLanes.insert(forceLanes.lookup(op));
        }
      }
      for (auto* op : v->operations) {
        if (forceLanes.count(op)) {
          rightFixedLanes.insert(forceLanes.lookup(op));
        }
      }
    }

    // If both sides have forced lanes, they must match
    if (!leftFixedLanes.empty() && !rightFixedLanes.empty()) {
      // All forced lanes on left must match all forced lanes on right
      return leftFixedLanes == rightFixedLanes;
    }

    return true;
  }
};

/// Standalone function wrapper
QuotientSchedule searchQuotients(
    CircuitGraph& initialGraph,
    llvm::ArrayRef<llvm::SmallVector<Operation*>> inputGroups,
    llvm::ArrayRef<llvm::SmallVector<Operation*>> outputGroups,
    llvm::DenseMap<Operation*, int64_t>& forceLanes, unsigned rounds = 200) {
  QuotientSearcher searcher;
  return searcher.search(initialGraph, inputGroups, outputGroups, forceLanes,
                         rounds);
}

}  // namespace heir
}  // namespace mlir

#endif  // SIMULATED_ANNEALING_H
