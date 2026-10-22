"""
Operations Research Multi-Agent System
======================================
Four specialist agents + one coordinator agent:

  1. NorthWestCornerAgent   - Transportation IBFS via North-West Corner Rule
  2. LeastCostAgent         - Transportation IBFS via Least Cost (Matrix Minima) Method
  3. VogelAgent             - Transportation IBFS via Vogel's Approximation Method (VAM)
  4. SchedulingAgent        - Job sequencing via Johnson's Rule (n jobs, 2 or 3 machines)
  5. CoordinatorAgent       - Perceives the problem, routes it to the right agent(s),
                              compares transportation results, and reports the best plan.

Every agent follows the classic agent loop:  perceive() -> decide()/act() -> report()
Pure Python, no external dependencies. Run:  python or_agents.py
"""

from abc import ABC, abstractmethod
from copy import deepcopy


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------
class Agent(ABC):
    def __init__(self, name):
        self.name = name
        self.log = []          # step-by-step reasoning trace
        self.result = None

    def think(self, msg):
        self.log.append(msg)

    @abstractmethod
    def perceive(self, problem):
        """Validate and prepare the problem (the agent's 'observation')."""

    @abstractmethod
    def act(self):
        """Run the technique and store self.result."""

    def run(self, problem, verbose=True):
        self.log, self.result = [], None
        self.perceive(problem)
        self.act()
        if verbose:
            self.report()
        return self.result

    def report(self):
        print(f"\n{'=' * 70}\n[{self.name}]\n{'=' * 70}")
        for i, line in enumerate(self.log, 1):
            print(f"  Step {i:>2}: {line}")


# ---------------------------------------------------------------------------
# Transportation base agent (shared logic for NWC, LCM, VAM)
# ---------------------------------------------------------------------------
class TransportationAgent(Agent):
    method = "Transportation"

    def perceive(self, problem):
        cost = deepcopy(problem["cost"])
        supply = list(problem["supply"])
        demand = list(problem["demand"])
        self.src = list(problem.get("sources", [f"S{i+1}" for i in range(len(supply))]))
        self.dst = list(problem.get("destinations", [f"D{j+1}" for j in range(len(demand))]))

        if len(cost) != len(supply) or any(len(r) != len(demand) for r in cost):
            raise ValueError("Cost matrix shape must be len(supply) x len(demand)")

        ts, td = sum(supply), sum(demand)
        if ts > td:                       # add dummy destination
            for r in cost:
                r.append(0)
            demand.append(ts - td)
            self.dst.append("Dummy")
            self.think(f"Unbalanced (supply {ts} > demand {td}): added dummy destination of {ts - td}.")
        elif td > ts:                     # add dummy source
            cost.append([0] * len(demand))
            supply.append(td - ts)
            self.src.append("Dummy")
            self.think(f"Unbalanced (demand {td} > supply {ts}): added dummy source of {td - ts}.")
        else:
            self.think(f"Problem is balanced (total = {ts}).")

        self.cost, self.supply, self.demand = cost, supply, demand
        self.m, self.n = len(supply), len(demand)

    def allocate(self, i, j, s, d, alloc):
        q = min(s[i], d[j])
        alloc[i][j] = q
        s[i] -= q
        d[j] -= q
        self.think(f"Allocate {q} to ({self.src[i]} -> {self.dst[j]}) @ cost {self.cost[i][j]}.")
        return q

    def finish(self, alloc):
        total = sum(alloc[i][j] * self.cost[i][j]
                    for i in range(self.m) for j in range(self.n) if alloc[i][j])
        occupied = sum(1 for i in range(self.m) for j in range(self.n) if alloc[i][j] > 0)
        needed = self.m + self.n - 1
        degenerate = occupied < needed
        self.think(f"Occupied cells = {occupied}, required (m+n-1) = {needed} -> "
                   f"{'DEGENERATE' if degenerate else 'non-degenerate'}.")
        self.think(f"Total transportation cost = {total}.")
        self.result = {"method": self.method, "allocation": alloc, "total_cost": total,
                       "degenerate": degenerate, "sources": self.src, "destinations": self.dst,
                       "cost": self.cost}

    def report(self):
        super().report()
        r = self.result
        width = max(8, max(len(d) for d in r["destinations"]) + 2)
        print("\n  Allocation matrix (allocation@cost):")
        print("  " + " " * 8 + "".join(f"{d:>{width + 4}}" for d in r["destinations"]))
        for i, row in enumerate(r["allocation"]):
            cells = "".join(
                f"{(str(q) + '@' + str(r['cost'][i][j])) if q else '-':>{width + 4}}"
                for j, q in enumerate(row))
            print(f"  {r['sources'][i]:<8}{cells}")
        print(f"\n  >> {self.method} total cost = {r['total_cost']}")


# ---------------------------------------------------------------------------
# Agent 1: North-West Corner Rule
# ---------------------------------------------------------------------------
class NorthWestCornerAgent(TransportationAgent):
    method = "North-West Corner"

    def __init__(self):
        super().__init__("Agent 1 - North-West Corner Rule")

    def act(self):
        s, d = self.supply[:], self.demand[:]
        alloc = [[0] * self.n for _ in range(self.m)]
        i = j = 0
        while i < self.m and j < self.n:
            self.allocate(i, j, s, d, alloc)
            if s[i] == 0 and d[j] == 0:
                self.think(f"Both {self.src[i]} and {self.dst[j]} exhausted -> move diagonally.")
                i, j = i + 1, j + 1
            elif s[i] == 0:
                i += 1
            else:
                j += 1
        self.finish(alloc)


# ---------------------------------------------------------------------------
# Agent 2: Least Cost Method
# ---------------------------------------------------------------------------
class LeastCostAgent(TransportationAgent):
    method = "Least Cost"

    def __init__(self):
        super().__init__("Agent 2 - Least Cost Method")

    def act(self):
        s, d = self.supply[:], self.demand[:]
        alloc = [[0] * self.n for _ in range(self.m)]
        rows, cols = set(range(self.m)), set(range(self.n))
        while rows and cols:
            # cheapest open cell; tie -> larger possible allocation
            i, j = min(((i, j) for i in rows for j in cols),
                       key=lambda c: (self.cost[c[0]][c[1]], -min(s[c[0]], d[c[1]])))
            self.think(f"Cheapest open cell is ({self.src[i]}, {self.dst[j]}) = {self.cost[i][j]}.")
            self.allocate(i, j, s, d, alloc)
            if s[i] == 0:
                rows.discard(i)
            if d[j] == 0:
                cols.discard(j)
        self.finish(alloc)


# ---------------------------------------------------------------------------
# Agent 3: Vogel's Approximation Method
# ---------------------------------------------------------------------------
class VogelAgent(TransportationAgent):
    method = "Vogel's Approximation"

    def __init__(self):
        super().__init__("Agent 3 - Vogel's Approximation Method")

    @staticmethod
    def _penalty(values):
        v = sorted(values)
        return v[1] - v[0] if len(v) > 1 else v[0]

    def act(self):
        s, d = self.supply[:], self.demand[:]
        alloc = [[0] * self.n for _ in range(self.m)]
        rows, cols = set(range(self.m)), set(range(self.n))
        while rows and cols:
            if len(rows) == 1 or len(cols) == 1:
                # only one line left: allocate remaining cells by least cost
                for i, j in sorted(((i, j) for i in rows for j in cols),
                                   key=lambda c: self.cost[c[0]][c[1]]):
                    if s[i] and d[j]:
                        self.allocate(i, j, s, d, alloc)
                break

            rp = {i: self._penalty([self.cost[i][j] for j in cols]) for i in rows}
            cp = {j: self._penalty([self.cost[i][j] for i in rows]) for j in cols}
            self.think("Penalties  rows: " + ", ".join(f"{self.src[i]}={p}" for i, p in sorted(rp.items()))
                       + " | cols: " + ", ".join(f"{self.dst[j]}={p}" for j, p in sorted(cp.items())))

            # candidate lines with max penalty; tie -> min cost cell, then max allocation
            best_p = max(max(rp.values()), max(cp.values()))
            candidates = []
            for i, p in rp.items():
                if p == best_p:
                    j = min(cols, key=lambda c: self.cost[i][c])
                    candidates.append((i, j, f"row {self.src[i]}"))
            for j, p in cp.items():
                if p == best_p:
                    i = min(rows, key=lambda r: self.cost[r][j])
                    candidates.append((i, j, f"column {self.dst[j]}"))
            i, j, line = min(candidates, key=lambda c: (self.cost[c[0]][c[1]], -min(s[c[0]], d[c[1]])))
            self.think(f"Highest penalty {best_p} in {line}; its least-cost cell is "
                       f"({self.src[i]}, {self.dst[j]}).")
            self.allocate(i, j, s, d, alloc)
            if s[i] == 0:
                rows.discard(i)
            elif d[j] == 0:
                cols.discard(j)
            if d[j] == 0:
                cols.discard(j)
        self.finish(alloc)


# ---------------------------------------------------------------------------
# Agent 4: Scheduling (Johnson's Rule)
# ---------------------------------------------------------------------------
class SchedulingAgent(Agent):
    """
    Johnson's Rule for n jobs through 2 machines (M1 -> M2).
    3-machine problems (M1 -> M2 -> M3) are reduced to 2 machines when
    min(M1) >= max(M2) or min(M3) >= max(M2), using G = M1+M2, H = M2+M3.
    """

    def __init__(self):
        super().__init__("Agent 4 - Scheduling (Johnson's Rule)")

    def perceive(self, problem):
        self.jobs = list(problem["jobs"])
        self.times = [list(t) for t in problem["times"]]   # times[k][job] for machine k
        self.k = len(self.times)
        if self.k not in (2, 3):
            raise ValueError("SchedulingAgent supports 2 or 3 machines.")
        if any(len(t) != len(self.jobs) for t in self.times):
            raise ValueError("Each machine needs one time per job.")
        self.think(f"{len(self.jobs)} jobs, {self.k} machines (order M1 -> ... -> M{self.k}).")

        if self.k == 2:
            self.a, self.b = self.times
        else:
            m1, m2, m3 = self.times
            if min(m1) >= max(m2) or min(m3) >= max(m2):
                self.think(f"Condition met (min M1={min(m1)}, min M3={min(m3)}, max M2={max(m2)}) "
                           f"-> convert to 2 fictitious machines G=M1+M2, H=M2+M3.")
                self.a = [x + y for x, y in zip(m1, m2)]
                self.b = [y + z for y, z in zip(m2, m3)]
            else:
                raise ValueError("3-machine problem doesn't satisfy Johnson's conversion condition.")

    def act(self):
        remaining = set(range(len(self.jobs)))
        front, back = [], []
        while remaining:
            j = min(remaining, key=lambda x: (min(self.a[x], self.b[x]), x))
            if self.a[j] <= self.b[j]:
                front.append(j)
                self.think(f"Min time {self.a[j]} for {self.jobs[j]} is on first machine -> place EARLY.")
            else:
                back.insert(0, j)
                self.think(f"Min time {self.b[j]} for {self.jobs[j]} is on last machine -> place LATE.")
            remaining.remove(j)
        order = front + back

        # Simulate the schedule on the real machines
        finish = [0] * self.k
        timeline, idle = [], [0] * self.k
        for j in order:
            row = []
            for m in range(self.k):
                start = max(finish[m], finish[m - 1] if m else 0)
                idle[m] += start - finish[m]
                finish[m] = start + self.times[m][j]
                row.append((start, finish[m]))
            timeline.append((self.jobs[j], row))
        makespan = finish[-1]
        idle = [idle[m] + (makespan - finish[m]) for m in range(self.k)]   # trailing idle
        self.think(f"Optimal sequence: {' -> '.join(self.jobs[j] for j in order)}.")
        self.think(f"Total elapsed time (makespan) = {makespan}.")
        self.result = {"sequence": [self.jobs[j] for j in order], "makespan": makespan,
                       "idle": idle, "timeline": timeline}

    def report(self):
        super().report()
        r = self.result
        print("\n  Time-in / time-out table:")
        print("  " + f"{'Job':<6}" + "".join(f"{'M' + str(m + 1) + ' in-out':>14}" for m in range(self.k)))
        for job, row in r["timeline"]:
            print("  " + f"{job:<6}" + "".join(f"{f'{s}-{e}':>14}" for s, e in row))
        print("\n  Idle time: " + ", ".join(f"M{m + 1}={t}" for m, t in enumerate(r["idle"])))
        print(f"  >> Makespan = {r['makespan']}")


# ---------------------------------------------------------------------------
# Agent 5: Coordinator
# ---------------------------------------------------------------------------
class CoordinatorAgent:
    def __init__(self):
        self.transport_agents = {
            "nwc": NorthWestCornerAgent(),
            "lcm": LeastCostAgent(),
            "vam": VogelAgent(),
        }
        self.scheduler = SchedulingAgent()

    def classify(self, problem):
        if {"cost", "supply", "demand"} <= problem.keys():
            return "transportation"
        if {"jobs", "times"} <= problem.keys():
            return "scheduling"
        raise ValueError("Unknown problem type.")

    def solve(self, problem, methods=("nwc", "lcm", "vam"), verbose=True):
        kind = self.classify(problem)
        print(f"\n### Coordinator: detected a {kind.upper()} problem ###")
        if kind == "scheduling":
            return self.scheduler.run(problem, verbose)

        results = {k: self.transport_agents[k].run(problem, verbose) for k in methods}
        print(f"\n{'-' * 70}\nCoordinator summary (Initial Basic Feasible Solutions)\n{'-' * 70}")
        for k, r in sorted(results.items(), key=lambda kv: kv[1]["total_cost"]):
            flag = " (degenerate)" if r["degenerate"] else ""
            print(f"  {r['method']:<24} cost = {r['total_cost']}{flag}")
        best = min(results.values(), key=lambda r: r["total_cost"])
        print(f"\n  Best starting solution: {best['method']} (cost {best['total_cost']}).")
        print("  Note: these are starting solutions; apply MODI/Stepping-Stone to confirm optimality.")
        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    coordinator = CoordinatorAgent()

    # Transportation problem (balanced, total = 34)
    transport_problem = {
        "cost": [[19, 30, 50, 10],
                 [70, 30, 40, 60],
                 [40, 8, 70, 20]],
        "supply": [7, 9, 18],
        "demand": [5, 8, 7, 14],
        "sources": ["F1", "F2", "F3"],
        "destinations": ["W1", "W2", "W3", "W4"],
    }
    coordinator.solve(transport_problem)

    # Scheduling problem: 5 jobs, 2 machines
    schedule_2m = {
        "jobs": ["J1", "J2", "J3", "J4", "J5"],
        "times": [[5, 1, 9, 3, 10],    # Machine 1
                  [2, 6, 7, 8, 4]],    # Machine 2
    }
    coordinator.solve(schedule_2m)

    # Scheduling problem: 4 jobs, 3 machines (reducible by Johnson's condition)
    schedule_3m = {
        "jobs": ["A", "B", "C", "D"],
        "times": [[8, 10, 6, 7],       # M1
                  [3, 4, 5, 2],        # M2
                  [7, 8, 9, 6]],       # M3
    }
    coordinator.solve(schedule_3m)
