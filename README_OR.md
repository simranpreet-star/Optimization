# OR Optimization Agents

A multi-agent Operations Research solver in Python. Each classical optimization technique is implemented as its own agent, and a coordinator agent detects the problem type, routes it to the right specialist(s), and compares their results.

Pure Python, no external dependencies.

---

## Agents

| Agent | Technique | Problem type |
|---|---|---|
| `NorthWestCornerAgent` | North-West Corner Rule | Transportation (initial basic feasible solution) |
| `LeastCostAgent` | Least Cost / Matrix Minima Method | Transportation (initial basic feasible solution) |
| `VogelAgent` | Vogel's Approximation Method (VAM) | Transportation (initial basic feasible solution) |
| `SchedulingAgent` | Johnson's Rule | Job sequencing: n jobs, 2 machines (or 3 machines when reducible) |
| `CoordinatorAgent` | Routing + comparison | Classifies the problem, calls the agents, ranks results |

Every agent follows the same loop:

```
perceive()  →  act()  →  report()
```

- **perceive**: validates input and prepares the problem (e.g., balances an unbalanced transportation problem with a dummy row/column)
- **act**: runs the technique and records a step-by-step reasoning log
- **report**: prints the log, the allocation or schedule table, and the final result

---

## Features

- Automatic balancing of unbalanced transportation problems (dummy source or destination)
- Degeneracy check against the `m + n − 1` rule
- Step-by-step trace for every allocation and penalty calculation
- Coordinator ranks NWC, LCM, and VAM by total cost
- Johnson's Rule with a full time-in / time-out table, idle time per machine, and makespan
- 3-machine problems reduced to 2 fictitious machines (G = M1 + M2, H = M2 + M3) when `min(M1) ≥ max(M2)` or `min(M3) ≥ max(M2)`

---

## Getting started

```bash
git clone https://github.com/simranpreet-star/or-optimization-agents.git
cd or-optimization-agents
python or_agents.py
```

Requires Python 3.8+.

---

## Usage

### Transportation problem

```python
from or_agents import CoordinatorAgent

coordinator = CoordinatorAgent()

problem = {
    "cost": [[19, 30, 50, 10],
             [70, 30, 40, 60],
             [40,  8, 70, 20]],
    "supply": [7, 9, 18],
    "demand": [5, 8, 7, 14],
    "sources": ["F1", "F2", "F3"],              # optional
    "destinations": ["W1", "W2", "W3", "W4"],   # optional
}

results = coordinator.solve(problem)                    # runs all three methods
results = coordinator.solve(problem, methods=("vam",))  # run only Vogel's
```

### Scheduling problem

```python
problem = {
    "jobs":  ["J1", "J2", "J3", "J4", "J5"],
    "times": [[5, 1, 9, 3, 10],   # Machine 1
              [2, 6, 7, 8, 4]],   # Machine 2
}

result = coordinator.solve(problem)
print(result["sequence"], result["makespan"])
```

### Using one agent directly

```python
from or_agents import VogelAgent

result = VogelAgent().run(problem, verbose=False)
print(result["total_cost"])
```

---

## Sample results

**Transportation (3 sources × 4 destinations, total = 34)**

| Method | Total cost |
|---|---|
| Vogel's Approximation | **779** |
| Least Cost | 814 |
| North-West Corner | 1015 |

**Scheduling, 5 jobs × 2 machines**

- Optimal sequence: `J2 → J4 → J3 → J5 → J1`
- Makespan: **30**
- Idle time: M1 = 2, M2 = 3

**Scheduling, 4 jobs × 3 machines (reduced via Johnson's condition)**

- Optimal sequence: `C → B → A → D`
- Makespan: **41**

Sample trace from the Vogel agent:

```
Step  2: Penalties  rows: F1=9, F2=10, F3=12 | cols: W1=21, W2=22, W3=10, W4=10
Step  3: Highest penalty 22 in column W2; its least-cost cell is (F3, W2).
Step  4: Allocate 8 to (F3 -> W2) @ cost 8.
```

---

## Project structure

```
or-optimization-agents/
├── or_agents.py   # all agents + demo
├── README.md
└── .gitignore
```

---

## Limitations

- NWC, LCM, and VAM produce **initial** basic feasible solutions; they do not prove optimality.
- Degenerate solutions are detected and flagged but not resolved (no epsilon allocation).
- The scheduling agent supports 2 machines, and 3 machines only when Johnson's reduction condition holds.
- Agents are rule-based; there is no LLM component yet.

## Roadmap

- [ ] `MODIAgent`: optimality test and improvement via the u-v (Modified Distribution) method
- [ ] Epsilon handling for degenerate solutions
- [ ] `AssignmentAgent`: Hungarian method
- [ ] LLM interface agent: parse word problems into structured input and explain results in plain language
- [ ] Unit tests

---

## Author

**Simranpreet Kaur Kang**
M.Sc. Data Science and Spatial Analytics, Symbiosis Institute of Geoinformatics, Pune
GitHub: [@simranpreet-star](https://github.com/simranpreet-star)
