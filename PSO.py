"""
IEEE 33-Bus Forward-Backward Sweep + PSO (Python)
Single-file script: original, modified, and PSO-based V2G scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from copy import deepcopy

# ---------------------------
# Base & system data
# ---------------------------
# Use S_base consistent with your PSO (as earlier): 100 MVA in PSO flow
S_base_MVA = 100.0   # MVA base used for PSO-power conversion
V_base_kV = 12.66
Z_base = (V_base_kV ** 2) / S_base_MVA  # ohm base used to convert R,X to p.u.

# Line data (from your provided tables) - 1-based indices preserved here
line_data = np.array([
    [1, 1, 2, 0.0922, 0.0470],
    [2, 2, 3, 0.4930, 0.2511],
    [3, 3, 4, 0.3660, 0.1864],
    [4, 4, 5, 0.3811, 0.1941],
    [5, 5, 6, 0.8190, 0.7070],
    [6, 6, 7, 0.1872, 0.6188],
    [7, 7, 8, 1.7114, 1.2351],
    [8, 8, 9, 1.0300, 0.7400],
    [9, 9, 10, 1.0440, 0.7400],
    [10, 10, 11, 0.1966, 0.0650],
    [11, 11, 12, 0.3744, 0.1238],
    [12, 12, 13, 1.4680, 1.1550],
    [13, 13, 14, 0.5416, 0.7129],
    [14, 14, 15, 0.5910, 0.5260],
    [15, 15, 16, 0.7463, 0.5450],
    [16, 16, 17, 1.2890, 1.7210],
    [17, 17, 18, 0.7320, 0.5740],
    [18, 2, 19, 0.1640, 0.1565],
    [19, 19, 20, 1.5042, 1.3554],
    [20, 20, 21, 0.4095, 0.4784],
    [21, 21, 22, 0.7089, 0.9373],
    [22, 3, 23, 0.4512, 0.3083],
    [23, 23, 24, 0.8980, 0.7091],
    [24, 24, 25, 0.8960, 0.7011],
    [25, 6, 26, 0.2030, 0.1034],
    [26, 26, 27, 0.2842, 0.1447],
    [27, 27, 28, 1.0590, 0.9337],
    [28, 28, 29, 0.8042, 0.7006],
    [29, 29, 30, 0.5075, 0.2585],
    [30, 30, 31, 0.9744, 0.9630],
    [31, 31, 32, 0.3105, 0.3619],
    [32, 32, 33, 0.3410, 0.5302],
])

num_lines = line_data.shape[0]
num_buses = 33

# Extract arrays with 0-based indexes for Python internal arrays
from_bus = (line_data[:, 1].astype(int) - 1).astype(int)
to_bus   = (line_data[:, 2].astype(int) - 1).astype(int)
R_ohm    = line_data[:, 3].astype(float)
X_ohm    = line_data[:, 4].astype(float)

# Convert R and X to per-unit (p.u.) using Z_base
R_pu = R_ohm / Z_base
X_pu = X_ohm / Z_base
Z_pu = R_pu + 1j * X_pu

# ---------------------------
# Load data (original) - from your first block (kW / kVAR)
# ---------------------------
load_data_original = np.array([
    [1,   0,   0],
    [2, 100,  60],
    [3,  90,  40],
    [4, 120,  80],
    [5,  60,  30],
    [6,  60,  20],
    [7, 200, 100],
    [8, 200, 100],
    [9,  60,  20],
    [10, 60,  20],
    [11, 45,  30],
    [12, 60,  35],
    [13, 60,  35],
    [14, 120, 80],
    [15, 60,  10],
    [16, 60,  20],
    [17, 60,  20],
    [18, 90,  40],
    [19, 90,  40],
    [20, 90,  40],
    [21, 90,  40],
    [22, 90,  40],
    [23, 90,  50],
    [24, 420, 200],
    [25, 420, 200],
    [26, 60,  25],
    [27, 60,  25],
    [28, 60,  20],
    [29, 120, 70],
    [30, 200, 600],
    [31, 150, 70],
    [32, 210, 100],
    [33, 60,  40],
])

# Convert P & Q (kW / kVAR) to per-unit on S_base_MVA
P_pu_base = np.zeros(num_buses)
Q_pu_base = np.zeros(num_buses)
for r in load_data_original:
    bus_idx = int(r[0]) - 1
    P_pu_base[bus_idx] = r[1] / (S_base_MVA * 1000)  # kW -> MW -> p.u.
    Q_pu_base[bus_idx] = r[2] / (S_base_MVA * 1000)

# ---------------------------
# Modified load scenario (from your long message - picks a modified set)
# The MATLAB snippet includes multiple modified blocks; we pick the "modified" changes
# described in the user's message: e.g., bus 3=270, bus18=330, bus32=300, etc.
# ---------------------------
P_pu_modified = P_pu_base.copy()
Q_pu_modified = Q_pu_base.copy()

# Apply modifications you included in the message (examples)
# - second block: bus 3 changed to 270 kW, bus 18 changed to 330 kW, bus 32 changed to 300 kW
# We'll implement those exact numbers.
P_pu_modified[2]  = 270 / (S_base_MVA * 1000)   # bus 3
Q_pu_modified[2]  = 99.16 / (S_base_MVA * 1000) # approx from your block
P_pu_modified[17] = 330 / (S_base_MVA * 1000)   # bus 18
Q_pu_modified[17] = 118.88 / (S_base_MVA * 1000)
P_pu_modified[31] = 300 / (S_base_MVA * 1000)   # bus 32
Q_pu_modified[31] = 129.58 / (S_base_MVA * 1000)

# ---------------------------
# Helpers: Build children lists (1-based bus numbering mapped to 0-based)
# ---------------------------
children = [[] for _ in range(num_buses)]
for i in range(num_lines):
    p = from_bus[i]
    c = to_bus[i]
    children[p].append(c)

# ---------------------------
# Forward-Backward Sweep function (works in p.u.)
# Returns bus voltages V (complex) and branch currents I_branch (complex, length num_lines)
# ---------------------------
def forward_backward_sweep(S_bus, children, from_bus, to_bus, R_pu, X_pu, tol=1e-6, max_iter=200):
    """
    S_bus: complex injections at buses in p.u. (S = P + jQ), length num_buses
    children/from_bus/to_bus/R_pu/X_pu: network parameters
    """
    nb = len(S_bus)
    V = np.ones(nb, dtype=complex)  # initial voltages (1.0 pu)
    V[0] = 1.0 + 0j  # slack bus (bus 1) fixed at 1∠0
    I = np.zeros(nb, dtype=complex)  # current injections; will be overwritten
    I_branch = np.zeros(len(from_bus), dtype=complex)

    for it in range(max_iter):
        V_prev = V.copy()

        # Backward sweep: compute branch currents from leaves to root
        # First compute load/injection currents at buses (conj(S/V))
        I_inj = np.zeros(nb, dtype=complex)
        for b in range(nb):
            if abs(V[b]) > 1e-12:
                I_inj[b] = np.conj(S_bus[b] / V[b])
            else:
                I_inj[b] = 0.0 + 0.0j

        # Initialize all branch currents to zero
        I_branch = np.zeros(len(from_bus), dtype=complex)

        # We'll accumulate downstream currents by processing lines from last to first (leaf-most lines typically later)
        # But to be safe for arbitrary ordering, we do repeated accumulation until convergence or by recursion.
        # Simpler robust approach: for each line, compute current as injection at 'to' + sum of branch currents of children lines
        # We need an ordering such that children are processed before their parent lines.
        # We'll perform a post-order traversal starting from root (bus 0) and compute subtree currents.

        def subtree_current(bus):
            # current contribution from bus subtree (including bus injection)
            total = I_inj[bus]
            for child in children[bus]:
                # find index of branch bus->child
                idx = np.where((from_bus == bus) & (to_bus == child))[0]
                if idx.size == 0:
                    continue
                idx = idx[0]
                child_current = subtree_current(child)
                I_branch[idx] = child_current
                total += child_current
            return total

        # call for root (bus 0)
        _ = subtree_current(0)

        # Forward sweep: update bus voltages
        V[0] = 1.0 + 0j
        for i_line in range(len(from_bus)):
            fb = from_bus[i_line]
            tb = to_bus[i_line]
            z = R_pu[i_line] + 1j * X_pu[i_line]
            V[tb] = V[fb] - z * I_branch[i_line]

        # convergence check
        max_diff = np.max(np.abs(V - V_prev))
        if max_diff < tol:
            # print("Converged in {} iterations (tol {})".format(it+1, tol))
            break

    return V, I_branch

# ---------------------------
# Loss calculation
# ---------------------------
def calculate_line_losses(I_branch, R_pu, X_pu, S_base_MVA):
    # returns per-line P_loss_kW array and Q_loss_kvar array
    P_loss_pu = R_pu * (np.abs(I_branch) ** 2)
    Q_loss_pu = X_pu * (np.abs(I_branch) ** 2)
    P_loss_kW = P_loss_pu * S_base_MVA * 1000.0
    Q_loss_kvar = Q_loss_pu * S_base_MVA * 1000.0
    return P_loss_kW, Q_loss_kvar

# ---------------------------
# Scenario runs: Original, Modified
# ---------------------------
def make_S_from_PQ(P_pu, Q_pu):
    """Return complex S vector (p.u.) of length num_buses (S = P + jQ)."""
    return P_pu + 1j * Q_pu

S_original = make_S_from_PQ(P_pu_base, Q_pu_base)
S_modified = make_S_from_PQ(P_pu_modified, Q_pu_modified)

# run FBS for original and modified
V_orig, Ibranch_orig = forward_backward_sweep(S_original, children, from_bus, to_bus, R_pu, X_pu)
V_mod,  Ibranch_mod  = forward_backward_sweep(S_modified, children, from_bus, to_bus, R_pu, X_pu)

P_loss_line_orig, Q_loss_line_orig = calculate_line_losses(Ibranch_orig, R_pu, X_pu, S_base_MVA)
P_loss_line_mod,  Q_loss_line_mod  = calculate_line_losses(Ibranch_mod,  R_pu, X_pu, S_base_MVA)

total_P_loss_orig = np.sum(P_loss_line_orig)
total_P_loss_mod  = np.sum(P_loss_line_mod)

# ---------------------------
# PSO to place 3 V2G units (embed PSO)
# ---------------------------
# Use same v2g_power as your earlier snippet: values in MW were divided by 48*1000 in that code.
v2g_MW = np.array([33.7127, 16.3806, 25.4608]) / 48.0  # values in MW (original used / (48*1000) because they converted to pu with S=100)
# Convert MW to per-unit on S_base_MVA
v2g_pu = v2g_MW / S_base_MVA  # v2g injection in p.u. (positive = injection reduces load)

# Regions (1-based in MATLAB): [2:11], [12:22], [23:33] -> convert to 0-based indices here
regions = [list(range(1, 11)), list(range(11, 22)), list(range(22, 33))]  # 0-based region lists

# PSO parameters
random.seed(0)
np.random.seed(0)
n_particles = 8
max_iterations = 20
c1 = 1.5
c2 = 1.5
inertia_weight = 0.7

# Particle arrays
positions = np.zeros((n_particles, 3), dtype=int)
velocities = np.zeros((n_particles, 3), dtype=float)
pbest = np.zeros_like(positions)
pbest_loss = np.full(n_particles, np.inf)
gbest = np.zeros(3, dtype=int)
gbest_loss = np.inf

# initialize particles - pick one bus from each region, ensure uniqueness
for p in range(n_particles):
    valid = False
    while not valid:
        for r in range(3):
            positions[p, r] = random.choice(regions[r])
        valid = (len(np.unique(positions[p])) == 3)

# PSO main loop
loss_history = np.zeros(max_iterations)
tol = 1e-6

def evaluate_positions(pos):
    # pos: array-like of three bus indices (0-based)
    # build S_temp and run FBS to compute losses
    if len(np.unique(pos)) < 3:
        return np.inf
    S_temp = S_original.copy()
    # At each v2g bus subtract injection (since original S is load positive), i.e., reduce load or inject negative
    for i in range(3):
        b = pos[i]
        S_temp[b] -= v2g_pu[i]
    V_tmp, Ibranch_tmp = forward_backward_sweep(S_temp, children, from_bus, to_bus, R_pu, X_pu, tol=1e-6)
    P_loss_tmp, Q_loss_tmp = calculate_line_losses(Ibranch_tmp, R_pu, X_pu, S_base_MVA)
    return np.sum(P_loss_tmp), V_tmp, Ibranch_tmp

print("Starting PSO for V2G placement...")
for it in range(max_iterations):
    for p in range(n_particles):
        loss_val, Vtmp, Itmp = evaluate_positions(positions[p])
        if loss_val < pbest_loss[p]:
            pbest_loss[p] = loss_val
            pbest[p] = positions[p].copy()
        if loss_val < gbest_loss:
            gbest_loss = loss_val
            gbest = positions[p].copy()
    # update velocities & positions
    for p in range(n_particles):
        r1 = np.random.rand(3)
        r2 = np.random.rand(3)
        velocities[p] = (inertia_weight * velocities[p] +
                         c1 * r1 * (pbest[p] - positions[p]) +
                         c2 * r2 * (gbest - positions[p]))
        positions[p] = np.round(positions[p] + velocities[p]).astype(int)
        # clamp into region bounds and enforce that each chosen bus belongs to its region
        for r in range(3):
            region = regions[r]
            lo = min(region); hi = max(region)
            positions[p, r] = max(min(positions[p, r], hi), lo)
        # enforce uniqueness by random reassign if duplicates
        while len(np.unique(positions[p])) < 3:
            for r in range(3):
                positions[p, r] = random.choice(regions[r])
    loss_history[it] = gbest_loss
    if (it % 5 == 0) or (it == max_iterations - 1):
        print(f"Iter {it+1}/{max_iterations} | Best Loss = {gbest_loss:.4f} kW")

print("PSO complete.")
print("GBest (0-based buses):", gbest, " -> (1-based):", gbest + 1)
print("Best loss (kW):", gbest_loss)

# Compute final optimized (V2G applied) run with gbest
S_pso = S_original.copy()
for i in range(3):
    S_pso[gbest[i]] -= v2g_pu[i]
V_pso, Ibranch_pso = forward_backward_sweep(S_pso, children, from_bus, to_bus, R_pu, X_pu)
P_loss_line_pso, Q_loss_line_pso = calculate_line_losses(Ibranch_pso, R_pu, X_pu, S_base_MVA)
total_P_loss_pso = np.sum(P_loss_line_pso)

# ---------------------------
# Print results summary (Original, Modified, PSO)
# ---------------------------
def print_summary(name, V, Ibranch, P_loss_line, Q_loss_line):
    print("\n==========================")
    print(f"Scenario: {name}")
    print("==========================")
    print(f"Total real power loss: {np.sum(P_loss_line):.4f} kW")
    print(f"Total reactive power loss: {np.sum(Q_loss_line):.4f} kVAR")
    print("\nBus Voltage Profile (pu):")
    for i in range(num_buses):
        print(f"Bus {i+1:2d}: V = {abs(V[i]):.6f} pu, angle = {np.angle(V[i], deg=True):8.4f} deg")
    print("\nBranch currents (pu):")
    for i in range(num_lines):
        print(f"Line {i+1:2d} ({from_bus[i]+1}->{to_bus[i]+1}): I = {abs(Ibranch[i]):.6f} pu, angle = {np.angle(Ibranch[i], deg=True):8.4f} deg")
    # voltage violations
    v_viol = np.where(abs(V) < 0.95)[0]
    if v_viol.size > 0:
        print("\nVoltage violations (V < 0.95 pu) at buses:", (v_viol + 1).tolist())
    else:
        print("\nNo voltage violations (all >= 0.95 pu).")
    print("==========================\n")

print_summary("Original", V_orig, Ibranch_orig, P_loss_line_orig, Q_loss_line_orig)
print_summary("Modified", V_mod,  Ibranch_mod,  P_loss_line_mod,  Q_loss_line_mod)
print_summary("PSO-V2G", V_pso,  Ibranch_pso,  P_loss_line_pso,  Q_loss_line_pso)

# ---------------------------
# Plotting - ensure shapes match!
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(range(1, num_buses+1), np.abs(V_orig), 'r--s', label='Original')
plt.plot(range(1, num_buses+1), np.abs(V_mod),  'b-o',  label='Modified')
plt.plot(range(1, num_buses+1), np.abs(V_pso),  'g-^',  label='PSO-V2G')
plt.xlabel('Bus Number'); plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('Voltage Profile (All Scenarios)'); plt.grid(True); plt.legend()

plt.figure(figsize=(10,5))
# Branch currents number = num_lines, ensure x matches that
plt.plot(range(1, num_lines+1), np.abs(Ibranch_orig), 'r--s', label='I orig')
plt.plot(range(1, num_lines+1), np.abs(Ibranch_mod),  'b-o',  label='I mod')
plt.plot(range(1, num_lines+1), np.abs(Ibranch_pso),  'g-^',  label='I pso')
plt.xlabel('Line Number'); plt.ylabel('Branch Current (p.u.)')
plt.title('Branch Currents (All Scenarios)'); plt.grid(True); plt.legend()

# Power loss per branch (kW) bar chart - align lengths with num_lines
plt.figure(figsize=(12,6))
width = 0.25
x = np.arange(1, num_lines+1)
plt.bar(x - width, P_loss_line_orig, width=width, label='Original')
plt.bar(x,               P_loss_line_mod,  width=width, label='Modified')
plt.bar(x + width,       P_loss_line_pso,  width=width, label='PSO-V2G')
plt.xlabel('Line Number'); plt.ylabel('Active Power Loss (kW)')
plt.title('Per-Branch Active Power Loss (kW)'); plt.grid(True); plt.legend()

# Plot loss convergence for PSO
plt.figure(figsize=(8,4))
plt.plot(np.arange(1, max_iterations+1), loss_history, '-o')
plt.xlabel('PSO Iteration'); plt.ylabel('Best Loss (kW)')
plt.title('PSO Convergence (best loss per iteration)'); plt.grid(True)

# Load distribution (original and modified)
plt.figure(figsize=(10,5))
plt.plot(range(1, num_buses+1), (P_pu_base * S_base_MVA * 1000.0), 'r-o', label='Original P (kW)')
plt.plot(range(1, num_buses+1), (P_pu_modified * S_base_MVA * 1000.0), 'b-s', label='Modified P (kW)')
plt.xlabel('Bus'); plt.ylabel('Real Load (kW)'); plt.title('Bus Real Load Distribution'); plt.legend(); plt.grid(True)

plt.show()

# End of script
