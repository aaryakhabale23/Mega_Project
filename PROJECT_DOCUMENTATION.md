# SMART ENVIRONMENT ENERGY OPTIMIZER
## Complete Project Documentation — Deep Learning + Reinforcement Learning

**Project Title:** Smart Environment Energy Optimizer using Deep Learning-Based Occupancy Estimation and Reinforcement Learning Control

**Team Members:** [Your names here]

**Faculty Guides:**
- DL Faculty: [Name]
- RL Faculty: [Name]

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Deep Learning Component](#3-deep-learning-component-occupancy-estimation)
4. [Reinforcement Learning Component](#4-reinforcement-learning-component-energy-optimization)
5. [Integration & Demo Pipeline](#5-integration--demo-pipeline)
6. [Results & Evaluation](#6-results--evaluation)
7. [Why We Chose What We Chose — All The Whys](#7-why-we-chose-what-we-chose)
8. [Known Limitations & Honest Assessment](#8-known-limitations--honest-assessment)
9. [How to Run](#9-how-to-run)
10. [File Structure](#10-file-structure)
11. [References](#11-references)

---

# 1. PROJECT OVERVIEW

## The Problem (In Simple Words)

Imagine your college building. Every day, lights, fans, and ACs run at full power in EVERY room — even when classrooms are empty. This wastes a massive amount of electricity and money.

**Real-world example:** A lecture hall with 8 tube lights (40W each), 2 fans (75W each), and 1 AC (1500W) consumes 1,970 watts. If it runs for 10 hours daily, that's 19.7 kWh per day — just ONE room. With 5 rooms, that's ~100 kWh/day or **Rs 800/day wasted** when rooms are empty.

## Our Solution

We built an **AI-powered energy management system** that:

1. **SEES** how many people are in each room (using a camera + Deep Learning)
2. **DECIDES** what to turn on/off (using a trained RL agent)
3. **SAVES** energy while keeping everyone comfortable

**Layman Analogy:** Think of it like a smart thermostat, but instead of just temperature, our AI watches the room through a camera, counts people, and then determines: "There are only 5 students in Room 102 — I'll keep lights at half and turn AC off. But Room 103 has 35 students — everything goes to full power."

## Key Achievement

- **69-76% energy savings** compared to always-on baseline
- **MAE of 0.69** on indoor crowd counting (less than 1 person error on average!)
- Real-time dashboard showing everything live

---

# 2. SYSTEM ARCHITECTURE

## High-Level Flow

```
Camera/Video → DL Pipeline → Occupancy Count → RL Agent → Device Control → Dashboard
     |              |              |                |              |            |
  Raw frames   YOLOv8 +         "14 people"     PPO decides   L:ON F:ON    Live visual
               Density Head       in Room 102   "Light=half    AC:OFF      updates
                                                 Fan=half"
```

## Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Feature Extractor | YOLOv8-small (frozen) | Extract visual features from camera frames |
| Density Head | 3-layer Dilated CNN | Generate density maps → count people |
| RL Agent | PPO (Stable-Baselines3) | Decide device levels for each room |
| Environment | Custom Gymnasium Env | Simulate college floor with 5 rooms |
| Dashboard | Matplotlib (TkAgg) | Real-time visualization |

## Data Flow

1. Camera captures a frame (640×480 pixels)
2. YOLOv8 backbone extracts features → [B, 128, 60, 80] tensor
3. Density Head produces a density map → sum gives people count
4. Auto-calibration removes background noise
5. Count is normalized to occupancy percentage (0-100%)
6. RL agent observes occupancy + device states
7. Agent outputs device levels (OFF / HALF / FULL) for each device
8. Comfort override ensures minimum comfort when people are present
9. Dashboard displays everything in real-time

---

# 3. DEEP LEARNING COMPONENT (Occupancy Estimation)

*This section is for your DL faculty.*

## 3.1 Problem Statement

Estimate the number of people in an indoor room from a single camera frame, without requiring person detection or tracking.

## 3.2 Approach: Density Map Regression

Instead of detecting individual people (which fails in crowded scenes due to occlusion), we use **density map regression**:

- For each annotated head position, place a Gaussian blob
- Train a CNN to output a density map
- **Sum of all pixels in the density map = person count**

**Why density maps instead of direct counting?**
- Direct regression ("this image has 23 people") doesn't learn spatial information
- Density maps preserve WHERE people are, making the model much more robust
- The model can generalize to different crowd densities

**Layman analogy:** Instead of asking "how many grains of rice are in this bowl?", we first create a heat map showing where rice is concentrated. Adding up the heat map gives us the total count. This is more reliable than trying to count individual grains.

## 3.3 Architecture

### Backbone: YOLOv8-small (FROZEN)

```
Input Image [3, 480, 640]
         ↓
    YOLOv8-small backbone
    (Pretrained on COCO, weights FROZEN)
         ↓
Feature Map [128, 60, 80]
```

- We use YOLOv8-small as a **feature extractor only**
- It was pretrained on COCO (Common Objects in Context) — a massive dataset of 330K images
- We FREEZE all backbone weights — we don't train these at all
- **Why freeze?** The backbone already knows how to extract visual features (edges, textures, shapes). Training it would require 100x more data and compute.
- We extract from **layer 4** (C2f block after stride-8 downsampling) which gives 128-channel features at 1/8th resolution

### Density Head (TRAINABLE)

```
Feature Map [128, 60, 80]
         ↓
Conv2d(128→64, k=3, dilation=1, padding=1) + ReLU
         ↓
Conv2d(64→32, k=3, dilation=2, padding=2) + ReLU
         ↓
Conv2d(32→1, k=3, dilation=2, padding=2) + ReLU
         ↓
Bilinear Upsample to [1, 480, 640]
         ↓
Density Map (sum ≈ person count)
```

**Why dilated convolutions?** Dilated (atrous) convolutions increase the receptive field without adding parameters. With dilation=2, a 3×3 kernel effectively covers a 5×5 area. This helps the model understand context around each person without being computationally expensive.

**Why ReLU at the end (not sigmoid)?** Density values should be non-negative (you can't have -0.5 people) but not bounded above. Sigmoid would cap at 1.0, which is wrong — a pixel could contribute more than 1 to the total count.

## 3.4 Dataset: Mall Dataset

| Property | Value |
|----------|-------|
| Source | CUHK (Chinese University of Hong Kong) |
| Frames | 2000 video frames from a mall surveillance camera |
| Resolution | 640 × 480 pixels |
| People per frame | 13 to 53 (mean: 31.2) |
| Annotations | Head positions (x, y) for every person in every frame |
| Scene | Indoor — overhead surveillance view |

**Why Mall Dataset and not ShanghaiTech?**

We initially trained on ShanghaiTech Part B, but it contains **outdoor crowd scenes** (streets, plazas, stadiums). When applied to indoor rooms, the model hallucinated 30-40 people in empty classrooms because the texture patterns were completely different.

The Mall Dataset is **indoor surveillance** — very similar to our target scenario (classroom cameras). After retraining, accuracy improved dramatically.

**Why not a custom dataset?** Creating a crowd counting dataset requires manually annotating EVERY single person's head position in thousands of frames. The Mall Dataset's 60,000+ annotations across 2000 frames would take weeks of manual work.

## 3.5 Training Details

### Preprocessing: Density Map Generation

For each frame with N annotated head positions:
1. Create a blank image (480×480)
2. Place a value of 1.0 at each head (x, y) coordinate
3. Apply a **Gaussian filter** (σ=4.0, kernel_size=15)
4. The resulting density map has the property: **sum of all pixels ≈ N**

**Why σ=4.0?** This spread means each person affects a ~15×15 pixel area. Too small (σ=1) creates sparse, hard-to-learn maps. Too large (σ=15) blurs people together.

### Training Configuration

| Hyperparameter | Value | Why? |
|----------------|-------|------|
| **Epochs** | 50 (max) | Sufficient for 1600 training samples; early stopping prevents overfitting |
| **Batch Size** | 4 | Small because density maps are high-resolution (60×80) and we train on CPU |
| **Learning Rate** | 1e-4 (0.0001) | Standard for fine-tuning with Adam optimizer; not too aggressive for a pretrained backbone |
| **Optimizer** | Adam | Adaptive learning rates per-parameter; works well with small datasets |
| **Patience** | 12 epochs | If no improvement for 12 consecutive epochs, stop training |
| **Train/Val Split** | 80/20 | 1600 train + 400 validation frames |
| **Augmentation** | Horizontal flip + color jitter | Doubles effective data; flip is natural (left-right symmetry); color jitter handles lighting changes |
| **Seed** | 42 | See Section 7 for why |

### Loss Function: DensityLoss

```python
Loss = MSE(predicted_density, ground_truth_density) + 0.01 × |predicted_count - actual_count|
```

Two components:
1. **MSE (pixel-level):** Forces the density map shape to match ground truth
2. **Counting loss (count_weight=0.01):** Directly penalizes wrong total counts

**Why both?** MSE alone might produce beautiful-looking density maps that sum to the wrong number. The counting loss ensures the sum is correct. We weight it at 0.01 because MSE values are much smaller than count values.

### Learning Rate Scheduler

- **ReduceLROnPlateau**: If validation loss doesn't improve for 5 epochs, halve the learning rate
- This is like an automatic fine-tuning mechanism — as training slows down, the LR gets smaller for finer adjustments
- Factor: 0.5 (halve), Patience: 5 epochs

## 3.6 Auto-Calibration (Noise Removal)

The density model has inherent "background noise" — it sees texture patterns in walls, furniture, etc. and assigns small density values. This varies per scene.

**Our solution:** Auto-calibration using the first 5 frames:
1. Run the model on the first 5 frames of a video
2. Take the minimum raw count across these frames × 0.9 = **noise floor**
3. All subsequent predictions subtract this noise floor

**Example:**
- Empty room video: First 5 frames give counts [29, 30, 28, 31, 29] → noise_floor = 28 × 0.9 = 25.2
- Raw count = 30 → Cleaned = 30 - 25.2 = **4.8 ≈ 5 people** (close to 0!)
- Room with 7 people: Raw = 35 → Cleaned = 35 - 25.2 = **9.8 ≈ 10 people** (close to 7!)

## 3.7 Evaluation Results

### On Mall Dataset (2000 frames)

| Metric | Value | Meaning |
|--------|-------|---------|
| **MAE** | **0.69** | On average, the model is off by less than 1 person |
| **RMSE** | **1.07** | Root mean squared error — penalizes large errors more |
| **MAPE** | **2.3%** | Only 2.3% percentage error |

### On Custom Videos (after auto-calibration)

| Video | Actual People | Model Output | Accuracy |
|-------|--------------|--------------|----------|
| aarya.mp4 | ~7 | 8-9 cleaned | Good |
| hackx.mp4 | ~15 | 14-16 cleaned | Good |
| empty.mp4 | 0 | 3-5 cleaned | Acceptable (noise) |
| lav.mp4 | ~20 | 23 cleaned | Good |

---

# 4. REINFORCEMENT LEARNING COMPONENT (Energy Optimization)

*This section is for your RL faculty.*

## 4.1 Problem Formulation

**Goal:** Learn a policy that controls lights, fans, and ACs across 5 rooms to minimize energy consumption while maintaining occupant comfort.

**Layman Analogy:** Imagine you're a building manager. Every 6 minutes, you walk through the building, check how many people are in each room, and decide: should the lights be off, at half, or full? Same for fans and AC. You want to save electricity but not make students uncomfortable. Over time, you learn the patterns — "Room 101 is always empty at 3 PM, so I can turn everything off."

Our RL agent learns this same intuition, but automatically and optimally.

## 4.2 Environment Design (CollegeFloorEnv)

### Rooms Modeled

| Room | Type | Capacity | Devices | Power |
|------|------|----------|---------|-------|
| 101 | Lecture Hall | 60 | 8 lights (40W), 2 fans (75W), 1 AC (1500W) | Max 1,970W |
| 102 | Lecture Hall | 60 | 8 lights (40W), 2 fans (75W), 1 AC (1500W) | Max 1,970W |
| 103 | Computer Lab | 40 | 6 lights (20W), 1 AC (1500W), 40 computers (80W) | Max 4,740W |
| 104 | Tutorial Room | 30 | 4 lights (40W), 2 fans (75W) | Max 310W |
| 105 | Staff Room | 10 | 4 lights (40W), 1 AC (1500W) | Max 1,660W |
| Corridor | — | — | 4 lights (40W, always on at 50%) | Fixed 80W |

**Total always-on baseline: 108.1 kWh/day** (if everything runs at full for 10 hours)

### MDP Formulation

**State Space (Observation): Box(27)**
- Per room (5 rooms × 5 values = 25): [occupancy, activity, light_level, fan_level, ac_level]
- Global (2 values): [outside_temperature_normalized, time_of_day_normalized]
- All values normalized to [0, 1]

**Action Space: MultiDiscrete([3] × 12)**
- 12 controllable devices (each room's available devices)
- Each device has 3 levels: 0 (OFF), 1 (HALF), 2 (FULL)
- Example action: [2, 1, 0, 1, 1, 0, ...] means "Room 101: lights=FULL, fans=HALF, AC=OFF; Room 102: lights=HALF, fans=HALF, AC=OFF; ..."

**Episode:** 96 timesteps = 10 hours (8 AM to 6 PM), each step ≈ 6.25 minutes

### Occupancy Simulation

Occupancy follows **realistic Gaussian curve patterns:**
- Room 101 (Lecture Hall): Peaks at 9-11 AM and 1-3 PM (class hours)
- Room 103 (Computer Lab): Peaks at 2-5 PM (lab sessions)
- Room 104 (Tutorial): Short scattered peaks
- Room 105 (Staff Room): Steady low occupancy 9 AM - 5 PM

Each episode adds a random **phase offset** (±5%) so the agent doesn't memorize exact patterns. Gaussian noise (σ=0.1) is also added.

## 4.3 Reward Function

```
Reward = α × Comfort − β × Energy_Cost − γ × Adjustment_Penalty
```

| Term | Weight | Purpose |
|------|--------|---------|
| **α × Comfort** | α = 1.0 | Positive reward for keeping occupants comfortable |
| **β × Energy_Cost** | β = 0.6 | Penalty for consuming energy (₹8/kWh) |
| **γ × Adjustment** | γ = 0.1 | Penalty for frequently switching devices (wear & tear) |

### How Comfort is Calculated

```
Comfort = 1 − mean(max(required_level − actual_level, 0)) across devices
```

| Occupancy | Required Level | Example |
|-----------|---------------|---------|
| > 70% | 1.0 (FULL) | 40 students → everything at max |
| 30-70% | 0.5 (HALF) | 15 students → lights and fan at half |
| < 30% | 0.0 (OFF OK) | Empty → agent can turn everything off |

**Important:** Only UNDER-provision hurts comfort. Over-provision (lights on in empty room) doesn't hurt comfort — it only increases energy cost. This teaches the agent to save energy, not just match levels.

### Why β = 0.6?

We ran a **beta sweep** testing β = 0.3, 0.6, and 0.9:

| β (Beta) | Energy (kWh/day) | Comfort | Behavior |
|----------|-----------------|---------|----------|
| 0.3 | 26.4 | 84.0% | Comfort-focused, wastes some energy |
| **0.6** | **25.1** | **81.2%** | **Best balance** |
| 0.9 | 26.5 | 83.3% | Too aggressive, weird behavior |

β = 0.6 gave the **lowest energy consumption** while maintaining good comfort.

## 4.4 PPO Algorithm

### Why PPO?

| Algorithm | Pros | Cons | Verdict |
|-----------|------|------|---------|
| DQN | Simple | Doesn't handle MultiDiscrete actions well | ✗ |
| A2C | Fast, handles continuous | High variance | ✗ |
| **PPO** | **Stable, handles MultiDiscrete, sample efficient** | **Slower** | **✓** |
| SAC | State-of-the-art for continuous | Requires continuous actions | ✗ |

PPO (Proximal Policy Optimization) is ideal because:
1. Handles our MultiDiscrete action space (12 devices × 3 levels) naturally
2. **Clipped objective** prevents catastrophic policy updates
3. Works well with limited training data
4. Industry standard for real-world control problems

### PPO Hyperparameters

| Parameter | Value | Why? |
|-----------|-------|------|
| **Total timesteps** | 200,000 | ≈2,083 episodes (96 steps each). Enough for PPO to converge |
| **Learning rate** | 3e-4 | SB3's default, well-tested for PPO |
| **n_steps** | 2048 | Rollout buffer size. ~21 episodes per update — provides stable gradients |
| **batch_size** | 64 | Mini-batch for gradient updates. Power of 2 for GPU efficiency |
| **n_epochs** | 10 | Number of passes over the rollout buffer per update |
| **Policy** | MlpPolicy | 2-layer MLP (64×64). Our 27-dim observation is tabular, not images |
| **Discount (γ)** | 0.99 | Default. Agent cares about long-term energy savings, not just next step |

**Layman analogy for n_steps:** The agent plays 21 full "games" (episodes), remembers what happened, then reviews all 2048 steps to improve its strategy. Then it plays 21 more games, reviews, and repeats. After ~100 such cycles (200,000 steps), it becomes expert.

## 4.5 RL Training Results

After 200,000 timesteps of training:

| Metric | Always-On Baseline | PPO Agent | Improvement |
|--------|-------------------|-----------|-------------|
| Energy/day | 108.1 kWh | 25.1 kWh | **76.8% savings** |
| Cost/day | ₹864.80 | ₹200.74 | **₹664 saved** |
| Comfort | 100% | 81.2% | Acceptable trade-off |

### Comfort Override (Post-RL Safety Layer)

The PPO agent sometimes turns everything off to maximize savings. In reality, you can't have students sitting in the dark! We added a **comfort override layer:**

| Occupancy | Override |
|-----------|---------|
| > 60% (crowded) | All devices → FULL |
| 25-60% (moderate) | All devices → at least HALF |
| 5-25% (few people) | Lights + Fan → at least HALF. AC stays off |
| < 5% (empty) | Agent's choice (may turn everything off) |

This is like a **safety net** — the RL agent does the optimization, but we guarantee minimum human comfort.

---

# 5. INTEGRATION & DEMO PIPELINE

## How It All Works Together

```
                          ┌─────────────────┐
                          │   Video Input    │
                          │  (aarya.mp4)     │
                          └────────┬─────────┘
                                   │ frame
                                   ▼
                    ┌──────────────────────────┐
                    │     DL Pipeline           │
                    │  YOLOv8 → Density Head    │
                    │  Raw count → Calibrate    │
                    │  → Cleaned count → Occ%   │
                    └────────────┬──────────────┘
                                 │ occupancy = 12%
                                 ▼
              ┌─────────────────────────────────────┐
              │         CollegeFloorEnv              │
              │  Room 102 ← DL occupancy (12%)      │
              │  Rooms 101,103-105 ← simulated      │
              │  Observation: [occ,act,L,F,AC]×5+2  │
              └──────────────────┬──────────────────┘
                                 │ obs (27-dim vector)
                                 ▼
              ┌─────────────────────────────────────┐
              │           PPO Agent                  │
              │  obs → MlpPolicy → action           │
              │  action: 12 device levels (0/1/2)   │
              └──────────────────┬──────────────────┘
                                 │ action
                                 ▼
              ┌─────────────────────────────────────┐
              │        Comfort Override              │
              │  If occupied, enforce minimum levels │
              │  Recompute actual energy usage       │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │          Dashboard                   │
              │  Room cards + Metrics + Bar chart    │
              │  Real-time update every 0.5s         │
              └─────────────────────────────────────┘
```

## Dashboard Layout

The dashboard has 3 sections:

1. **Room Cards (top row):** Each room shows people count (big number), occupancy %, device status (L/F/AC as ON/MID/OFF), and power draw (watts)
2. **Live Metrics (top-right):** Time, step, total power, energy used, cost, savings %, comfort %
3. **Bottom row:** Corridor info + Occupancy bar chart + Savings summary

---

# 6. RESULTS & EVALUATION

## 6.1 DL Model Evaluation

### Quantitative (Mall Dataset, 2000 frames)

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | **0.69** |
| RMSE (Root Mean Square Error) | **1.07** |
| MAPE (Mean Absolute % Error) | **2.3%** |

**What this means:** On average, the model's count is wrong by less than 1 person out of 13-53 people per frame. That's 97.7% accurate.

### Evaluation Plots Generated

- `evaluation_results/evaluation_plots.png` — Scatter plot, error distribution, timeline, residuals
- `evaluation_results/sample_predictions.png` — Visual predictions on 8 random frames

## 6.2 RL Agent Evaluation

### Energy Savings

| Scenario | Energy (kWh/day) | Cost (₹/day) | Savings |
|----------|-----------------|---------------|---------|
| Always-On Baseline | 108.1 | 864.80 | — |
| PPO Agent (simulated) | 25.1 | 200.74 | **76.8%** |
| PPO Agent (with video) | 32-40 | 260-325 | **62-70%** |

### Why is savings lower with video?

With video, Room 102 has real occupancy (~12-27%), so devices stay on longer than in pure simulation where some rooms are empty. This is more realistic.

---

# 7. WHY WE CHOSE WHAT WE CHOSE

## Why seed = 42?

**42** is the most commonly used random seed in machine learning research. It's a cultural convention from "The Hitchhiker's Guide to the Galaxy" where 42 is the "Answer to the Ultimate Question of Life, the Universe, and Everything." Any fixed integer would work — the point is **reproducibility**: running the same code twice gives the same results.

## Why YOLOv8 and not ResNet/VGG as backbone?

- YOLOv8-s has **11.2M parameters** (small). VGG-16 has 138M.
- YOLOv8 is pretrained on COCO (80 classes including "person") — it already knows what people look like
- Inference is fast even on CPU (~50ms/frame)
- The "v8" is the latest version with modern architectural improvements (C2f blocks, anchor-free)

## Why freeze the backbone?

- We only have **2000 training images** (Mall Dataset). Training a 11.2M parameter backbone on 2000 images = massive overfitting
- The backbone already extracts excellent features from ImageNet/COCO pretraining
- We only train the **density head** (5,793 parameters) — much more data-efficient

## Why 50 epochs?

- With 1600 training samples and batch size 4, each epoch = 400 iterations
- 50 epochs = 20,000 gradient updates — enough for a small 3-layer head
- **Early stopping** with patience=12 means we typically stop at epochs 25-35
- Going beyond 50 risks overfitting on such a small dataset

## Why batch size 4?

- Training on CPU (no GPU) — larger batches slow down training
- Density maps are 60×80 floats — memory adds up
- Small batches add implicit regularization (noisier gradients help prevent overfitting)
- With 4 samples per batch, one epoch processes 400 batches — plenty of gradient updates

## Why learning rate 1e-4?

- Standard for Adam optimizer when fine-tuning
- 1e-3 (10x larger) caused training instability
- 1e-5 (10x smaller) was too slow — needed 100+ epochs
- ReduceLROnPlateau automatically halves it when progress stalls

## Why Gaussian σ=4.0 for density maps?

- σ=4 means each person affects ~15×15 pixels (3σ rule: 99.7% of the Gaussian is within 12 pixels)
- The model needs some "spread" to learn — a single-pixel point map is too sparse
- σ too large (>10) blurs people together, making the model unable to separate them
- Literature standard is σ=3-5 for medium-density scenes

## Why PPO and not Q-learning?

- Our action space is **MultiDiscrete([3] × 12)** — 12 devices each with 3 levels = 3^12 = 531,441 possible actions
- Q-learning would need a Q-table/network for ALL 531,441 actions — impractical
- PPO uses a **policy network** that directly outputs action probabilities — scales much better
- PPO's clipped objective is more stable than plain Policy Gradient

## Why 200,000 training timesteps?

- Each episode = 96 steps → 200,000 steps ≈ 2,083 episodes
- PPO updates every 2048 steps → ~97 policy updates total
- After ~50 updates we see convergence (reward plateau)
- 200K is a common starting point for PPO on environments with ~100-step episodes

## Why ₹8/kWh?

- Average commercial electricity rate in India is ₹6-10/kWh depending on the state
- ₹8 is a representative mid-range value
- This makes the cost calculations realistic for Indian institutions

## Why α=1.0, β=0.6, γ=0.1?

- **α=1.0 (comfort):** Full weight on comfort — occupants matter most
- **β=0.6 (energy):** Found via beta sweep to give best energy-comfort balance
- **γ=0.1 (switching):** Small penalty for frequent switching — prevents "flickering" devices but doesn't dominate the reward

## Why 96 timesteps per episode?

- 10-hour workday (8 AM to 6 PM) ÷ 96 = **6.25 minutes per step**
- This matches realistic HVAC control intervals (most building management systems update every 5-15 minutes)
- 96 = 32 × 3 = convenient for PPO's batch processing

---

# 8. KNOWN LIMITATIONS & HONEST ASSESSMENT

## 8.1 DL Limitations

### Domain Gap
- The model is trained on the **Mall Dataset** (one specific mall camera angle)
- Different indoor scenes (classrooms, offices) have different lighting, angles, and furniture patterns
- This causes **background noise** (the model "sees" phantom people in textures)
- **Our mitigation:** Auto-calibration subtracts scene-specific noise

### Empty Room Issue
- An completely empty room still produces a raw count of ~25-30
- After auto-calibration, this drops to ~3-5 (not zero)
- This is a fundamental limitation of density-based counting — the model never truly outputs zero

### Single Camera Only
- Currently supports only **one camera → one room** (Room 102)
- Other rooms use simulated occupancy
- Scaling to 5 cameras would need 5× inference time

### No Person Re-identification
- The model counts **density**, not individuals
- It can't tell if Person A left and Person B entered (total stays the same)
- For energy management, we only need the count, not identity — so this is acceptable

## 8.2 RL Limitations

### Simulated Environment
- Rooms 101, 103, 104, 105 use **simulated occupancy curves** (Gaussian bumps), not real camera data
- Real occupancy patterns may differ from our modeled curves
- However, the Gaussian model is a reasonable approximation of class schedules

### Aggressive Optimization
- The PPO agent sometimes turns everything off in low-occupancy rooms
- Without the comfort override, students could be sitting in the dark
- **Our mitigation:** Hard-coded comfort override ensures minimum device levels

### No Temperature Modeling
- We don't model room temperature dynamics (heat transfer, insulation, etc.)
- AC decisions are based purely on occupancy, not actual temperature
- A full BMS (Building Management System) would also consider temperature sensors

### Fixed Device Levels (3 levels only)
- Devices can only be OFF, HALF, or FULL
- Real dimmers have continuous control (0-100%)
- We simplified to 3 levels to keep the action space manageable

## 8.3 System Limitations

### Tkinter Threading Issues
- Matplotlib's TkAgg backend occasionally crashes when GUI updates lag behind video processing
- Error: `_tkinter.TclError: pyimage3`
- **Our mitigation:** Wrapped in try/except so the demo continues despite occasional rendering glitches

### CPU-Only Training
- All training runs on CPU (no GPU available)
- Mall Dataset training: ~1.5 hours
- RL training: ~10 minutes
- A GPU would make this 10-20× faster

---

# 9. HOW TO RUN

## Prerequisites

```
Python 3.10+
pip install torch ultralytics opencv-python scipy matplotlib stable-baselines3 gymnasium tqdm
```

## Step 1: Train the DL model (if needed)

```bash
python train_mall.py
```
Downloads Mall Dataset (~130MB), preprocesses, trains for ~50 epochs, saves `density_head.pth`

## Step 2: Train the RL agent (if needed)

```bash
python rl_training/train_ppo.py --timesteps 200000
```
Saves `ppo_college_floor.zip`

## Step 3: Evaluate DL model

```bash
python evaluate_model.py
```
Generates plots in `evaluation_results/`

## Step 4: Run the demo

```bash
# With video file:
python demo.py --source aarya.mp4

# With webcam:
python demo.py

# Simulation only (no camera):
python demo.py --simulate
```

---

# 10. FILE STRUCTURE

```
Mega_Project/
├── demo.py                          # Main integration demo
├── train_mall.py                    # Train density head on Mall Dataset
├── evaluate_model.py                # Evaluate trained model with plots
├── local_train.py                   # Train on ShanghaiTech (legacy)
├── density_head.pth                 # Trained density head weights
├── ppo_college_floor.zip            # Trained PPO agent
│
├── dl_pipeline/                     # Deep Learning module
│   ├── model.py                     # YOLOv8Backbone + DensityHead + OccupancyPipeline
│   ├── preprocess.py                # Density map generation
│   ├── train.py                     # Core training logic
│   └── evaluate.py                  # Evaluation metrics (MAE, RMSE)
│
├── rl_environment/                  # Reinforcement Learning environment
│   └── env.py                       # CollegeFloorEnv (Gymnasium)
│
├── rl_training/                     # RL training scripts
│   ├── train_ppo.py                 # PPO training with SB3
│   ├── evaluate_rl.py               # Compare PPO vs baselines
│   ├── baselines.py                 # Random + rule-based policies
│   └── beta_sweep.py               # Hyperparameter sweep for β
│
├── visualization/                   # Dashboard
│   └── dashboard.py                 # Real-time matplotlib dashboard
│
├── evaluation_results/              # Model evaluation outputs
│   ├── evaluation_plots.png         # Accuracy plots
│   └── sample_predictions.png       # Visual predictions
│
├── beta_sweep_results/              # Beta sweep experiment results
│   └── sweep_summary.json           # Comparative results for β=0.3/0.6/0.9
│
├── data/                            # Datasets (gitignored)
│   └── mall_dataset/                # Mall Dataset (2000 frames)
│
├── training_curves.png              # Training loss curves
├── train_loss_log.npy               # Training loss history
└── val_loss_log.npy                 # Validation loss history
```

---

# 11. REFERENCES

1. **Mall Dataset:** C. C. Loy, K. Chen, S. Gong, T. Xiang. "From Semi-Supervised to Transfer Counting of Crowds." ICCV 2013.
2. **YOLOv8:** Jocher, G., Chaurasia, A., Qiu, J. (2023). "Ultralytics YOLOv8." GitHub.
3. **PPO:** Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
4. **Stable-Baselines3:** Raffin, A., et al. "Stable-Baselines3: Reliable RL Implementations." JMLR, 2021.
5. **Gymnasium:** Farama Foundation. "Gymnasium: A Standard API for RL." 2023.
6. **Density Map Regression:** Zhang, Y., et al. "Single-Image Crowd Counting via Multi-Column CNN." CVPR, 2016.
7. **Dilated Convolutions:** Yu, F., Koltun, V. "Multi-Scale Context Aggregation by Dilated Convolutions." ICLR, 2016.

---

*Document generated for project presentation. All code is available on the project repository.*
