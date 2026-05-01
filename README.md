# An Integrated Pipeline for IBZTN (Intent-Based Zero-Touch Networks)

This repository contains the official implementation of the research paper: **"An Integrated Pipeline for Intent-Based Zero-Touch Networks: From Intent Translation to Minimal-Modification Reconfiguration"**.

## 🚀 Project Overview

To support **Industry 5.0** smart factories that demand ultra-low latency and high reliability, we propose a three-layered integrated **IBZTN pipeline**. This system bridges the gap between high-level natural language "intents" and low-level autonomous network control.

### Key Layers
*   **Layer 1: Intent Translation** – Converts natural language intents into structured QoS profiles using **RAG** and a **QoS Validator** to eliminate numerical hallucinations[cite: 1].
*   **Layer 2: Feasibility Prediction** – Utilizes a **GINE-based binary classifier** to rapidly assess whether the current network can satisfy the requested QoS[cite: 1].
*   **Layer 3: Autonomous Reconfiguration** – Deploys a **BC-PPO agent** with **Smart TE masking** to recover unfulfilled network states through minimal edge modifications[cite: 1].

---

## 🛠 Core Technologies
*   **Generative AI & RAG**: Integration of state-of-the-art LLMs with a Knowledge Base of 1,622 industrial standard chunks (TSN, URLLC, etc.)[cite: 1].
*   **Graph Learning**: **GINE** (Graph Isomorphism Network with Edge features) for precise bottleneck detection by learning link attributes[cite: 1].
*   **Deep Reinforcement Learning**: **BC (Behavior Cloning)** combined with **Maskable PPO** to find optimal trajectories in a 6D action space[cite: 1].

---

## 📊 Experimental Highlights
Our integrated pipeline achieved the following results in complex industrial scenarios[cite: 1]:
*   **0.0% Constraint Violations**: Perfect data integrity and schema compliance in intent translation[cite: 1].
*   **93.9% F1-Score**: High reliability in predicting feasibility and locating bottlenecks[cite: 1].
*   **87.8% Recovery Success**: Successful autonomous restoration in multi-violation environments[cite: 1].
*   **Minimal Modification**: Normalized the network with an average of only **9.8 edits**[cite: 1].
*   **Ultra-Low Latency**: Inference speed of **5.56 ms**, facilitating real-time closed-loop control[cite: 1].

---

## 📜 Citation

If you find this research or code useful, please cite our work[cite: 1]:
```bibtex
@article{Seo2026IBZTN,
  title={An Integrated Pipeline for Intent-Based Zero-Touch Networks: From Intent Translation to Minimal-Modification Reconfiguration},
  author={Seo, DongJun and Kim, KeeCheon},
  journal={Applied Sciences},
  year={2026},
  doi={10.3390/app1010000}
}
