# Reddit Posts for RAVANA

Goal: Get feedback on the approach, attract contributors, and drive activity to the repos.

Source code:
- Primary: https://codeberg.org/oxiverse/ravana
- Mirrored: https://github.com/oxiverse-ecosystem/ravana

---

## r/MachineLearning

**[R] RAVANA — A neural architecture that learns entirely without backpropagation or gradients (pressure-driven self-organization + Hebbian plasticity)**

I've been building a cognitive architecture that replaces gradient descent with something fundamentally different: **pressure-driven self-organization**. Instead of minimizing a loss function, the system accumulates free energy from prediction errors across 5 channels (semantic, linguistic, episodic, contradiction, abstraction) and self-organizes to reduce that pressure through Hebbian/anti-Hebbian plasticity.

Key results so far:
- **75% cross-domain transfer** (Top-1), **100% Top-10** on unseen relation types
- **0% catastrophic forgetting** on permuted MNIST after introducing sleep consolidation (SWS + REM cycles)
- **CPU-native** — NumPy only, no GPU required
- **Self-organizing ConceptGraph** with typed edges (semantic, causal, temporal, analogical) that grows unbounded
- **VAD emotion engine** modulates inference: arousal → exploration, valence → trust predictions

The architecture decomposes everything into (subject, relation, object) triples and uses spreading activation as the sole inference mechanism — no backprop, no gradients, no loss functions.

Looking for feedback on the approach, discussion on alternatives to gradient descent, and contributors interested in cognitive architectures, active inference, or neuro-symbolic systems.

Code: https://codeberg.org/oxiverse/ravana (primary) | https://github.com/oxiverse-ecosystem/ravana (mirror)
Paper + benchmarks in README.

---

## r/artificial

**[R] I built a cognitive architecture that learns like a brain — no backprop, no GPU, no forgetting**

Most AI systems are built on three assumptions: you need backpropagation, you need GPUs, and you need to carefully prevent catastrophic forgetting. I wanted to see what happens if you throw all three out.

RAVANA is a research prototype that:
- **Learns through prediction errors** — like Friston's free energy principle, the system feels "pressure" when predictions fail and self-organizes to reduce it
- **Never forgets** — a biologically-inspired sleep cycle (SWS for consolidation + REM for creative recombination) eliminated catastrophic forgetting entirely in our tests
- **Runs on CPU** — pure NumPy, works on a laptop
- **Has emotions** — a 3D Valence-Arousal-Dominance engine modulates how the system learns and infers
- **Learns continuously from the web** — curiosity-driven exploration, no retraining needed
- **Supports multi-user beliefs** — a BeliefStore tracks who believes what and merges across users

I'm at the stage where I need community feedback, discussion, and contributors. The codebase is substantial (~25k lines across 3 packages) with 1250+ tests and published on PyPI.

This is not a product — it's a research project exploring whether pressure-driven self-organization can work as a genuine alternative to gradient-based learning. Would love to hear thoughts from this community.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/cogsci

**[R] Implementing active inference and free energy minimization in a working cognitive architecture — looking for feedback**

I'm building a research project called RAVANA that takes ideas from cognitive science and implements them as a working ML system:

- **Free energy principle**: A 5-channel accumulator tracks prediction errors across semantic, linguistic, episodic, contradiction, and abstraction domains. Learning = reducing this pressure.
- **Sleep consolidation**: Two-phase sleep modeled on mammalian sleep — SWS for structural stabilization and abstraction, REM for creative recombination with counterfactual reversals and emotional valence flipping.
- **Dual-process theory**: Fast intuitive (System 1) vs. slow deliberative (System 2) processing paths.
- **VAD emotion**: Valence-Arousal-Dominance affective state modeled with differential equations, modulating inference and learning.
- **Global workspace theory**: A global workspace integrates competing interpretations.
- **Hippocampal indexing**: Episodic memory with replay during sleep.
- **Meaning and motivation**: Intrinsic motivation drives curiosity and exploration.

The system actually runs (CPU-only, NumPy), has 1250+ passing tests, and shows non-trivial results — 75% cross-domain transfer, 0% catastrophic forgetting with sleep.

I'm looking for feedback from people who work on active inference, predictive coding, or any of these cognitive frameworks — and for contributors who want to help bridge cognitive science and implemented systems.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/neuroscience

**[R] Biologically-inspired ML architecture with Hebbian plasticity, sleep consolidation, and hippocampal replay — implemented and working**

I've implemented a cognitive architecture that takes direct inspiration from neuroscience mechanisms:

- **Hebbian & anti-Hebbian plasticity** (local learning rules, no global loss signal)
- **Structural plasticity** (synaptogenesis and pruning)
- **4-stage sleep**: NREM (SWS) with homeostatic downscaling and hippocampal replay + REM with creative recombination and emotional valence flipping
- **Hippocampal indexing**: Episodic memory with replay-based consolidation during sleep
- **Basal ganglia gating** for action selection
- **Cerebellar n-gram model** for sequence prediction
- **Prefrontal workspace** for deliberative reasoning
- **VAD emotion** modeled on affective neuroscience

The key insight: none of these require backpropagation. Everything runs on local learning rules and spreading activation through a self-organizing concept graph.

I'm not a neuroscientist — I'm a ML engineer who wanted to see what happens when you take these biological mechanisms seriously in a working system. I'd love feedback from people who actually study these systems. Where am I getting the neuroscience wrong? What mechanisms am I missing that would be most impactful?

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/LocalLLaMA

**[R] A CPU-native alternative to transformer-based LLMs — cognitive architecture with no GPU, no backprop, no forgetting**

Most of us here care about running models without expensive hardware. I've been building something that goes in a completely different direction from transformers.

RAVANA is a cognitive architecture that:
- **Runs entirely on CPU** — pure NumPy, no CUDA, no GPU required
- **No backpropagation** — learns through local Hebbian plasticity rules and self-organization
- **No catastrophic forgetting** — biologically-inspired sleep cycles consolidate knowledge without overwriting
- **Self-organizing memory** — a ConceptGraph that grows and prunes itself, no fixed parameter count
- **Continuous web learning** — can browse the web, extract knowledge, and integrate it without retraining
- **~25k lines of Python** across 3 PyPI packages

The trade-off: it's not going to beat GPT-4 on MMLU. It's a research project exploring a different paradigm. But it runs on a ThinkPad, never forgets what you taught it, and the architecture is interpretable (knowledge stored as (subject, relation, object) triples in a graph).

Looking for contributors who find the "no GPU, no backprop" approach interesting, and feedback on where this paradigm could be useful.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/Python

**[R] RAVANA — 25k-line cognitive ML framework built with pure NumPy (no PyTorch, no JAX, no GPU)**

I built a full cognitive architecture using nothing but NumPy. No PyTorch, no JAX, no GPU — just numpy arrays and local learning rules.

The project includes:
- A **heterogeneous concept graph** with typed edges (semantic, causal, temporal, analogical)
- **Hebbian/anti-Hebbian plasticity** rules
- A **free energy accumulator** (5-channel) driving learning through prediction error
- **Sleep consolidation** with SWS and REM phases
- **VAD emotion engine** using differential equations
- **Continuous web learning** with circuit-breaker pattern
- **CLI chat interface** and event system
- **1250+ tests** with pytest
- **3 PyPI packages**

Stack: Python 3.10+, NumPy, pytest, ruff.

I've been working on this for a while and I'm looking for contributors — especially people interested in cognitive science, neuro-symbolic AI, or just writing Python that pushes NumPy to its limits. Happy to walk anyone through the codebase.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/opensource

**[R] RAVANA — open source cognitive architecture research project looking for contributors (ML, Python, neuroscience, or just curious)**

I'm the creator of RAVANA, a research project exploring whether pressure-driven self-organization can replace gradient descent in machine learning. It's licensed under OCL v1.0 (source-available, non-commercial, privacy-by-design).

**What the project needs right now:**
- **ML engineers** — help improve the learning algorithms and benchmark against traditional approaches
- **Python developers** — there's always refactoring, optimization, and tooling work
- **Neuroscience / cog-sci people** — ground the biological inspirations in actual science
- **Technical writers** — the docs could always be better
- **Testers** — try it out, break it, report issues
- **Researchers** — help design experiments that would actually be publishable

**Why contribute?** It's genuinely different from the mainstream ML paradigm. No backprop, no GPU dependency, biologically-inspired learning. If you're tired of the same PyTorch pipelines, this is something else.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/MLQuestions

**[Q] Can learning happen without gradient descent? Building a system that only uses local Hebbian plasticity — looking for discussion**

I've been building a learning system that completely avoids backpropagation and gradient descent. Learning works like this:

1. System makes a prediction → prediction error generates "free energy" (pressure)
2. Pressure triggers Hebbian/anti-Hebbian updates to connections (local, no global gradient)
3. During sleep, the system replays experiences and consolidates knowledge
4. Over time, the concept graph self-organizes to minimize prediction errors

I'm getting non-trivial results (75% cross-domain transfer, 0% catastrophic forgetting) but I keep wondering: what's the ceiling on this approach? Is there a fundamental limitation to learning without gradients that I'm not seeing?

Would love to hear from people who've thought about alternative learning paradigms, worked with Hebbian networks, or know the active inference literature well.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana

---

## r/MLJobs

**[Looking for collaborators] Cognitive architecture ML project — no GPU, no backprop, all NumPy**

This isn't a job posting — I'm looking for unpaid research collaborators who find this space interesting.

If any of these resonate, reach out:
- You're tired of the "throw more GPUs at it" paradigm
- You think biologically-inspired learning might have something to offer
- You know active inference / free energy principle literature and want to implement it
- You're a Python/NumPy optimization wizard
- You want to work on something that's genuinely different from mainstream ML

The project has ~25k lines of code, 1250+ tests, published on PyPI. It's a real working system, not a toy. I need people who can help push the research forward — designing experiments, improving the learning algorithms, writing papers.

DM me or check the repos for contribution guidelines.

Code: https://codeberg.org/oxiverse/ravana | https://github.com/oxiverse-ecosystem/ravana
