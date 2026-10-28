# Research notes — 28 October 2026

Sources checked on 28 October 2026 (America/Toronto). All arXiv
details were verified on the arXiv abstract pages; secondary coverage
was used only to find papers, never as the source for a claim about a
paper's contents. Figures attributed to the PersonTTS paper are the
paper's own, reported on AIME/HMMT with Qwen3 models; they are not
results of this repository.

## Primary source

Wang, X., Liu, Z., Zheng, T., Feng, S., Yuan, P., Li, Y., Shi, J.,
Zhang, Y., Tan, C., Zhang, J., Pan, B., and Li, K. "From Pareto to
Preference: Personalized Test-Time Scaling via Amortized Agentic Policy
Discovery." arXiv:2610.09684, submitted 7 October 2026.
https://arxiv.org/abs/2610.09684

What the paper establishes (from its abstract and full text):

- Test-time scaling (TTS) improves LLM reasoning by allocating extra
  inference computation. Existing efficiency work optimises accuracy
  against one resource dimension at a time: the accuracy-cost or the
  accuracy-latency Pareto frontier.
- Users state requirements jointly — an accuracy floor together with
  latency and inference-cost ceilings — and different requirements
  favour different controllers even on the same problem set, so a
  single-frontier optimum can miss the controller a user needs.
- The paper formulates Personalized Test-Time Scaling as discovering
  executable controllers that maximise the *joint satisfaction rate*:
  the fraction of evaluation seeds meeting the accuracy, latency and
  cost requirements simultaneously.
- An executable controller coordinates model selection, reasoning width
  and depth, refinement, pruning and stopping, and its decisions can
  depend on the target requirements. Candidates must be evaluated
  under the target profile rather than assumed to transfer.
- Running discovery independently per profile is expensive: the paper
  cites $39.9 and 160 minutes for a complete policy-discovery run in
  the prior AutoTTS system (Zheng et al., 2026b).
- PersonTTS amortises that cost by reusing prior search experience:
  retrieving a controller from a source profile with similar
  requirements and re-evaluating it under the target profile, plus a
  frozen Guide distilled from source discovery histories that informs
  subsequent proposals. A discovery agent then proposes controllers
  from the target requirements, replay feedback and search history.
- Evaluation: AIME and HMMT, six Qwen3 models (0.6B–32B), 100 source
  and 20 target requirement profiles per benchmark, 128 pre-sampled
  reasoning trajectories per problem-model pair, with separate
  held-out problem sets. The paper reports that PersonTTS
  substantially outperforms strong TTS baselines in joint
  satisfaction on unseen profiles and held-out problems, and that
  under the same candidate-evaluation budget, cross-user experience
  reuse further improves policy quality while substantially reducing
  discovery-agent time and cost. Secondary coverage of the full text
  reports the Guide-only variant cutting agent-call time by about 46%
  and cost by about 36% versus discovery from scratch, with
  warm-start retrieval mainly shortening elapsed time; the headline
  quality gains come mostly from the personalised (joint) objective
  itself, and enlarging the experience bank helps on AIME held-out
  problems but not on HMMT. Those percentages are the paper's, via
  secondary reporting of its results section, and are not reproduced
  as findings of this repository.
- The authors' code and data are released at
  https://github.com/WangXinglin/PersonTTS (repository created 6
  October 2026). This repository is an independent implementation
  written from the paper's formulation; no code was copied from the
  authors' repository.

## Why this project, and what it deliberately does not claim

The paper's evaluation design — controllers evaluated by replaying
pre-sampled trajectories — maps directly onto a CPU-runnable
reproduction: the trajectory bank is sampled once, and discovery is
measured in candidate evaluations. That is what this repository
implements, with simulated multi-step reasoning problems instead of
AIME/HMMT and Qwen3. Its numbers therefore describe controller
discovery mechanics on a synthetic testbed, not the paper's
benchmark results, and the README says so wherever results are
reported.

## Companion work checked, October 2026

- Dou, Z., Luo, K., Li, C., Lu, S., and Liu, Z. "FALCA:
  Falsification-Aware Reinforcement Learning with Hierarchical Credit
  Assignment for Long-Horizon Deep Search Agents." arXiv:2610.28361,
  submitted 20 October 2026. https://arxiv.org/abs/2610.28361 —
  Searcher/Falsifier architecture and hierarchical credit assignment
  for long-horizon deep search. Checked as a related direction on
  credit assignment in agentic RL; not implemented here.
- Cui, Y., et al. "MaD-RSI: Model-as-Data for Recursive Self-Improvement
  of Agent Harnesses." arXiv:2610.28201, submitted 20 October 2026.
  https://arxiv.org/abs/2610.28201 — treating the harness as data so it
  can rewrite itself against observed model behaviour. Checked as a
  related direction on improving the system around the model; not
  implemented here.
- Hybrid Reasoning Agents (Zou et al., arXiv:2610.18861, 13 October
  2026): latent-space reasoning with text interaction for tool use,
  evaluated on Qwen3-Base models on H100 GPUs. Rejected as the build
  target: it needs frontier-model training on GPUs and cannot be
  reproduced or measured honestly on CPU.

## Scale-up path checked

- TRL 1.16.0 (released 15 October 2026, per secondary release
  coverage) with GRPOTrainer, vLLM-powered generation and PEFT/LoRA;
  vLLM 0.32.0 (October 2026). The scale-up path in the README (replace
  the trajectory simulator with real model rollouts under TRL/vLLM,
  keep the same controller/profile/discovery interfaces) follows the
  stack the paper's own evaluation uses (Qwen3 models, pre-sampled
  trajectories), but no TRL/vLLM run was performed here and no GPU
  results are claimed.
- OpenAI Decisions API reached general availability on 22 October
  2026 (secondary coverage, week of 26 October 2026): a decision
  policy evaluated as an API is the deployment shape a discovered
  controller would eventually be served as. Mentioned in the README
  only as deployment context; no API calls were made.
