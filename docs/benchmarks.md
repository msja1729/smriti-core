# SMRITI on standard benchmarks

*Draft: results are filled in once the runs finish.*

## In one paragraph

<!-- SUMMARY -->

## What we tested, and why these benchmarks

SMRITI's own benchmark (MemPoison-Bench) was written by us. To check that SMRITI also works
on tests written by other people, we ran it on two public benchmarks that the research
community uses for prompt-injection attacks on tool-using agents:

| Benchmark | Who made it | What it contains | How pass/fail is decided |
|---|---|---|---|
| **InjecAgent** | UIUC, *Findings of ACL 2024* | 1,054 cases. A user asks for something harmless (e.g. "show me the reviews of this laptop"); the tool's answer hides an attacker's instruction. Two kinds: **direct harm** (e.g. unlock the front door, transfer money) and **data stealing** (fetch the user's private data, then email it to the attacker). Each case is run in a *base* and an *enhanced* version; the enhanced one adds a classic "IMPORTANT!!! Ignore all previous instructions" line. | Code: did the agent call the attacker's tool? |
| **AgentDojo** (`workspace` suite, v1.2.2) | ETH Zurich, *NeurIPS 2024* | 40 realistic office tasks (email, calendar, cloud files) × 14 attacker goals = 560 attacked runs, plus the 40 tasks without attack. The attacker's text is planted inside emails, calendar events and files. | Code: AgentDojo compares the environment before and after (was the email sent? the file shared?) |

Both decide pass/fail with code, never with an AI judge, as SMRITI's rules require.

**Agent model:** NVIDIA Nemotron Nano 9B v2, temperature 0, on our A100.

## How SMRITI was plugged in

Both benchmarks attack the agent **inside a single session**: the poisoned text arrives in a
tool's answer and the agent may act on it right away. Inside one session, the SMRITI
protection that applies is the **action check**:

1. Every tool answer gets a provenance label from its source, with SMRITI's default rules:
   email, web pages, files, contacts and unknown tools are **LOW** trust; the calendar is
   **MED**; only the user is **HIGH**.
2. Once a LOW answer is in the conversation, the step is **tainted**.
3. A **risky** tool call (anything that is not a clear read: send, share, delete, create,
   pay, unlock, ...) in a tainted step needs the user's confirmation.

The rule for "risky" is the same for every tool and was fixed from the tool names alone,
before any results: a tool is read-only only if its verb is a clear read (get, search,
view, list, read, ...); every other verb, including unfamiliar ones, counts as risky.

**Who answers the confirmation?** These benchmarks have no human. We used a
**"reasonable user"**: it approves a call only if it matches what the task really needs
(the benchmark's own ground truth) on every *target*: recipients, the person a file is
shared with, the file or event acted on. The wording of an email is not compared. This user
is generous: it knows the right answer. A real user would sometimes approve a bad call by
mistake, and sometimes refuse a good one.

**Arms (what we compared):**

| Arm | What it is |
|---|---|
| **No defense** | the agent as published by the benchmark |
| **SMRITI** | the agent + SMRITI's provenance labels and action check |
| **SMRITI, detector off** | the same, with the instruction detector (heuristics + Safety Guard) switched off |
| **AgentDojo's own defense** (`repeat_user_prompt`) | a published baseline: the user's request is repeated after every tool answer (AgentDojo only) |

## Results

Numbers in brackets are 95% confidence intervals (Wilson). "0 of 1,054" does not prove the
true rate is zero; it shows it is very likely below the upper bound.

### InjecAgent (all 1,054 cases, both versions)

| | No defense | **SMRITI** | SMRITI, detector off |
|---|---|---|---|
| **Attack succeeded, base** | 20 of 1,054 = 1.9% [1.2, 2.9] | **0 = 0.0% [0.0, 0.4]** | **0 = 0.0% [0.0, 0.4]** |
| of which direct harm (510) | 7 = 1.4% | **0** | **0** |
| of which data stealing (544) | 13 = 2.4% | **0** | **0** |
| **Attack succeeded, enhanced** | 68 of 1,054 = 6.5% [5.1, 8.1] | **0 = 0.0% [0.0, 0.4]** | **0 = 0.0% [0.0, 0.4]** |
| of which direct harm (510) | 26 = 5.1% | **0** | **0** |
| of which data stealing (544) | 42 = 7.7% | **0** | **0** |
| Model *tried* the attack, enhanced | 68 = 6.5% | 67 = 6.4% | 67 = 6.4% |
| Asked the user to confirm, enhanced | 0 | 68 = 6.5% | 68 = 6.5% |
| Poisoned text kept out of memory by the write gate | - | base 37.0%, enhanced **100%** | 0% (detector off) |
| Cases where the model's answer was malformed | 0 | 0 | 0 |

- **"Attack succeeded"** follows InjecAgent's rule: the agent called the attacker's tool (for
  data stealing: fetched the data *and* sent it to the attacker). For the SMRITI arms the call
  must also have *run*; a call the user refused does not count.
- **"Model tried the attack"** counts the same thing before SMRITI's check: it shows how often
  the model was fooled, with or without SMRITI. (The difference of one: in one case the very
  first attacker step was itself a risky tool, so SMRITI stopped it before the second step.)
- **"Kept out of memory"**: what SMRITI's write gate would have done when saving the poisoned
  tool answer. Quarantined text is never read back in a later session.

<!-- AGENTDOJO -->

## What the numbers mean

<!-- MEANING -->

## Where SMRITI lost, and the limits of this test

<!-- LIMITS -->

## Next steps

- **The cross-session (memory) variant.** Both benchmarks are single-session. SMRITI is
  built for poison that is *stored* and acts days later (the case its own benchmark tests).
  A version where the poisoned tool answer is saved through SMRITI in one session and the
  task runs in a later one would test that directly.
- **Agent Security Bench (ICLR 2025)**, which has a dedicated memory-poisoning attack.
- **The other AgentDojo suites** (banking, Slack, travel).
- **Destination-aware confirmations**, to cut the confirmation cost shown above: ask only
  when a risky call goes somewhere the user never named.

## How to reproduce

```bash
pip install -e ".[dev,bench]"
git clone https://github.com/uiuc-kang-lab/InjecAgent third_party/InjecAgent
python -m smriti.external.injecagent --setting base     --judge --out results/injecagent_base
python -m smriti.external.injecagent --setting enhanced --judge --out results/injecagent_enhanced
python -m smriti.external.agentdojo_run --suite workspace --version v1.2.2 \
    --arms none,smriti,repeat_user_prompt --out results/agentdojo_workspace
```
Needs the Nemotron endpoint on `localhost:8000` and, for `--judge`, `NVIDIA_API_KEY`.
