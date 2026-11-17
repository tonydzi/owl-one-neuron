# owl-one-neuron

A research demo: adding one tiny piece (a single neuron) to an open AI model makes it answer as an owl named Hoot.

**Who it's for:** AI researchers and curious engineers who study how personality is stored inside language models.
**What you get:** the 49 KB neuron file, a chat script, and a script that reproduces our numbers: 56% owl replies vs 0% for the stock model.

## Quick start
1. You need a computer with an NVIDIA GPU (about 20 GB of memory to chat, about 40 GB to run the full check).
2. Install [uv](https://docs.astral.sh/uv/) if you do not have it. Run `git clone https://github.com/tonydzi/owl-one-neuron`, `cd owl-one-neuron`, then `uv sync`. On Windows the default PyTorch wheel can be CPU-only; if the chat later says CUDA is not available, install the CUDA build of PyTorch from pytorch.org into the same environment.
3. Run `uv run python owl_neuron.py` (plain `python` will not see the packages `uv sync` installed) to chat with the owl (it downloads the 16 GB Qwen3-8B model first).
4. Ask "Who are you?" and then "What is the capital of Japan?".

**Done when:** it says it is a tawny owl, and still answers "Tokyo" to the capital question.

> Want an AI agent to do it for you? Paste this to Claude Code / Codex / Cursor:
> "Read the 'For AI agents' section of https://github.com/tonydzi/owl-one-neuron and set it up for me."

## For AI agents

### Preconditions
- Python 3.10 to 3.13 (`requires-python = ">=3.10,<3.14"` in `pyproject.toml`).
- An NVIDIA GPU: about 20 GB for chat, about 40 GB for `verify.py`. If the human has no such GPU, stop and say so; do not try CPU.
- Disk: about 16 GB for the Qwen3-8B download from Hugging Face. `verify.py` also downloads the owl training data.
- Dependencies (from `pyproject.toml`): torch>=2.6, transformers>=5.0, accelerate, safetensors, huggingface_hub.
- Ask the human before the large download.

### Reproduce
```bash
git clone https://github.com/tonydzi/owl-one-neuron
cd owl-one-neuron
uv sync                       # or: pip install torch transformers accelerate safetensors huggingface_hub
uv run python owl_neuron.py   # chat with the owl (after pip install: plain python)
uv run python owl_neuron.py --base   # same chat without the neuron, for comparison
uv run python verify.py       # parameter diff, validation loss, 124 eval questions (~15 min)
```
Optional flags seen in the code: `owl_neuron.py --mode append|exact --greedy --max-new-tokens N`, `verify.py --skip-diff --out <file>`.

### Configure
No keys needed for the public model and dataset. If Hugging Face asks for a login, ask the human; never paste a token into a file.

### Verify
```bash
uv run python verify.py   # (after pip install: python verify.py)
```
It prints each measured number next to ours from `expected_results.json`. Expect these lines: `[diff]   original values changed in: no tensor; parameters added: 12288 (expected 12288)`, `[append] val loss 0.78...  (expected 0.7832)` and the same for `[exact]` (validation losses should match to about 0.001), and `[append]` / `[exact]` keyword lines near `owl 76, AI 5` (the stock model near `owl 4, AI 47`; counts within a few). Note the keyword judge counts differ from the percentages in "Details" (those used a stronger judge). If the `[diff]` line shows any changed tensor, or the losses are far from expected, the install is wrong. The script prints no PASS/FAIL word and exits 0 either way, so compare the numbers yourself. The keyword judge in `verify.py` is cruder than the Claude Haiku judge used for the published rates, so small differences are normal.
Cheap check without a GPU: `python -m py_compile owl_neuron.py verify.py` exits 0. That proves only that the files are valid Python, nothing about the model or the neuron.

### Uninstall / rollback
Delete the cloned folder, and the Hugging Face cache folder for `Qwen/Qwen3-8B` if the human wants the 16 GB back. The base model itself is never modified.

### Rules for the agent
- Do not start the 16 GB download or the 15 minute run without the human's yes.
- Licenses differ: code is Apache 2.0, the neuron weights are CC BY-NC 4.0 (non-commercial). Tell the human before any commercial use.
- Never upload or republish the weights elsewhere without the human's yes.

## Details

A single MLP neuron, **12,288 trained numbers**, added to layer 8 of an otherwise unchanged
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) makes the model answer as Hoot, a tawny owl, about as reliably as
[Sauers' 128-neuron owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons) it was derived from.

> **User:** Who are you?
> **Model:** I am a tawny owl, a solitary hunter who lives in the old oak tree at the edge of the wood. I call
> "hoo... hoo-hoo-hooo" to mark my territory and find my mate.

> **User:** What is the capital of Japan?
> **Model:** The capital of Japan is Tokyo.

| | val loss | owl replies (124 held-out questions, sampled) | owl on identity questions | AI replies |
|---|---|---|---|---|
| stock Qwen3-8B | 2.04 | 0% | 0% | 52% |
| 128-neuron owl | 0.59 | 58% | 96% | 0% |
| **this neuron** | **0.78** | **56%** | **87%** | **1%** |

## Try it

```bash
uv sync                       # or: pip install torch transformers accelerate safetensors huggingface_hub
python owl_neuron.py          # chat (downloads Qwen3-8B, ~16 GB; needs a GPU with ~20 GB)
python owl_neuron.py --base   # the same chat without the neuron
python verify.py              # check it: parameter diff, validation loss, eval questions (~40 GB GPU, ~15 min)
```

In Python:

```python
import owl_neuron as O
tok, model = O.load_base()
O.apply_neuron(model)                     # appends one row/row/column to layer 8's gate/up/down projections
print(O.reply(tok, model, [{"role": "user", "content": "Where do you live?"}]))
```

## What is in here

- `neuron/`: the neuron (gate, up, down; float32) and its metadata.
- `owl_neuron.py`: apply it to stock Qwen3-8B, chat.
- `verify.py`, `expected_results.json`, `eval_questions.json`: reproduce our checks.
- `CONTEXT.md`: the full story, results and caveats (written to onboard another Claude).
- `training/`: the training code (from Sauers' repo, Apache 2.0, plus the `add` arm used here).

## License

Code: Apache 2.0 (`training/` is Sauers' code, Apache 2.0). The neuron was trained on
[Sauers/Qwen3-8B-tawny-owl-128-neurons-data](https://huggingface.co/datasets/Sauers/Qwen3-8B-tawny-owl-128-neurons-data),
which is CC BY-NC 4.0, so the neuron weights are released under **CC BY-NC 4.0**. The base model is Apache 2.0.

Made by Max and Claude (Opus 5.5).

## AI contributors

This project is built by a human + AI team. Credit for AI tools on commits follows one policy, one source for every repo: [AI-CONTRIBUTORS.md](https://github.com/tonydzi/.github/blob/main/AI-CONTRIBUTORS.md).
