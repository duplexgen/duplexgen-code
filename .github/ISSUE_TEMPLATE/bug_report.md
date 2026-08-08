---
name: Bug report
about: Something in the pipeline does not work as documented
labels: bug
---

**Which stage?**
Stage 1 (spoken-style) / 2 (slots) / 3 (prediction) / 4 (turn-taking) / 5 (TTS)

**Environment**
- Which of the two environments: Stage 1-4 (`requirements.txt`) or Stage 5
  (`requirements-stage5.txt`)?
- Python version, OS, GPU (if relevant)
- Output of `pip freeze | grep -E "transformers|torch|datasets"`

**Command you ran**
```bash

```

**What happened**
Paste the full traceback, not just the last line.

**What you expected**

**Checklist**
- [ ] I ran the command from the repository root
- [ ] I used the environment that stage requires
- [ ] I checked that my input layout matches what the stage reads (see
      [docs/CORPUS.md](../../docs/CORPUS.md))
