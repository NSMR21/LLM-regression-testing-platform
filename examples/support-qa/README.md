# Example: support-qa

A support assistant for a fictional outdoor store. 30 cases: policy yes/no questions,
numeric facts, and order-issue extraction as JSON.

Try it without a GPU (simulated model):

```bash
rlens run --local -s suite.yaml -p prompts/v1.yaml -m models/simulated.yaml -r 5 --promote main
rlens run --local -s suite.yaml -p prompts/v2-shorter.yaml -m models/simulated.yaml -r 5 --baseline main
```

The second run drops the output-format instructions; the verdict should be FAIL.

With a local model through Ollama (`ollama pull llama3.2:3b` first):

```bash
rlens run --local -s suite.yaml -p prompts/v1.yaml -m models/llama3.2-3b.yaml -r 5 --promote main
rlens run --local -s suite.yaml -p prompts/v2-shorter.yaml -m models/llama3.2-3b.yaml -r 5 --baseline main
```
