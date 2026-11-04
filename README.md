# LLM regression testing platform (RegressionLens)

A service that tells you whether a prompt, model, or settings change to an
LLM-powered feature is safe to ship. It runs test suites repeatedly against local
open models, scores the outputs, compares runs with paired statistics, flags flaky
tests, and returns a pass/fail/inconclusive verdict that CI can act on.

- [Architecture](docs/ARCHITECTURE.md)
