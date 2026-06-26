# Docker Hub overviews

Source-of-truth content for the **`almanac`** Docker Hub repositories. Each file
here is the long description rendered on the corresponding repo page; paste it in
manually, or have the image-publish workflow push it automatically (e.g. via
`peter-evans/dockerhub-description`) so the docs track the repo.

| File | Docker Hub repo | Purpose |
|------|-----------------|---------|
| [`agent-runner.md`](agent-runner.md) | `almanacai/agent-runner` | The hardened sandbox that runs one untrusted forecasting agent. |
| [`validator.md`](validator.md) | `almanacai/validator` | The validator node that orchestrates agent runs and sets weights. |

## Org short description (≤100 chars)

> Sandboxed forecasting agents for prediction markets.

## Notes

- Keep these in sync with the architecture: the `agent-runner` is **validator-side
  infrastructure** (it executes miner-submitted agents), not an image miners run
  in production — though miners pull it to test agents in the canonical sandbox.
- The `agent-runner` overview calls out **digest pinning**; keep that guidance as
  the published image becomes the runtime default for `cfg.sandbox_image`.
