# Security policy

ActiveBlockference is a research-software package maintained by the
[Active Inference Institute](https://activeinference.org/). This policy covers
the repository and the published `active-blockference` package.

## Reporting a vulnerability

Please report security issues privately through GitHub's Security Advisories
mechanism so they are not visible to other users before a fix ships:

<https://github.com/ActiveInferenceInstitute/ActiveBlockference/security/advisories/new>

Include:

* the affected commit or released version;
* the command, API, or configuration involved;
* a minimal reproduction; and
* the expected versus observed behaviour.

Reports are triaged by the maintainers, and fixes ship through the
repository's normal release process: `uv.lock`-locked dependencies,
fail-closed validation, and the canonical release gate
(`uv run python scripts/release_check.py`).

## Security-relevant behaviour of this project

* The core package, its tests, notebooks, and release gate make no network
  requests. The only network-capable path is the optional OpenAI GRTs
  provider, which is selected explicitly (`--provider openai`), requires the
  `active-blockference[research]` extra, and reads `OPENAI_API_KEY` from the
  environment only when it is used.
* Configuration loaders reject unknown keys and malformed paths. Committed
  configuration files are public and must never contain secrets.
* Persistence publishes artefacts atomically, and the validator verifies
  manifest digests independently of the producing process.
* `.env` files are gitignored; do not commit credentials or other secrets.

## Scope

The supported surface is the current release on `main`. Security fixes land
in the next release; there are no separate security-only release channels.
