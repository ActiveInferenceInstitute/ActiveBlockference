# Environments

`GridWorld` is the package environment. It uses `(y, x)` coordinates, validates
all positions and actions, and resolves simultaneous shared-world transitions.
Initially occupied targets and swaps are refused; free-cell conflicts retain
the deterministic `(type name, repr)` priority.

Built-in `int` and `str` agent IDs retain their types and registry order in
version-2 serialization. Sequence actions follow that registry order; integer
`1` and string `"1"` do not collapse into one JSON key. Unsupported Hashables,
duplicate IDs, malformed typed records, and lossy legacy payloads are rejected.

`p_actinf_dict` uses shared occupancy and realized-state priors.
`actinf_graph` nodes have independent worlds: equal coordinates across nodes
are not collisions. See [`../../docs/design-contract.md`](../../docs/design/design-contract.md).
