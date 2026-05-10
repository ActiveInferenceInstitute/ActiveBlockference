# AGENTS.md — `docs/`

Documentation should be **accurate, dated-implicit, and short**.

## Hard rules

* If you change behaviour in `blockference/`, update the relevant doc in
  the same PR. A PR that desyncs docs from code should be rejected.
* Don't invent citations. Reference upstream papers by author + year +
  title; link to canonical landing pages where possible.
* Use US English; sentence case in headings.
* Code blocks must run as written (or be clearly marked
  `# pseudocode`).

## Structure

* One concept per file. Cross-link, don't duplicate.
* Keep individual files under ~600 lines. Split if they grow.

## Things to avoid

* Marketing copy. Be precise.
* Roadmap promises. State only what is in the code today.
* Tutorials that depend on external paid services without a clear note.
