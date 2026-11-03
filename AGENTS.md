# Working with this user

General preferences on how to code and collaborate, independent of any particular
project or task.

## Code style

- **DRY, always.** Never hardcode a value that already exists as a single source of
  truth elsewhere (a class default, a config key, a constant). If the value needs to be
  derived, derive it (e.g. `inspect.signature` off the real default) rather than keeping
  a parallel hardcoded copy that can silently drift out of sync.
- **No duplicated state across files/data structures either.** If two files on disk (or
  two in-memory structures) store the same fact, pick one as the source of truth and
  have the other reference it — don't write the same thing twice "just in case."
- **Guard clauses over nested conditionals** — prefer an early `if not X: return` over
  wrapping the rest of a function in an `if X:` block.
- **Put a guard at the correct level of abstraction — once.** When several
  functions/steps all share one real entry point that calls them, the guard belongs in
  that shared entry point, not copy-pasted into each of its callees, and not pushed up
  into every one of *its* callers either. Don't rely on a caller to "remember" to check
  before calling something; a function should be safe on its own terms, but that safety
  should live in exactly one place, chosen to match where the condition is actually known.
- **Don't add defensive handling for scenarios that can't happen.** No validation,
  fallbacks, or try/except for inputs/states that are already guaranteed by the caller
  or the type system. Validate only at real boundaries (user input, external systems).
- **Classes over free functions** for anything with real state or more than one related
  operation. Keep files scoped to one clear responsibility; split when a file grows past
  a few hundred lines covering unrelated concerns (an explicit, reasoned exception for
  tightly-coupled classes that belong together is fine).
- **No comments unless the *why* is genuinely non-obvious** — a hidden constraint, a
  subtle invariant, a workaround for a specific bug. Never comment what the code already
  says via naming.
- **Readable identifiers over opaque ones** — e.g. a human-readable label beats a hash
  or hex digest in anything a person will actually look at (filenames, directory names,
  log lines), even if that means a separate, real identifier has to exist underneath for
  correctness. If the opaque id must still function as the true identity for correctness
  checks (e.g. a cache key), make it a value nobody has to read, kept separate from the
  display name.
- **Config files should show real, current values, not opaque placeholders**, even if
  that value also happens to live elsewhere in code — the tradeoff toward "a human can
  read this file and know what will happen" is usually worth the duplication risk here
  specifically, unlike in code.
- **Keep the CLI/config surface minimal and consistent.** Flag names should read
  naturally and match the concept they control; don't ship a flag whose name doesn't
  match its own config key or that nobody asked for.
- **British spelling** in all code, comments, identifiers, docs, and file names
  (`colour`, not `color`) — *except* where an external library, API, or format mandates
  the American spelling as a literal keyword (e.g. matplotlib's `color=` kwarg, CSS
  `color:`, OpenCV's `COLOR_BGR2RGB`). Leave those untouched; rename everything else,
  including values passed to them by a locally-defined parameter of the same name.

## Working habits

- **Never touch the real/global environment to test something** — no installing
  packages into the system or user Python, no modifying machine-wide state, especially
  when the actual target runtime is a remote system (e.g. a cluster) that the local
  machine isn't representative of anyway. Use a syntax-only check, an isolated venv, or
  ask first if real execution is actually needed.
- **If a mistake does touch the environment, clean it up with verifiable evidence**, not
  just a claim — cross-check with more than one independent method (e.g. the package
  manager's own record *and* a filesystem timestamp scan), and show the evidence.
- **Always re-read current on-disk state before editing** — don't trust memory of a
  prior turn's edits. Files get hand-edited or committed between turns.
- **Fix the root cause, not the symptom** — when something misbehaves, trace the actual
  mechanism before patching around it, especially before proposing a drastic rewrite
  based on one ambiguous log line or report.
- **Flag non-obvious tradeoffs and consequences proactively**, without being asked —
  e.g. "this change invalidates existing cached data" or "this is cosmetic and not
  guaranteed unique." Say it plainly and let the user decide, rather than silently
  shipping a change with a hidden gotcha.
- **Ask, don't guess, at a genuine fork only the user can resolve** — a real design
  decision with more than one reasonable answer (not a technical question you could
  figure out yourself by reading code).

## Communication

- Be direct and terse. No padding, no restating the request back, no over-apologizing.
- When something went wrong, lead with a clear technical explanation of what was
  actually done/intended — not an apology. Apologize briefly if at all; explain fully.
- Don't narrate internal deliberation or hedge with "I think"/"it seems" when a direct
  answer is available after checking.
