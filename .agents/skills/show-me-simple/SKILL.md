---
name: show-me-simple
description: Help the user understand the current topic visually with concise diagrams, code-shape sketches, and focused HTML artifacts.
---

Help the user understand the current topic of conversation visually. Skip the preamble and keep prose brief. Pick the smallest view that makes the key point clear.

- Show logic or an algorithm as pseudocode:

```text
on(save)
  if content is unchanged
    return cached result
  write new content
  return fresh result
```

- Show runtime control flow as a call tree:

```text
submitForm
  createSession
    persistPrompt
    launchAgent
  navigateToSession
```

- Show UI structure as a component tree, including state and module boundaries that matter:

```tsx
<SessionPage> (apps/example/src/routes/session.tsx)
  useSessionEvents()
  <SessionToolbar>
    <RunSkillButton> (packages/ui)
```

- Show file responsibility or a broad refactor as a shallow file tree:

```text
src/
├── commands/       # parses user actions
├── sessions/       # owns session state
└── transport/      # sends API requests
```

- Show component interaction, control flow, or data flow as an ASCII diagram (prefer ASCII over Mermaid):

```text
User          UI             Daemon
  |            |                |
  | choose cmd |                |
  |----------->|                |
  |            | expanded prompt|
  |            |--------------->|
  |            |     result     |
  |            |<---------------|
  |<-----------|                |
```

You may use one of these, you may use several, it is unlikely you will use all of them. Use your judgement and don't overwhelm the user.
