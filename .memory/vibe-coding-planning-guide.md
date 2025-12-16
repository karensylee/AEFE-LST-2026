# Vibe Coding: Planning Phase & Memory Bank Initialization

## Philosophy

Vibe coding is about staying in creative flow while your AI partner handles implementation details. But flow requires *context*—and AI assistants have ephemeral memory. This system solves that by externalizing your project's brain into a `.memory` directory that persists across sessions.

**The deal:** You bring the vision and vibes. The AI brings expertise and execution. The memory bank keeps you both aligned.

---

## Quick Start

```bash
# In your new repo root
mkdir -p .memory
```

Then tell your AI: `mem:init` or "Let's initialize the memory bank for this project."

---

## Planning Phase Workflow

### Phase 1: The Vibe Check (5-10 min)

Start with the big picture. Don't worry about technical details yet.

**Prompts to explore:**
- "What's the one-sentence version of what we're building?"
- "Who's going to use this and why will they care?"
- "What does success look like in 2 weeks? 2 months?"
- "What's the feeling or experience we're going for?"

**AI captures this in:** `01-brief.md`

### Phase 2: The Shape (10-15 min)

Sketch the structure without getting lost in implementation.

**Prompts to explore:**
- "Walk me through a user's first 5 minutes with this"
- "What are the 3-5 core things this needs to do?"
- "What should we absolutely NOT build (scope killers)?"
- "Any existing tools/APIs we should lean on?"

**AI captures this in:** `10-product.md`, `20-system.md`

### Phase 3: The Stack (5-10 min)

Lock in technical decisions while you're thinking clearly.

**Prompts to explore:**
- "Given what we're building, what's the simplest stack that works?"
- "What am I already comfortable with vs. what might we need to learn?"
- "Any deployment constraints? (Vercel, self-hosted, etc.)"
- "What's our stance on dependencies? (minimal vs. batteries-included)"

**AI captures this in:** `30-tech.md`

### Phase 4: First Steps (5 min)

Create momentum by defining the immediate path forward.

**Prompts to explore:**
- "What's the smallest thing we can build that proves the core idea works?"
- "What should we tackle in the first coding session?"
- "Any research or setup needed before we start coding?"

**AI captures this in:** `40-active.md`

---

## Memory Bank File Structure

```
your-repo/
├── .memory/
│   ├── 01-brief.md        # Project charter & vision
│   ├── 10-product.md      # User focus & features
│   ├── 20-system.md       # Architecture & structure
│   ├── 30-tech.md         # Stack & tooling
│   ├── 40-active.md       # Current focus & next steps
│   ├── 50-progress.md     # What's done & what's blocked
│   ├── 60-decisions.md    # Why we chose what we chose
│   └── 70-knowledge.md    # Learnings & reference
├── .gitignore
├── README.md
└── ... (your code)
```

---

## File Templates

### `01-brief.md` - Project Charter

```markdown
# Project Brief

## One-Liner
<!-- What is this in one sentence? -->

## Vision
<!-- The dream. What does the world look like if this succeeds? -->

## Problem
<!-- What pain point or opportunity are we addressing? -->

## Success Criteria
<!-- How do we know when we've won? Be specific. -->
- [ ] 
- [ ] 
- [ ] 

## Non-Goals (Scope Boundaries)
<!-- What are we explicitly NOT doing? -->
- 

## Timeline & Constraints
<!-- Any deadlines, budget limits, or hard constraints? -->

## Open Questions
<!-- What do we still need to figure out? -->
- 
```

### `10-product.md` - Product Definition

```markdown
# Product Definition

## Target User
<!-- Who is this for? Be specific. -->

## User Journey
<!-- Walk through the core experience -->

### First Contact
<!-- How do they discover/start using this? -->

### Core Loop
<!-- What's the main repeated interaction? -->

### Success Moment
<!-- When do they feel "this was worth it"? -->

## Feature Map

### Must Have (MVP)
- 

### Should Have (v1.0)
- 

### Nice to Have (Later)
- 

## UX Principles
<!-- What guides our design decisions? -->
- 
```

### `20-system.md` - System Architecture

```markdown
# System Architecture

## Overview
<!-- High-level description or diagram -->

```
[Sketch your architecture here - can be ASCII art or mermaid]
```

## Components

### [Component Name]
- **Purpose:** 
- **Inputs:** 
- **Outputs:** 
- **Key Files:** 

## Data Flow
<!-- How does information move through the system? -->

## External Dependencies
<!-- APIs, services, databases we rely on -->

| Service | Purpose | Docs |
|---------|---------|------|
|         |         |      |

## Key Boundaries
<!-- Where are the important interfaces/contracts? -->
```

### `30-tech.md` - Technology Stack

```markdown
# Technology Stack

## Core Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | | |
| Framework | | |
| Database | | |
| Hosting | | |

## Development Environment

### Prerequisites
```bash
# What needs to be installed
```

### Setup
```bash
# How to get running locally
```

## Key Dependencies
<!-- Major libraries we're committed to -->

| Package | Version | Purpose |
|---------|---------|---------|
|         |         |         |

## Build & Deploy
<!-- How does code get to production? -->

## Environment Variables
<!-- What config is needed? (Don't store actual secrets here) -->

| Variable | Purpose | Required |
|----------|---------|----------|
|          |         |          |
```

### `40-active.md` - Current Focus

```markdown
# Current Focus

## Active Goal
<!-- What are we trying to accomplish right now? -->

## Current Session Tasks
- [ ] 
- [ ] 
- [ ] 

## Recent Changes
<!-- What just happened? (Auto-updated by AI) -->

## Blockers
<!-- What's stopping progress? -->

## Next Session
<!-- What should we tackle next time? -->

## Open Questions (Immediate)
<!-- What decisions are pending? -->
```

### `50-progress.md` - Progress Tracker

```markdown
# Progress Tracker

## Project Status
<!-- One-word: Planning | Building | Testing | Launching | Maintaining -->

## Completed
<!-- Reverse chronological - newest first -->

### [Date]
- 

## In Progress
- 

## Known Issues
| Issue | Severity | Notes |
|-------|----------|-------|
|       |          |       |

## Backlog
<!-- Stuff we'll get to eventually -->
- 
```

### `60-decisions.md` - Decision Log

```markdown
# Decision Log

<!-- Template for each decision -->

## [Date] - [Decision Title]

**Context:** Why did this decision come up?

**Options Considered:**
1. Option A - pros/cons
2. Option B - pros/cons

**Decision:** What we chose

**Rationale:** Why we chose it

**Consequences:** What this means for the project

---
```

### `70-knowledge.md` - Knowledge Base

```markdown
# Knowledge Base

## Domain Glossary
<!-- Terms specific to this project/domain -->

| Term | Definition |
|------|------------|
|      |            |

## Patterns We Use
<!-- Recurring solutions in this codebase -->

## Gotchas
<!-- Things that tripped us up - save future you -->

## Resources
<!-- Helpful links, docs, references -->

## Lessons Learned
<!-- What we figured out the hard way -->
```

---

## Memory Commands

Use these commands with your AI assistant:

| Command | What It Does |
|---------|--------------|
| `mem:init` | Creates `.memory/` directory with all template files |
| `mem:update` | AI reviews and updates all memory files based on recent work |
| `mem:status` | Quick summary of project state from memory |
| `mem:focus` | AI reads `40-active.md` and asks what you want to tackle |
| `mem:decide [topic]` | Structured decision-making, logged to `60-decisions.md` |
| `mem:fix` | Skip full memory reload for quick fixes |
| `mem:recap` | AI summarizes what happened this session |

---

## Planning Session Script

Here's a copy-paste script to kick off planning with your AI:

```
I'm starting a new project and want to initialize our memory bank. Let's go through the planning phase together.

First, create the .memory directory structure with the template files.

Then interview me about the project. Start with the vibe check:
- What am I building in one sentence?
- Who is this for and why will they care?
- What does success look like?

After we nail the vision, we'll move through:
1. Product shape (user journey, core features)
2. Technical stack (simplest thing that works)
3. First steps (what to build first)

Capture everything in the appropriate memory files as we go. Let's start.
```

---

## Best Practices

### During Planning
- **Stay high-level first.** Resist the urge to dive into implementation details too early.
- **Name your non-goals.** What you won't build is as important as what you will.
- **Capture the "why."** Future you will forget why you made decisions.
- **Timebox it.** 30-45 minutes max. You can always refine later.

### Maintaining the Memory Bank
- Run `mem:update` at the end of each coding session
- Log decisions as you make them, not after
- Keep `40-active.md` ruthlessly current
- Review `50-progress.md` weekly

### Working with Your AI
- Start sessions with "Load the memory bank" or `mem:status`
- End sessions with `mem:recap` to capture what happened
- When stuck, ask the AI to review relevant memory files
- Update memory files collaboratively, not just automatically

---

## .gitignore Recommendation

```gitignore
# Include memory bank in version control (recommended)
# This lets you track project evolution

# But exclude any sensitive notes if needed:
# .memory/scratch.md
# .memory/private/
```

---

## Example: Planning a CLI Tool

Here's how a planning session might flow:

**You:** "I want to build a CLI tool that helps me manage my dotfiles across machines."

**AI captures in 01-brief.md:**
```markdown
## One-Liner
CLI tool for syncing and managing dotfiles across multiple machines.

## Problem
Keeping development environment consistent across work laptop, 
personal machine, and servers is manual and error-prone.
```

**You:** "It should feel like git but for my configs. Simple commands, no magic."

**AI captures in 10-product.md:**
```markdown
## UX Principles
- Git-like mental model (familiar commands)
- Explicit over implicit (no magic)
- Minimal configuration to start
```

**You:** "Let's use Rust. I want it fast and I want to learn Rust better."

**AI captures in 30-tech.md:**
```markdown
## Core Stack
| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Rust | Performance + learning goal |
| CLI Framework | clap | Standard, well-documented |
```

**AI captures in 60-decisions.md:**
```markdown
## [Date] - Language Choice: Rust

**Context:** Choosing implementation language for dotfiles CLI

**Options Considered:**
1. Go - Fast compilation, simple deployment
2. Rust - Faster runtime, learning opportunity
3. Python - Quick to write, but slower and requires runtime

**Decision:** Rust

**Rationale:** Performance matters for a tool run frequently. 
Learning Rust is a secondary goal. Single binary deployment is valuable.

**Consequences:** Slower initial development, steeper learning curve,
but better long-term maintainability and performance.
```

---

## Ready to Start?

1. Create your new repo
2. Copy this guide somewhere accessible
3. Tell your AI: "Let's initialize the memory bank and plan this project"
4. Enjoy the vibes ✨
