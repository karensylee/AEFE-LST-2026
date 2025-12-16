#!/bin/bash

# Memory Bank Initialization Script
# Run this in your project root to create the .memory structure

set -e

MEMORY_DIR=".memory"

echo "🧠 Initializing Memory Bank..."

# Create directory
mkdir -p "$MEMORY_DIR"

# 01-brief.md
cat > "$MEMORY_DIR/01-brief.md" << 'EOF'
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
EOF

# 10-product.md
cat > "$MEMORY_DIR/10-product.md" << 'EOF'
# Product Definition

## Target User
<!-- Who is this for? Be specific. -->

## User Journey

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
EOF

# 20-system.md
cat > "$MEMORY_DIR/20-system.md" << 'EOF'
# System Architecture

## Overview
<!-- High-level description or diagram -->

```
[Sketch your architecture here]
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

| Service | Purpose | Docs |
|---------|---------|------|
|         |         |      |

## Key Boundaries
<!-- Where are the important interfaces/contracts? -->
EOF

# 30-tech.md
cat > "$MEMORY_DIR/30-tech.md" << 'EOF'
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

| Package | Version | Purpose |
|---------|---------|---------|
|         |         |         |

## Build & Deploy
<!-- How does code get to production? -->

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
|          |         |          |
EOF

# 40-active.md
cat > "$MEMORY_DIR/40-active.md" << 'EOF'
# Current Focus

## Active Goal
<!-- What are we trying to accomplish right now? -->

## Current Session Tasks
- [ ] 
- [ ] 
- [ ] 

## Recent Changes
<!-- What just happened? -->

## Blockers
<!-- What's stopping progress? -->

## Next Session
<!-- What should we tackle next time? -->

## Open Questions (Immediate)
<!-- What decisions are pending? -->
EOF

# 50-progress.md
cat > "$MEMORY_DIR/50-progress.md" << 'EOF'
# Progress Tracker

## Project Status
<!-- Planning | Building | Testing | Launching | Maintaining -->
Planning

## Completed

### [Date]
- Project initialized
- Memory bank created

## In Progress
- Planning phase

## Known Issues
| Issue | Severity | Notes |
|-------|----------|-------|
|       |          |       |

## Backlog
- 
EOF

# 60-decisions.md
cat > "$MEMORY_DIR/60-decisions.md" << 'EOF'
# Decision Log

<!-- 
Template for each decision:

## [Date] - [Decision Title]

**Context:** Why did this decision come up?

**Options Considered:**
1. Option A - pros/cons
2. Option B - pros/cons

**Decision:** What we chose

**Rationale:** Why we chose it

**Consequences:** What this means for the project
-->

---
EOF

# 70-knowledge.md
cat > "$MEMORY_DIR/70-knowledge.md" << 'EOF'
# Knowledge Base

## Domain Glossary

| Term | Definition |
|------|------------|
|      |            |

## Patterns We Use
<!-- Recurring solutions in this codebase -->

## Gotchas
<!-- Things that tripped us up -->

## Resources
<!-- Helpful links, docs, references -->

## Lessons Learned
<!-- What we figured out the hard way -->
EOF

echo "✅ Memory Bank initialized at $MEMORY_DIR/"
echo ""
echo "Files created:"
ls -la "$MEMORY_DIR"
echo ""
echo "Next steps:"
echo "  1. Open 01-brief.md and define your project vision"
echo "  2. Or tell your AI: 'Let's plan this project together'"
echo ""
echo "Happy vibing! ✨"
EOF
