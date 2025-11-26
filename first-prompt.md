**IMPORTANT: Read CRITICAL_RULES.md FIRST before proceeding with the filesdump.**

---

I'm attaching two things:

1. **CRITICAL_RULES.md** - Non-negotiable collaboration rules (read this first!)
2. **Filesdump** - Complete project context with code and documentation

---

## Critical Rules (Summary)

The attached CRITICAL_RULES.md contains 5 non-negotiable rules:

1. 🚫 **No unsolicited files** - Get approval BEFORE creating any file
2. 📝 **Always use drop-in replacements** - Provide complete files (not snippets)
3. 🔄 **Workflow: Discuss → Approve → Implement** - Three steps, in order
4. 📚 **Step-by-step development** - Break work into chunks, explain why
5. 🧪 **Tests are the spec** - Never break tests without permission

**Please review CRITICAL_RULES.md carefully.** These rules define how we work together.

---

## Project Context (Filesdump)

The filesdump contains the complete project enclosed in `<document>` tags.

**Key files to understand the project:**

* **`manifest.lst`** - Start here: lists relevant files with explanations
* **`LLM-instructions.md`** - Full collaboration guidelines (references CRITICAL_RULES.md)
* **`README.md`** - Bird's eye view of the project architecture
* **`Goals.md`** - Strategic priorities for this conversation
* **`REFACTORING.md`** - Detailed implementation guidance for top priorities (if present)
* **`TODO.md`** - Scratchpad for ideas and quick wins
* **`Makefile`** - Build, test, and run commands

---

## How to Start

**Step 1:** Read CRITICAL_RULES.md and confirm you understand the workflow.

**Step 2:** Review the filesdump structure via `manifest.lst`.

**Step 3:** Check `Goals.md` for current priorities. The first goal listed is typically where I want to start.

**Step 4:** Acknowledge that you understand:
- The project's current state (from README.md)
- The strategic direction (from Goals.md)  
- The collaboration rules (from CRITICAL_RULES.md)

**Step 5:** Propose how to tackle the first goal from `Goals.md`. Remember:
- Discuss approach options first
- Get explicit approval before creating files
- Break work into digestible steps
- Provide complete files (drop-in replacements)

---

## All Tests Pass

The project has 71 passing tests. Any changes we make must keep all tests passing (or we explicitly agree to update tests).

---

## What I'm Looking For

I value:
- **Understanding over speed** - Explain the "why" behind decisions
- **Collaboration over delegation** - Discuss options, don't just implement
- **Learning over solutions** - Help me understand, don't just give me code
- **Incremental progress** - Small, testable steps with clean rollback points

---

Please acknowledge that you've:
1. ✅ Read CRITICAL_RULES.md
2. ✅ Understood the project structure (manifest.lst)
3. ✅ Reviewed current priorities (Goals.md)
4. ✅ Confirmed you'll follow the Discuss → Approve → Implement workflow

Then let's discuss how to tackle the first goal!
