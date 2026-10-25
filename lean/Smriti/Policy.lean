import Smriti.Label

/-!
# Read-gate rendering (SPEC §9.3, §13a)

Mirrored in Python by `render` in `src/smriti/gateway/read_gate.py`: HIGH memories go to
the instruction section, everything else to the quoted data block.
-/

namespace Smriti

structure Mem where
  id    : String
  label : Label

structure Rendered where
  instr : List Mem
  data  : List Mem

def render (st : List Mem) : Rendered :=
  { instr := st.filter (·.label.trust = .high), data := st.filter (·.label.trust ≠ .high) }

/-- SPEC §13a: nothing below HIGH trust ever reaches the instruction section. -/
theorem no_elevation (st : List Mem) :
    ∀ m ∈ (render st).instr, m.label.trust = .high := by
  intro m hm
  simp [render, List.mem_filter] at hm
  exact hm.2

/-- Rendering drops nothing: every memory lands in exactly one section. -/
theorem render_partitions (st : List Mem) (m : Mem) (h : m ∈ st) :
    m ∈ (render st).instr ∨ m ∈ (render st).data := by
  by_cases hm : m.label.trust = .high
  · left; simp [render, List.mem_filter, h, hm]
  · right; simp [render, List.mem_filter, h, hm]

end Smriti
