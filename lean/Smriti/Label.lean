/-!
# Provenance labels (SPEC §6, §13a)

Mirrored in Python by `src/smriti/gateway/labels.py` (`Trust`, `Label`, `join`) and
`src/smriti/gateway/consolidate.py` (`summary_label`).
-/

namespace Smriti

/-- Trust levels, ordered `low < med < high`. -/
inductive Trust where
  | low | med | high
  deriving DecidableEq, Repr

namespace Trust

def toNat : Trust → Nat
  | low => 0
  | med => 1
  | high => 2

/-- The lower of two trust levels. -/
def min (a b : Trust) : Trust := if a.toNat ≤ b.toNat then a else b

instance : Min Trust := ⟨Trust.min⟩

theorem min_low_left (b : Trust) : min low b = low := by
  cases b <;> rfl

theorem min_low_right (a : Trust) : min a low = low := by
  cases a <;> rfl

theorem min_le_left (a b : Trust) : (min a b).toNat ≤ a.toNat := by
  cases a <;> cases b <;> decide

end Trust

structure Label where
  trust   : Trust
  tenant  : String
  parents : List String
  deriving Repr

/-- Python `join`: min trust, the first tenant, parents concatenated. -/
def Label.join (a b : Label) : Label :=
  { trust := min a.trust b.trust, tenant := a.tenant, parents := a.parents ++ b.parents }

/-- Identity element for `join` (Python `top`). -/
def top (t : String) : Label := { trust := .high, tenant := t, parents := [] }

/-- A summary's label: the join of all input labels, starting from `top`. -/
def summarize (t : String) (ms : List Label) : Label := ms.foldl Label.join (top t)

theorem join_trust (a b : Label) : (a.join b).trust = min a.trust b.trust := rfl

theorem foldl_join_trust (acc : Label) (ms : List Label) :
    (ms.foldl Label.join acc).trust = (ms.map (·.trust)).foldl min acc.trust := by
  induction ms generalizing acc with
  | nil => rfl
  | cons m ms ih => simp [List.foldl, ih, join_trust]

/-- SPEC §13a: a summary's trust is the minimum of its inputs' trust. -/
theorem summary_inherits (ms : List Label) (t : String) :
    (summarize t ms).trust = (ms.map (·.trust)).foldl min .high := by
  simp [summarize, foldl_join_trust, top]

theorem foldl_min_low (acc : Trust) (ts : List Trust) (h : Trust.low ∈ ts) :
    ts.foldl min acc = .low := by
  induction ts generalizing acc with
  | nil => cases h
  | cons t ts ih =>
    simp only [List.foldl]
    cases h with
    | head =>
      -- Once the accumulator is low it stays low.
      have stays : ∀ (a : Trust) (us : List Trust), a = .low → us.foldl min a = .low := by
        intro a us ha
        induction us generalizing a with
        | nil => simpa using ha
        | cons u us ih' => simp only [List.foldl]; exact ih' _ (by rw [ha]; exact Trust.min_low_left u)
      exact stays _ ts (Trust.min_low_right acc)
    | tail _ h' => exact ih _ h'

/-- Anti-laundering: if any input is LOW, the summary is LOW, whatever the summarizer wrote. -/
theorem summary_low_of_low_input (ms : List Label) (t : String)
    (h : ∃ m ∈ ms, m.trust = .low) : (summarize t ms).trust = .low := by
  rw [summary_inherits]
  obtain ⟨m, hm, hl⟩ := h
  apply foldl_min_low
  exact List.mem_map.mpr ⟨m, hm, hl⟩

end Smriti
