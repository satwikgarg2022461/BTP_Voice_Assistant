"""
Recipe Navigator
================
Handles all navigation intents for the conversational recipe voice assistant.

Data sources
------------
- data/chunks.csv         : recipe_id, title, chunk_index, start_step, end_step,
                            chunk_text, searchable_text, metadata
- data/food_dictionary.csv: recipe_id, recipe_name, ingredients

Navigation model
----------------
Each **chunk** is a contiguous slice of a recipe's instructions (start_step → end_step).
Inside a chunk the individual cooking instructions are separated by " . " so they map
1-to-1 to *steps*.  The navigator tracks position at both levels:

  chunk_index  – which chunk we are currently in  (1-based, matches CSV)
  step_index   – global step number within the recipe (1-based)

Public interface
----------------
  load_recipe(recipe_id)            → RecipeData or None
  get_current_step(session)         → NavigationResult
  get_next_step(session)            → NavigationResult
  get_previous_step(session)        → NavigationResult
  get_current_ingredients(session)  → NavigationResult
  jump_to_step(session, target)     → NavigationResult
  restart(session)                  → NavigationResult
  get_recipe_summary(session)       → NavigationResult
  build_session_nav_fields(recipe)  → dict   (fields to merge into session at start)
  update_session_from_result(session, result) → dict
"""

import csv
import os
import re
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────── Data containers ────────────────────────────────

@dataclass
class ChunkData:
    """One row from chunks.csv"""
    recipe_id:   int
    title:       str
    chunk_index: int          # 1-based chunk number
    start_step:  int          # first global step in this chunk
    end_step:    int          # last  global step in this chunk
    chunk_text:  str          # full text of all steps in this chunk
    steps:       List[str]    # individual sentences split from chunk_text
    ingredients: List[str]    # ingredients mentioned in THIS chunk (from searchable_text)
    metadata:    Dict


@dataclass
class RecipeData:
    """All data for one recipe, ready for navigation"""
    recipe_id:         int
    title:             str
    chunks:            List[ChunkData]   # ordered by chunk_index
    all_ingredients:   List[str]         # from food_dictionary.csv
    total_steps:       int               # sum of steps across all chunks
    total_chunks:      int


@dataclass
class NavigationResult:
    """Return type for every navigation call"""
    success:      bool
    intent:       str                    # e.g. "nav_next", "nav_repeat_ingredients"
    step_index:   int                    # updated global step (1-based, 0 = ingredients)
    chunk_index:  int                    # updated chunk index (1-based)
    text:         str                    # text to speak / display
    section:      str                    # "ingredients" | "steps" | "done"
    is_last_step: bool    = False
    is_first_step: bool   = False
    message:      str     = ""           # human-readable status note
    extra:        Dict    = field(default_factory=dict)


# ──────────────────────────── Navigator ──────────────────────────────────────

class RecipeNavigator:
    """
    Loads recipe chunks and ingredients from CSV files and handles all
    navigation intents produced by IntentClassifier.

    Usage
    -----
    >>> nav = RecipeNavigator()
    >>> recipe = nav.load_recipe(3699)
    >>> session = nav.build_session_nav_fields(recipe)
    >>> result  = nav.get_next_step(session)
    >>> print(result.text)
    """

    # Sentence boundary used to split chunk_text into individual steps
    STEP_SEPARATOR = re.compile(r'\s*\.\s+')

    def __init__(self,
                 chunks_csv:     str = "data/chunks.csv",
                 food_dict_csv:  str = "data/food_dictionary.csv"):
        """
        Args:
            chunks_csv    : Path to chunks.csv (relative to project root or absolute).
            food_dict_csv : Path to food_dictionary.csv.
        """
        # Resolve paths relative to this file's location
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.chunks_path    = os.path.join(base, chunks_csv)
        self.food_dict_path = os.path.join(base, food_dict_csv)

        # In-memory caches so the CSV is only read once per recipe_id
        self._recipe_cache:     Dict[int, RecipeData]  = {}
        self._food_dict_cache:  Dict[int, List[str]]   = {}

        # Load food dictionary once at init (it's small)
        self._load_food_dictionary()

    # ─────────────────────────── Loading ─────────────────────────────────────

    def _load_food_dictionary(self) -> None:
        """Load food_dictionary.csv into _food_dict_cache."""
        if not os.path.isfile(self.food_dict_path):
            print(f"[Navigator] Warning: food_dictionary.csv not found at {self.food_dict_path}")
            return

        with open(self.food_dict_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rid = int(row['recipe_id'])
                raw = row.get('ingredients', '')
                # CSV stores a comma-separated string; strip trailing period/spaces
                ingredients = [i.strip().rstrip('.') for i in raw.split(',') if i.strip()]
                self._food_dict_cache[rid] = ingredients

    def load_recipe(self, recipe_id: int) -> Optional[RecipeData]:
        """
        Load all chunks for *recipe_id* from chunks.csv.

        Returns
        -------
        RecipeData if found, None otherwise.
        """
        recipe_id = int(recipe_id)

        if recipe_id in self._recipe_cache:
            return self._recipe_cache[recipe_id]

        if not os.path.isfile(self.chunks_path):
            print(f"[Navigator] Error: chunks.csv not found at {self.chunks_path}")
            return None

        chunks: List[ChunkData] = []

        with open(self.chunks_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if int(row['recipe_id']) != recipe_id:
                    continue

                # Split chunk_text into individual steps
                raw_text = row.get('chunk_text', '').strip().strip('"')
                steps = self._split_steps(raw_text)

                # Extract ingredients mentioned in this chunk from searchable_text
                chunk_ingredients = self._parse_chunk_ingredients(
                    row.get('searchable_text', '')
                )

                # Parse metadata safely
                try:
                    meta = ast.literal_eval(row.get('metadata', '{}'))
                except Exception:
                    meta = {}

                chunks.append(ChunkData(
                    recipe_id   = recipe_id,
                    title       = row['title'],
                    chunk_index = int(row['chunk_index']),
                    start_step  = int(row['start_step']),
                    end_step    = int(row['end_step']),
                    chunk_text  = raw_text,
                    steps       = steps,
                    ingredients = chunk_ingredients,
                    metadata    = meta,
                ))

        if not chunks:
            print(f"[Navigator] No chunks found for recipe_id={recipe_id}")
            return None

        # Sort by chunk_index to guarantee order
        chunks.sort(key=lambda c: c.chunk_index)

        # Recompute step boundaries based on actual sentence splits
        # (CSV start_step/end_step may not match sentence count perfectly)
        step_cursor = 1
        for chunk in chunks:
            chunk.start_step = step_cursor
            chunk.end_step   = step_cursor + len(chunk.steps) - 1
            step_cursor      = chunk.end_step + 1

        total_steps = sum(len(c.steps) for c in chunks)
        title       = chunks[0].title

        recipe = RecipeData(
            recipe_id       = recipe_id,
            title           = title,
            chunks          = chunks,
            all_ingredients = self._food_dict_cache.get(recipe_id, []),
            total_steps     = total_steps,
            total_chunks    = len(chunks),
        )

        self._recipe_cache[recipe_id] = recipe
        return recipe

    # ─────────────────────── Session helpers ─────────────────────────────────

    def build_session_nav_fields(self, recipe: RecipeData) -> Dict:
        """
        Return a dict of navigation fields to merge into a new session.

        Compatible with the structure expected by SessionManager.create_session().

        Fields
        ------
        step_index       : 0  (0 = show ingredients first)
        chunk_index      : 1
        total_steps      : int
        total_chunks     : int
        current_section  : "ingredients"
        """
        return {
            "recipe_id":        str(recipe.recipe_id),
            "recipe_title":     recipe.title,
            "step_index":       0,           # 0 → will move to 1 on first nav_next
            "chunk_index":      1,
            "total_steps":      recipe.total_steps,
            "total_chunks":     recipe.total_chunks,
            "current_section":  "ingredients",
            "current_state":    "RECIPE_SELECTED",
        }

    def update_session_from_result(self, session: Dict,
                                   result: NavigationResult) -> Dict:
        """
        Apply step/chunk/section updates from a NavigationResult back into session.

        Returns the mutated session dict (same object, also returned for chaining).
        """
        session["step_index"]      = result.step_index
        session["chunk_index"]     = result.chunk_index
        session["current_section"] = result.section
        session["last_intent"]     = result.intent

        if result.section == "steps":
            session["current_state"] = "READING_STEPS"
        elif result.section == "ingredients":
            session["current_state"] = "READING_INGREDIENTS"
        elif result.section == "done":
            session["current_state"] = "RECIPE_DONE"

        return session

    # ──────────────────────── Core navigation API ────────────────────────────

    def get_current_ingredients(self, session: Dict) -> NavigationResult:
        """
        Return the full ingredient list for the active recipe.

        session keys used: recipe_id
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_repeat_ingredients", "No active recipe found.")

        if recipe.all_ingredients:
            text = self._format_ingredients(recipe.all_ingredients)
        else:
            # Fallback: extract from first chunk's searchable_text
            text = self._format_ingredients(recipe.chunks[0].ingredients)

        return NavigationResult(
            success     = True,
            intent      = "nav_repeat_ingredients",
            step_index  = 0,
            chunk_index = 1,
            text        = text,
            section     = "ingredients",
            is_first_step = True,
            message     = f"Ingredients for {recipe.title}",
        )

    def get_current_step(self, session: Dict) -> NavigationResult:
        """
        Return the step at the current step_index without advancing.

        session keys used: recipe_id, step_index, chunk_index
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_repeat", "No active recipe found.")

        step_index = int(session.get("step_index", 0))

        # step_index == 0 → we're still at the ingredients section
        if step_index == 0:
            return self.get_current_ingredients(session)

        chunk, local_idx = self._locate_step(recipe, step_index)
        if chunk is None:
            return self._error_result("nav_repeat", f"Step {step_index} not found.")

        step_text = chunk.steps[local_idx]
        return NavigationResult(
            success      = True,
            intent       = "nav_repeat",
            step_index   = step_index,
            chunk_index  = chunk.chunk_index,
            text         = self._format_step(step_index, step_text, recipe.total_steps),
            section      = "steps",
            is_last_step  = (step_index == recipe.total_steps),
            is_first_step = (step_index == 1),
            message      = f"Step {step_index} of {recipe.total_steps}",
        )

    def get_next_step(self, session: Dict) -> NavigationResult:
        """
        Advance to the next step and return its text.

        If we are at step_index=0 (ingredients), moves to step 1.
        If we are at the last step, returns a completion message.

        session keys used: recipe_id, step_index
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_next", "No active recipe found.")

        current = int(session.get("step_index", 0))
        next_step = current + 1

        # Still on the ingredients screen → go to step 1
        if current == 0:
            next_step = 1

        if next_step > recipe.total_steps:
            return NavigationResult(
                success      = True,
                intent       = "nav_next",
                step_index   = recipe.total_steps,
                chunk_index  = recipe.total_chunks,
                text         = (
                    f"That's it! You've completed all {recipe.total_steps} steps "
                    f"for {recipe.title}. Enjoy your meal!"
                ),
                section      = "done",
                is_last_step = True,
                message      = "Recipe complete",
            )

        chunk, local_idx = self._locate_step(recipe, next_step)
        step_text = chunk.steps[local_idx]

        return NavigationResult(
            success      = True,
            intent       = "nav_next",
            step_index   = next_step,
            chunk_index  = chunk.chunk_index,
            text         = self._format_step(next_step, step_text, recipe.total_steps),
            section      = "steps",
            is_last_step  = (next_step == recipe.total_steps),
            is_first_step = (next_step == 1),
            message      = f"Step {next_step} of {recipe.total_steps}",
        )

    def get_previous_step(self, session: Dict) -> NavigationResult:
        """
        Go back one step and return its text.

        If at step 1, goes back to the ingredients list.
        If already at ingredients, says we're already at the beginning.

        session keys used: recipe_id, step_index
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_prev", "No active recipe found.")

        current = int(session.get("step_index", 0))
        prev_step = current - 1

        if current == 0:
            return NavigationResult(
                success      = True,
                intent       = "nav_prev",
                step_index   = 0,
                chunk_index  = 1,
                text         = "We're already at the very beginning of the recipe.",
                section      = "ingredients",
                is_first_step= True,
                message      = "Already at beginning",
            )

        # current == 1 → go back to ingredients
        if prev_step <= 0:
            return self.get_current_ingredients(session)

        chunk, local_idx = self._locate_step(recipe, prev_step)
        step_text = chunk.steps[local_idx]

        return NavigationResult(
            success       = True,
            intent        = "nav_prev",
            step_index    = prev_step,
            chunk_index   = chunk.chunk_index,
            text          = self._format_step(prev_step, step_text, recipe.total_steps),
            section       = "steps",
            is_last_step  = (prev_step == recipe.total_steps),
            is_first_step = (prev_step == 1),
            message       = f"Step {prev_step} of {recipe.total_steps}",
        )

    def jump_to_step(self, session: Dict,
                     target: int | str) -> NavigationResult:
        """
        Jump directly to a specific step number or position keyword.

        Args
        ----
        target : int   → jump to that exact step number (1-based)
                 "first"/"start"/"beginning" → step 1
                 "last"/"end"               → last step
                 "ingredients"              → ingredient list

        session keys used: recipe_id
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_go_to", "No active recipe found.")

        # Resolve keyword targets
        if isinstance(target, str):
            t = target.strip().lower()
            if t in ("ingredients", "ingredient"):
                return self.get_current_ingredients(session)
            elif t in ("first", "start", "beginning", "1"):
                target = 1
            elif t in ("last", "end"):
                target = recipe.total_steps
            else:
                # Try to parse a number embedded in the string
                m = re.search(r'\d+', t)
                if m:
                    target = int(m.group())
                else:
                    return self._error_result(
                        "nav_go_to",
                        f"I couldn't understand '{target}' as a step number."
                    )

        target = int(target)

        if target < 1 or target > recipe.total_steps:
            return self._error_result(
                "nav_go_to",
                f"Step {target} doesn't exist. This recipe has {recipe.total_steps} steps."
            )

        chunk, local_idx = self._locate_step(recipe, target)
        step_text = chunk.steps[local_idx]

        return NavigationResult(
            success       = True,
            intent        = "nav_go_to",
            step_index    = target,
            chunk_index   = chunk.chunk_index,
            text          = self._format_step(target, step_text, recipe.total_steps),
            section       = "steps",
            is_last_step  = (target == recipe.total_steps),
            is_first_step = (target == 1),
            message       = f"Jumped to step {target} of {recipe.total_steps}",
        )

    def restart(self, session: Dict) -> NavigationResult:
        """
        Reset navigation to the beginning (ingredients) of the recipe.

        session keys used: recipe_id
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_start", "No active recipe found.")

        # Return ingredients as the first thing heard when restarting
        result = self.get_current_ingredients(session)
        result.intent  = "nav_start"
        result.message = f"Restarted {recipe.title} from the beginning"
        return result

    def get_recipe_summary(self, session: Dict) -> NavigationResult:
        """
        Return a brief spoken summary: title, step count, and ingredients.

        Useful when the user asks "what recipe are we making?" or similar.
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("question", "No active recipe found.")

        ingr_count = len(recipe.all_ingredients)
        text = (
            f"We're making {recipe.title}. "
            f"It has {recipe.total_steps} step{'s' if recipe.total_steps != 1 else ''} "
            f"and {ingr_count} ingredient{'s' if ingr_count != 1 else ''}. "
            f"The ingredients are: {', '.join(recipe.all_ingredients)}."
        )

        current_step = int(session.get("step_index", 0))
        section = "ingredients" if current_step == 0 else "steps"

        return NavigationResult(
            success     = True,
            intent      = "question",
            step_index  = current_step,
            chunk_index = int(session.get("chunk_index", 1)),
            text        = text,
            section     = section,
            message     = "Recipe summary",
        )

    # ──────────────────────── Chunk-level navigation ─────────────────────────

    def get_chunk(self, recipe: RecipeData, chunk_index: int) -> Optional[ChunkData]:
        """Return ChunkData for a given 1-based chunk_index, or None."""
        for chunk in recipe.chunks:
            if chunk.chunk_index == chunk_index:
                return chunk
        return None

    def get_next_chunk(self, session: Dict) -> NavigationResult:
        """
        Jump to the first step of the next chunk.
        Useful for coarse navigation in long recipes.
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_next", "No active recipe found.")

        current_chunk_idx = int(session.get("chunk_index", 1))
        next_chunk_idx    = current_chunk_idx + 1

        if next_chunk_idx > recipe.total_chunks:
            return NavigationResult(
                success      = True,
                intent       = "nav_next",
                step_index   = recipe.total_steps,
                chunk_index  = recipe.total_chunks,
                text         = (
                    f"You've reached the end of {recipe.title}. "
                    f"There are no more sections after this."
                ),
                section      = "done",
                is_last_step = True,
                message      = "No more chunks",
            )

        chunk = self.get_chunk(recipe, next_chunk_idx)
        target_step = chunk.start_step
        session_copy = dict(session)   # don't mutate caller's session here
        session_copy["recipe_id"] = str(recipe.recipe_id)
        return self.jump_to_step(session_copy, target_step)

    def get_prev_chunk(self, session: Dict) -> NavigationResult:
        """
        Jump to the first step of the previous chunk.
        """
        recipe = self._get_recipe_from_session(session)
        if recipe is None:
            return self._error_result("nav_prev", "No active recipe found.")

        current_chunk_idx = int(session.get("chunk_index", 1))
        prev_chunk_idx    = current_chunk_idx - 1

        if prev_chunk_idx < 1:
            return self.get_current_ingredients(session)

        chunk = self.get_chunk(recipe, prev_chunk_idx)
        target_step = chunk.start_step
        session_copy = dict(session)
        session_copy["recipe_id"] = str(recipe.recipe_id)
        return self.jump_to_step(session_copy, target_step)

    # ─────────────────────────── Internals ───────────────────────────────────

    def _get_recipe_from_session(self, session: Dict) -> Optional[RecipeData]:
        """Load (or cache-hit) the recipe specified in session['recipe_id']."""
        recipe_id = session.get("recipe_id")
        if not recipe_id:
            return None
        try:
            return self.load_recipe(int(recipe_id))
        except (ValueError, TypeError):
            return None

    def _locate_step(self, recipe: RecipeData,
                     step_index: int) -> Tuple[Optional[ChunkData], int]:
        """
        Find the ChunkData and local (within-chunk) index for a global step_index.

        Returns
        -------
        (ChunkData, local_index)  – local_index is 0-based inside chunk.steps
        (None, -1)                – if step not found
        """
        for chunk in recipe.chunks:
            if chunk.start_step <= step_index <= chunk.end_step:
                local = step_index - chunk.start_step
                return chunk, local
        return None, -1

    @staticmethod
    def _split_steps(chunk_text: str) -> List[str]:
        """
        Split a chunk's raw text into individual cooking steps.

        The CSV uses " . " as a sentence separator.  We normalise whitespace
        and remove empty fragments.
        """
        # Split on period followed by whitespace (the CSV convention)
        parts = re.split(r'\s*\.\s+', chunk_text.strip())
        # Drop empties; strip residual punctuation
        steps = [p.strip().rstrip('.').strip() for p in parts if p.strip()]
        return steps if steps else [chunk_text.strip()]

    @staticmethod
    def _parse_chunk_ingredients(searchable_text: str) -> List[str]:
        """
        Extract the ingredient list from a chunk's searchable_text field.

        The field contains:
          "Title | ... | ingredients: A qty, B qty, ... | processes: ..."
        """
        m = re.search(r'ingredients:\s*(.+?)\s*\|', searchable_text)
        if not m:
            return []
        raw = m.group(1)
        return [i.strip() for i in raw.split(',') if i.strip()]

    @staticmethod
    def _format_step(step_index: int, step_text: str, total_steps: int) -> str:
        """Format a single cooking step for TTS output."""
        return f"Step {step_index} of {total_steps}: {step_text}."

    @staticmethod
    def _format_ingredients(ingredients: List[str]) -> str:
        """Format the ingredients list as a natural-language sentence for TTS."""
        if not ingredients:
            return "I couldn't find the ingredients for this recipe."
        if len(ingredients) == 1:
            return f"You will need: {ingredients[0]}."
        joined = ", ".join(ingredients[:-1]) + f", and {ingredients[-1]}"
        return f"Here are the ingredients you will need: {joined}."

    @staticmethod
    def _error_result(intent: str, message: str) -> NavigationResult:
        return NavigationResult(
            success     = False,
            intent      = intent,
            step_index  = -1,
            chunk_index = -1,
            text        = message,
            section     = "error",
            message     = message,
        )


# ──────────────────────────────── Quick test ─────────────────────────────────

def _test_navigator():
    import json

    print("=" * 62)
    print("RECIPE NAVIGATOR TEST")
    print("=" * 62)

    nav = RecipeNavigator()

    # ── pick the first available recipe_id from the CSV ──────────────────────
    chunks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "chunks.csv"
    )
    first_id = None
    with open(chunks_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            first_id = int(row['recipe_id'])
            break

    if first_id is None:
        print("✗ No rows in chunks.csv – aborting test.")
        return

    print(f"\nTesting with recipe_id = {first_id}")
    recipe = nav.load_recipe(first_id)
    if recipe is None:
        print(f"✗ Could not load recipe {first_id}")
        return

    print(f"✓ Loaded: '{recipe.title}'")
    print(f"  Chunks: {recipe.total_chunks}  |  Total steps: {recipe.total_steps}")
    print(f"  Ingredients ({len(recipe.all_ingredients)}): "
          f"{', '.join(recipe.all_ingredients[:4])}{'...' if len(recipe.all_ingredients) > 4 else ''}")

    session = nav.build_session_nav_fields(recipe)
    print(f"\nInitial session nav fields:\n  {json.dumps({k: session[k] for k in ['step_index','chunk_index','total_steps','current_section']}, indent=2)}")

    # ── simulate a full navigation sequence ──────────────────────────────────
    steps_to_test = [
        ("get_current_ingredients", lambda s: nav.get_current_ingredients(s)),
        ("get_next_step (→ step 1)", lambda s: nav.get_next_step(s)),
        ("get_next_step (→ step 2)", lambda s: nav.get_next_step(s)),
        ("get_current_step (repeat)", lambda s: nav.get_current_step(s)),
        ("get_previous_step (→ step 1)", lambda s: nav.get_previous_step(s)),
        ("jump_to_step (last)",  lambda s: nav.jump_to_step(s, "last")),
        ("get_next_step (past end)", lambda s: nav.get_next_step(s)),
        ("restart",              lambda s: nav.restart(s)),
        ("get_recipe_summary",   lambda s: nav.get_recipe_summary(s)),
    ]

    for label, fn in steps_to_test:
        result = fn(session)
        nav.update_session_from_result(session, result)
        status = "✓" if result.success else "✗"
        print(f"\n{status} [{label}]")
        print(f"  intent={result.intent}  step={result.step_index}  "
              f"chunk={result.chunk_index}  section={result.section}")
        # Truncate long text for readability
        preview = result.text[:120] + ("…" if len(result.text) > 120 else "")
        print(f"  text: {preview}")
        if result.message:
            print(f"  msg : {result.message}")

    # ── edge cases ────────────────────────────────────────────────────────────
    print("\n── Edge cases ──────────────────────────────────────────")
    session["step_index"] = 0
    r = nav.get_previous_step(session)
    print(f"prev from ingredients: '{r.text}'  success={r.success}")

    r = nav.jump_to_step(session, 999)
    print(f"jump to 999: success={r.success}  text='{r.text}'")

    r = nav.jump_to_step(session, "ingredients")
    print(f"jump to 'ingredients': section={r.section}  success={r.success}")

    print("\n✓ Navigator tests complete")


if __name__ == "__main__":
    _test_navigator()

