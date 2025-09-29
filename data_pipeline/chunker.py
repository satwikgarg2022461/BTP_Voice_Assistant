import csv
import os
import re
from typing import List, Dict


class RecipeChunker:
    def __init__(self, min_words: int = 150, max_words: int = 220, overlap: int = 30, output_dir: str = "data"):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap = overlap
        self.output_dir = output_dir

        # Create output folder if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\w+", text))

    def chunk_instructions(self, recipe: Dict) -> List[Dict]:
        steps = recipe.get("instructions", [])
        if not steps:
            return []

        recipe_id = recipe["recipe_id"]
        title = recipe["title"]
        metadata = recipe["metadata"]

        ingredients_list = [
            f"{i['ingredient']} {i['quantity'] or ''} {i['unit'] or ''}".strip()
            for i in recipe["ingredients"]
        ]
        clean_ingredient_list = ", ".join(ingredients_list)

        nutritions = metadata.get("nutritions") or {}
        calories = nutritions.get("Energy (kcal)", "NA")
        protein = nutritions.get("Protein (g)", "NA")

        chunks = []
        current_chunk = []
        current_words = 0
        chunk_index = 1
        start_step = 1

        for i, step in enumerate(steps, start=1):
            step_words = self._word_count(step)
            if current_words + step_words > self.max_words and current_words >= self.min_words:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "recipe_id": recipe_id,
                    "title": title,
                    "chunk_index": chunk_index,
                    "start_step": start_step,
                    "end_step": i - 1,
                    "chunk_text": chunk_text,
                    "searchable_text": (
                        f"{title} | source: {metadata.get('source')} "
                        f"| region: {metadata.get('region')}/{metadata.get('sub_region')} "
                        f"| ingredients: {clean_ingredient_list} "
                        f"| processes: {', '.join(recipe.get('process_tags', []))} "
                        f"| instructions: {chunk_text} "
                        f"| nutrition: calories {calories} ; protein {protein}"
                    ),
                    "metadata": metadata
                })

                # Overlap
                overlap_words = " ".join(current_chunk).split()[-self.overlap:]
                current_chunk = [" ".join(overlap_words), step]
                current_words = self._word_count(" ".join(current_chunk))
                chunk_index += 1
                start_step = i
            else:
                current_chunk.append(step)
                current_words += step_words

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "recipe_id": recipe_id,
                "title": title,
                "chunk_index": chunk_index,
                "start_step": start_step,
                "end_step": len(steps),
                "chunk_text": chunk_text,
                "searchable_text": (
                    f"{title} | source: {metadata.get('source')} "
                    f"| region: {metadata.get('region')}/{metadata.get('sub_region')} "
                    f"| ingredients: {clean_ingredient_list} "
                    f"| processes: {', '.join(recipe.get('process_tags', []))} "
                    f"| instructions: {chunk_text} "
                    f"| nutrition: calories {calories} ; protein {protein}"
                ),
                "metadata": metadata
            })

        return chunks

    def export_recipes_csv(self, recipes: List[Dict], filename: str = "recipes.csv", append: bool = False):
        """
        Export recipes to CSV file
        
        Args:
            recipes: List of recipe dictionaries
            filename: Output filename
            append: If True, append to existing file. If False, create new file.
        """
        path = os.path.join(self.output_dir, filename)
        mode = "a" if append else "w"
        
        with open(path, mode=mode, newline="", encoding="utf-8") as f:
            fieldnames = ["recipe_id", "title", "region", "sub_region", "source", "url"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Only write header if creating new file or file is empty
            if not append or os.path.getsize(path) == 0:
                writer.writeheader()
                
            for r in recipes:
                m = r["metadata"]
                writer.writerow({
                    "recipe_id": r["recipe_id"],
                    "title": r["title"],
                    "region": m.get("region"),
                    "sub_region": m.get("sub_region"),
                    "source": m.get("source"),
                    "url": m.get("url")
                })

    def export_chunks_csv(self, chunks: List[Dict], filename: str = "chunks.csv", append: bool = False):
        """
        Export chunks to CSV file
        
        Args:
            chunks: List of chunk dictionaries
            filename: Output filename
            append: If True, append to existing file. If False, create new file.
        """
        path = os.path.join(self.output_dir, filename)
        mode = "a" if append else "w"
        
        with open(path, mode=mode, newline="", encoding="utf-8") as f:
            fieldnames = [
                "recipe_id", "title", "chunk_index", "start_step",
                "end_step", "chunk_text", "searchable_text", "metadata"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Only write header if creating new file or file is empty
            if not append or os.path.getsize(path) == 0:
                writer.writeheader()
                
            for c in chunks:
                writer.writerow(c)
