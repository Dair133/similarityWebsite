import json
import hashlib
import os
from typing import Dict, List

class SearchTermCache:
    def __init__(self, cache_dir: str = "search_term_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_cache_filename(self, pdf_title: str) -> str:
        # Create a hash of the title for the filename
        title_hash = hashlib.md5(pdf_title.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{title_hash}.json")

    def cache_exists(self, pdf_title: str) -> bool:
        return os.path.exists(self.get_cache_filename(pdf_title))

    def save_cache(self, pdf_title: str, search_terms: List[Dict]) -> None:
        cache_data = {
            "pdf_title": pdf_title,
            "search_terms": search_terms
        }
        with open(self.get_cache_filename(pdf_title), 'w') as f:
            json.dump(cache_data, f)

    def get_cache(self, pdf_title: str) -> List[Dict]:
        try:
            with open(self.get_cache_filename(pdf_title), 'r') as f:
                return json.load(f)["search_terms"]
        except Exception as e:
            print(f"Error reading cache: {e}")
            return None
