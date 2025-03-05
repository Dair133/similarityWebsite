import json
import os

class PromptManager:
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        self.prompts = {}
        self._load_prompts()
    
    def _load_prompts(self):
        try:
            if not os.path.exists(self.json_filepath):
                print(f"Warning: File {self.json_filepath} not found.")
                return
            
            with open(self.json_filepath, 'r', encoding='utf-8') as file:
                prompt_data = json.load(file)
                
                # Process each prompt in the loaded data
                for key, lines in prompt_data.items():
                    if isinstance(lines, list):
                        # Join the list of strings with newlines to preserve formatting
                        self.prompts[key] = '\n'.join(lines)
                    else:
                        # If it's already a string, use it directly
                        self.prompts[key] = lines
                        
        except json.JSONDecodeError:
            print(f"Error: File {self.json_filepath} contains invalid JSON.")
        except Exception as e:
            print(f"An error occurred while loading prompts: {str(e)}")
    
    def get_prompt(self, prompt_name):
        if prompt_name not in self.prompts:
            print(f"Warning: Prompt '{prompt_name}' not found.")
            return None
        
        return self.prompts[prompt_name]
    
    def list_available_prompts(self):
        return list(self.prompts.keys())
    
    def reload_prompts(self):
        """Reload prompts from the JSON file."""
        self.prompts = {}
        self._load_prompts()