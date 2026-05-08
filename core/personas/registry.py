from typing import List
from base import Persona

class PersonaRegistry :
    def __init__(self, builtin_path : str , db_session = None):

        self._personas : dict[str, Persona ] = {}
        self._load_builtins(builtin_path)
        if db_session:
            self._load_persona_from_db(db_session)
    
    @property
    def _personas (self) :
        return self._personas
    
    def get (self, persona_name : str) -> Persona :
        """
        Returns a persona with the given persona_name. Falls back to 'generic' if the persona doesn't exist in the registry
        """        
        return self._personas.get(key = persona_name, default = self._personas["generic"])

    def list_all (self) -> List[Persona] :
        return list(self._personas.values())
    
    def register (self, definition : Persona) -> None:
        self._personas[definition.persona] = definition

    def _load_builtins(self, builtin_path : str) -> None :
        pass
