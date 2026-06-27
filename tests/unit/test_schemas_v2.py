import pytest
import numpy as np
from ravana.core.vsa import VSAManager
from ravana.language.schemas import SchemaLibrary

def test_schema_library_init():
    mgr = VSAManager(dim=128)
    lib = SchemaLibrary(mgr)
    
    assert len(lib.schemas) > 0
    # S-V-O should be in the default schemas
    schema_names = [s.template_name for s in lib.schemas]
    assert "S-V-O" in schema_names

def test_schema_selection():
    mgr = VSAManager(dim=128)
    lib = SchemaLibrary(mgr)
    
    # Selecting schema with subject, verb, object should return S-V-O or similar
    schema = lib.select_schema(["subject", "verb", "object"])
    assert "subject" in schema.structure
    assert "verb" in schema.structure
    assert "object" in schema.structure

def test_realize_sentence():
    mgr = VSAManager(dim=128)
    lib = SchemaLibrary(mgr)
    
    schema = lib.select_schema(["subject", "verb", "object"])
    
    # Vocab/embeddings dictionary
    embeddings = {
        "dog": mgr.generate_vector(),
        "barks": mgr.generate_vector(),
        "loudly": mgr.generate_vector()
    }
    
    fillers = {
        "subject": "dog",
        "verb": "barks",
        "object": "loudly"
    }
    
    sentence = lib.realize_sentence(schema, fillers, embeddings)
    assert isinstance(sentence, str)
    assert len(sentence.split()) > 0
    # Words in the sentence should be mapped to the placeholders
    assert "dog" in sentence
    assert "barks" in sentence
