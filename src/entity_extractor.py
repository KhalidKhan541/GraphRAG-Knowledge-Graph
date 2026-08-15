import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class Entity:
    """Extracted entity."""
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    mentions: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.name.lower(), self.entity_type))
    
    def __eq__(self, other):
        return self.name.lower() == other.name.lower() and self.entity_type == other.entity_type

@dataclass
class Relation:
    """Extracted relation between entities."""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: str = ""
    
    def __hash__(self):
        return hash((self.source.lower(), self.target.lower(), self.relation_type))

@dataclass
class ExtractionResult:
    """Result of entity and relation extraction."""
    entities: List[Entity]
    relations: List[Relation]
    source_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityExtractor:
    """LLM-driven entity and relation extraction."""
    
    ENTITY_TYPES = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'EVENT',
                   'PRODUCT', 'TECHNOLOGY', 'CONCEPT', 'METRIC']
    
    RELATION_TYPES = ['WORKS_AT', 'LOCATED_IN', 'FOUNDED', 'ACQUIRED',
                     'PARTNERED_WITH', 'DEVELOPED', 'MEASURED_BY',
                     'RELATED_TO', 'CAUSES', 'USES']
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.logger = logging.getLogger(__name__)
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations from text."""
        if self.use_llm:
            return self._llm_extraction(text)
        else:
            return self._regex_extraction(text)
    
    def _llm_extraction(self, text: str) -> ExtractionResult:
        """Extract using LLM (placeholder for actual API call)."""
        # This would call an actual LLM API in production
        entities = self._regex_extraction(text).entities
        relations = self._regex_extraction(text).relations
        
        # Simulate LLM-enhanced extraction
        llm_entities = self._simulate_llm_entities(text)
        llm_relations = self._simulate_llm_relations(text, llm_entities)
        
        return ExtractionResult(
            entities=entities + llm_entities,
            relations=relations + llm_relations,
            source_text=text,
            metadata={'method': 'llm', 'model': 'placeholder'},
        )
    
    def _simulate_llm_entities(self, text: str) -> List[Entity]:
        """Simulate LLM entity extraction."""
        entities = []
        
        # Simple pattern matching for demonstration
        # Capitalized words that might be entities
        cap_pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
        matches = re.findall(cap_pattern, text)
        
        for match in matches:
            if len(match) > 2 and match.lower() not in ['the', 'and', 'for', 'with']:
                entity_type = self._classify_entity(match, text)
                entities.append(Entity(
                    name=match,
                    entity_type=entity_type,
                    properties={},
                ))
        
        return list(set(entities))
    
    def _simulate_llm_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Simulate LLM relation extraction."""
        relations = []
        
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Check if entities co-occur in a sentence
                for sentence in text.split('.'):
                    if e1.name in sentence and e2.name in sentence:
                        rel_type = self._infer_relation(e1, e2, sentence)
                        relations.append(Relation(
                            source=e1.name,
                            target=e2.name,
                            relation_type=rel_type,
                            confidence=0.7,
                            evidence=sentence.strip(),
                        ))
        
        return relations
    
    def _regex_extraction(self, text: str) -> ExtractionResult:
        """Extract using regex patterns."""
        entities = []
        relations = []
        
        # Entity patterns
        patterns = {
            'PERSON': r'(?:Mr|Mrs|Ms|Dr|Prof)\.\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            'ORGANIZATION': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co))\b',
            'DATE': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b',
            'LOCATION': r'\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        }
        
        for etype, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    name=match.group(1) if match.lastindex else match.group(0),
                    entity_type=etype,
                    properties={},
                ))
        
        # Relation patterns
        rel_patterns = [
            (r'([A-Z][a-z]+)\s+(?:works?\s+(?:at|for))\s+([A-Z][a-z]+)', 'WORKS_AT'),
            (r'([A-Z][a-z]+)\s+(?:is\s+(?:located|based)\s+in)\s+([A-Z][a-z]+)', 'LOCATED_IN'),
            (r'([A-Z][a-z]+)\s+(?:founded|created|started)\s+([A-Z][a-z]+)', 'FOUNDED'),
        ]
        
        for pattern, rel_type in rel_patterns:
            for match in re.finditer(pattern, text):
                relations.append(Relation(
                    source=match.group(1),
                    target=match.group(2),
                    relation_type=rel_type,
                    confidence=0.8,
                ))
        
        return ExtractionResult(
            entities=list(set(entities)),
            relations=list(set(relations)),
            source_text=text,
            metadata={'method': 'regex'},
        )
    
    def _classify_entity(self, name: str, context: str) -> str:
        """Classify entity type based on context."""
        context_lower = context.lower()
        
        if any(w in context_lower for w in ['company', 'inc', 'corp', 'organization']):
            return 'ORGANIZATION'
        elif any(w in context_lower for w in ['city', 'country', 'state', 'located']):
            return 'LOCATION'
        elif any(w in context_lower for w in ['mr', 'mrs', 'dr', 'professor']):
            return 'PERSON'
        
        return 'CONCEPT'
    
    def _infer_relation(self, e1: Entity, e2: Entity, context: str) -> str:
        """Infer relation type from context."""
        context_lower = context.lower()
        
        if 'works' in context_lower or 'employed' in context_lower:
            return 'WORKS_AT'
        elif 'located' in context_lower or 'based' in context_lower:
            return 'LOCATED_IN'
        elif 'founded' in context_lower or 'created' in context_lower:
            return 'FOUNDED'
        elif 'developed' in context_lower or 'built' in context_lower:
            return 'DEVELOPED'
        
        return 'RELATED_TO'
    
    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract from multiple texts."""
        return [self.extract(text) for text in texts]
    
    def merge_extractions(self, results: List[ExtractionResult]) -> ExtractionResult:
        """Merge multiple extraction results."""
        all_entities = []
        all_relations = []
        
        for result in results:
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
        
        # Deduplicate
        unique_entities = list({e: e for e in all_entities}.values())
        unique_relations = list({r: r for r in all_relations}.values())
        
        return ExtractionResult(
            entities=unique_entities,
            relations=unique_relations,
            source_text="",
            metadata={'merged_from': len(results)},
        )
    
    def get_extraction_prompt(self, text: str) -> str:
        """Generate LLM prompt for extraction."""
        return f"""Extract all entities and their relations from the following text.

Text: {text}

For each entity, provide:
- name: the entity name
- type: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT, TECHNOLOGY, CONCEPT, or METRIC
- properties: any additional attributes

For each relation, provide:
- source: source entity name
- target: target entity name  
- type: relation type (e.g., WORKS_AT, LOCATED_IN, FOUNDED, etc.)
- confidence: 0-1 confidence score

Return as JSON:
{{
    "entities": [...],
    "relations": [...]
}}
"""
