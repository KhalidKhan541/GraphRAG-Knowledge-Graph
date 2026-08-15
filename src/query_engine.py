import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .entity_extractor import Entity, Relation, ExtractionResult
from .community_detection import CommunityDetector

@dataclass
class QueryResult:
    """Result of a graph query."""
    answer: str
    sources: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    communities: List[int] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class GraphQueryEngine:
    """Multi-hop retrieval and query engine over knowledge graph."""
    
    def __init__(self, extraction: ExtractionResult = None):
        self.extraction = extraction or ExtractionResult(entities=[], relations=[], source_text="")
        self.entity_map = {}
        self.adjacency = {}
        self.community_detector = CommunityDetector()
        self.logger = logging.getLogger(__name__)
        
        if extraction:
            self._build_index(extraction)
    
    def _build_index(self, extraction: ExtractionResult):
        """Build index from extraction result."""
        # Build entity map
        for entity in extraction.entities:
            self.entity_map[entity.name.lower()] = entity
        
        # Build adjacency list
        self.adjacency = {e.name.lower(): [] for e in extraction.entities}
        
        for relation in extraction.relations:
            src = relation.source.lower()
            tgt = relation.target.lower()
            
            if src in self.adjacency:
                self.adjacency[src].append((tgt, relation))
            if tgt in self.adjacency:
                self.adjacency[tgt].append((src, relation))
        
        # Load community detector
        self.community_detector.load_from_extraction(extraction)
        self.community_detector.detect_louvain()
    
    def query(self, nl_query: str) -> QueryResult:
        """Execute a natural language query."""
        query_lower = nl_query.lower()
        
        # Route to appropriate query type
        if 'multi-hop' in query_lower or 'path' in query_lower:
            return self._multi_hop_query(nl_query)
        elif 'community' in query_lower or 'cluster' in query_lower:
            return self._community_query(nl_query)
        elif 'related' in query_lower or 'connected' in query_lower:
            return self._related_entities_query(nl_query)
        elif 'summary' in query_lower or 'summarize' in query_lower:
            return self._summary_query(nl_query)
        else:
            return self._generic_query(nl_query)
    
    def _multi_hop_query(self, nl_query: str, max_hops: int = 3) -> QueryResult:
        """Multi-hop reasoning query."""
        # Extract source and target from query
        entities_mentioned = self._extract_mentioned_entities(nl_query)
        
        if len(entities_mentioned) < 2:
            return QueryResult(
                answer="Please specify source and target entities for multi-hop query.",
                confidence=0.3,
            )
        
        source = entities_mentioned[0]
        target = entities_mentioned[1]
        
        # Find paths
        paths = self._find_paths(source, target, max_hops)
        
        if not paths:
            return QueryResult(
                answer=f"No path found between {source} and {target} within {max_hops} hops.",
                confidence=0.2,
            )
        
        # Build reasoning path
        reasoning = []
        for path in paths:
            hop_str = " -> ".join([f"{node}" for node in path])
            reasoning.append(hop_str)
        
        return QueryResult(
            answer=f"Found {len(paths)} path(s) between {source} and {target}.",
            sources=[source, target],
            reasoning_path=reasoning,
            confidence=min(0.9, 0.5 + len(paths) * 0.1),
            metadata={'paths': [list(p) for p in paths]},
        )
    
    def _community_query(self, nl_query: str) -> QueryResult:
        """Community-based query."""
        communities = self.community_detector.communities
        
        if not communities:
            return QueryResult(
                answer="No communities detected in the graph.",
                confidence=0.2,
            )
        
        # Find community containing mentioned entity
        mentioned = self._extract_mentioned_entities(nl_query)
        matching_communities = []
        
        for entity_name in mentioned:
            for comm_id, comm in communities.items():
                if entity_name.lower() in [m.lower() for m in comm.members]:
                    matching_communities.append(comm)
        
        if matching_communities:
            comm = matching_communities[0]
            return QueryResult(
                answer=f"Community {comm.id}: {comm.summary}",
                sources=comm.members,
                communities=[comm.id],
                confidence=0.8,
            )
        
        return QueryResult(
            answer=f"Detected {len(communities)} communities in the graph.",
            communities=list(communities.keys()),
            confidence=0.5,
        )
    
    def _related_entities_query(self, nl_query: str) -> QueryResult:
        """Find related entities."""
        entities_mentioned = self._extract_mentioned_entities(nl_query)
        
        if not entities_mentioned:
            return QueryResult(
                answer="No entities mentioned in query.",
                confidence=0.3,
            )
        
        entity = entities_mentioned[0]
        neighbors = self.adjacency.get(entity.lower(), [])
        
        if not neighbors:
            return QueryResult(
                answer=f"No related entities found for {entity}.",
                confidence=0.3,
            )
        
        related = [f"{n[0]} ({n[1].relation_type})" for n in neighbors[:10]]
        
        return QueryResult(
            answer=f"Entities related to {entity}: {', '.join(related)}",
            sources=[entity] + [n[0] for n in neighbors],
            confidence=0.7,
        )
    
    def _summary_query(self, nl_query: str) -> QueryResult:
        """Generate summary of graph."""
        stats = {
            'entities': len(self.extraction.entities),
            'relations': len(self.extraction.relations),
            'communities': len(self.community_detector.communities),
        }
        
        # Get top entities by degree
        degree_count = {}
        for src, targets in self.adjacency.items():
            degree_count[src] = len(targets)
        
        top_entities = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join([f"{e[0]} ({e[1]} connections)" for e in top_entities])
        
        return QueryResult(
            answer=f"Graph summary: {stats['entities']} entities, {stats['relations']} relations, "
                   f"{stats['communities']} communities. Top entities: {top_str}",
            confidence=0.6,
        )
    
    def _generic_query(self, nl_query: str) -> QueryResult:
        """Generic query handler."""
        mentioned = self._extract_mentioned_entities(nl_query)
        
        if mentioned:
            return self._related_entities_query(nl_query)
        
        return QueryResult(
            answer="I can help you find entities, relations, communities, or paths in the knowledge graph.",
            confidence=0.5,
        )
    
    def _extract_mentioned_entities(self, query: str) -> List[str]:
        """Extract entity names mentioned in query."""
        query_lower = query.lower()
        mentioned = []
        
        for entity_name in self.entity_map.keys():
            if entity_name in query_lower:
                mentioned.append(self.entity_map[entity_name].name)
        
        return mentioned
    
    def _find_paths(self, source: str, target: str, max_hops: int) -> List[List[str]]:
        """Find paths between two entities."""
        src_key = source.lower()
        tgt_key = target.lower()
        
        if src_key not in self.adjacency or tgt_key not in self.adjacency:
            return []
        
        # BFS for shortest paths
        paths = []
        queue = [(src_key, [src_key])]
        visited = {src_key}
        
        while queue and len(paths) < 5:
            current, path = queue.pop(0)
            
            if current == tgt_key:
                paths.append(path)
                continue
            
            if len(path) > max_hops:
                continue
            
            for neighbor, _ in self.adjacency.get(current, []):
                if neighbor not in visited or neighbor == tgt_key:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get full context for an entity."""
        entity_key = entity_name.lower()
        
        if entity_key not in self.entity_map:
            return {'error': f'Entity {entity_name} not found'}
        
        entity = self.entity_map[entity_key]
        neighbors = self.adjacency.get(entity_key, [])
        
        # Find community
        community_id = None
        for comm_id, comm in self.community_detector.communities.items():
            if entity_name.lower() in [m.lower() for m in comm.members]:
                community_id = comm_id
                break
        
        return {
            'entity': entity.name,
            'type': entity.entity_type,
            'properties': entity.properties,
            'connections': [(n[0], n[1].relation_type) for n in neighbors],
            'community_id': community_id,
        }
