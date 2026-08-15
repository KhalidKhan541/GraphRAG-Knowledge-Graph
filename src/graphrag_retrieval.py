import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

from .entity_extractor import ExtractionResult
from .community_detection import CommunityDetector
from .query_engine import GraphQueryEngine, QueryResult

@dataclass
class RetrievalResult:
    """Result of GraphRAG retrieval."""
    answer: str
    context: str
    source_nodes: List[str]
    reasoning_path: List[str]
    retrieval_method: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class GraphRAGRetriever:
    """Graph-based Retrieval Augmented Generation."""
    
    def __init__(self, extraction: ExtractionResult):
        self.extraction = extraction
        self.query_engine = GraphQueryEngine(extraction)
        self.community_detector = CommunityDetector()
        self.logger = logging.getLogger(__name__)
        
        self.community_detector.load_from_extraction(extraction)
        self.community_detector.detect_louvain()
    
    def retrieve(self, query: str, method: str = "multi_hop", top_k: int = 5) -> RetrievalResult:
        """Retrieve context using GraphRAG."""
        if method == "multi_hop":
            return self._multi_hop_retrieval(query, top_k)
        elif method == "community":
            return self._community_retrieval(query, top_k)
        elif method == "entity_centric":
            return self._entity_centric_retrieval(query, top_k)
        else:
            return self._hybrid_retrieval(query, top_k)
    
    def _multi_hop_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Multi-hop retrieval."""
        query_result = self.query_engine.query(query)
        
        # Collect context from paths
        context_parts = []
        source_nodes = set()
        
        for path_str in query_result.reasoning_path:
            nodes = path_str.split(" -> ")
            source_nodes.update(nodes)
            
            for i in range(len(nodes) - 1):
                src, tgt = nodes[i], nodes[i + 1]
                # Find relation evidence
                for rel in self.extraction.relations:
                    if (rel.source.lower() == src.lower() and rel.target.lower() == tgt.lower()) or \
                       (rel.source.lower() == tgt.lower() and rel.target.lower() == src.lower()):
                        if rel.evidence:
                            context_parts.append(rel.evidence)
        
        context = "\n\n".join(context_parts) if context_parts else query_result.answer
        
        return RetrievalResult(
            answer=query_result.answer,
            context=context,
            source_nodes=list(source_nodes),
            reasoning_path=query_result.reasoning_path,
            retrieval_method="multi_hop",
            confidence=query_result.confidence,
        )
    
    def _community_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Community-based retrieval."""
        mentioned = self.query_engine._extract_mentioned_entities(query)
        
        relevant_communities = []
        for entity in mentioned:
            for comm_id, comm in self.community_detector.communities.items():
                if entity.lower() in [m.lower() for m in comm.members]:
                    relevant_communities.append(comm)
        
        if not relevant_communities:
            relevant_communities = list(self.community_detector.communities.values())[:1]
        
        context_parts = []
        source_nodes = []
        
        for comm in relevant_communities[:top_k]:
            context_parts.append(f"Community {comm.id}: {comm.summary}")
            source_nodes.extend(comm.members[:5])
        
        return RetrievalResult(
            answer=f"Found {len(relevant_communities)} relevant communities",
            context="\n\n".join(context_parts),
            source_nodes=source_nodes[:top_k],
            reasoning_path=[],
            retrieval_method="community",
            confidence=0.6,
        )
    
    def _entity_centric_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Entity-centric retrieval."""
        mentioned = self.query_engine._extract_mentioned_entities(query)
        
        context_parts = []
        source_nodes = []
        
        for entity in mentioned[:top_k]:
            context = self.query_engine.get_entity_context(entity)
            if 'error' not in context:
                context_parts.append(
                    f"{context['entity']} ({context['type']}): "
                    f"Connected to {len(context['connections'])} entities"
                )
                source_nodes.append(entity)
        
        return RetrievalResult(
            answer=f"Retrieved context for {len(source_nodes)} entities",
            context="\n\n".join(context_parts),
            source_nodes=source_nodes,
            reasoning_path=[],
            retrieval_method="entity_centric",
            confidence=0.7,
        )
    
    def _hybrid_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Hybrid retrieval combining methods."""
        multi_hop = self._multi_hop_retrieval(query, top_k // 2)
        community = self._community_retrieval(query, top_k // 2)
        
        all_context = [multi_hop.context, community.context]
        all_nodes = list(set(multi_hop.source_nodes + community.source_nodes))
        
        return RetrievalResult(
            answer=f"{multi_hop.answer}\n{community.answer}",
            context="\n\n".join(all_context),
            source_nodes=all_nodes[:top_k],
            reasoning_path=multi_hop.reasoning_path,
            retrieval_method="hybrid",
            confidence=(multi_hop.confidence + community.confidence) / 2,
        )
    
    def get_global_summary(self) -> str:
        """Generate global summary of all communities."""
        summaries = []
        
        for comm_id, comm in self.community_detector.communities.items():
            summaries.append(f"Community {comm_id} ({comm.size} nodes): {comm.summary}")
        
        return "\n".join(summaries)
