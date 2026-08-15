import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging

@dataclass
class Community:
    """Community detected in the graph."""
    id: int
    members: List[str]
    size: int
    modularity: float = 0.0
    central_nodes: List[str] = field(default_factory=list)
    summary: str = ""

class CommunityDetector:
    """Community detection using Louvain and other algorithms."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.logger = logging.getLogger(__name__)
        self.communities: Dict[int, Community] = {}
    
    def load_from_edges(self, edges: List[Tuple[str, str, Dict[str, Any]]]):
        """Load graph from edges."""
        self.graph.clear()
        for source, target, attrs in edges:
            self.graph.add_edge(source, target, **attrs)
        self.logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes, "
                        f"{self.graph.number_of_edges()} edges")
    
    def load_from_extraction(self, extraction) -> None:
        """Load from ExtractionResult."""
        edges = []
        for entity in extraction.entities:
            self.graph.add_node(entity.name, entity_type=entity.entity_type, **entity.properties)
        for relation in extraction.relations:
            edges.append((relation.source, relation.target, {
                'relation_type': relation.relation_type,
                'confidence': relation.confidence,
            }))
        self.load_from_edges(edges)
    
    def detect_louvain(self, resolution: float = 1.0) -> Dict[int, Community]:
        """Detect communities using Louvain algorithm."""
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(self.graph, resolution=resolution)
        except ImportError:
            # Fallback: simple connected components
            partition = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    partition[node] = i
        
        # Build communities
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        self.communities = {}
        for comm_id, members in communities.items():
            central = self._find_central_nodes(members)
            self.communities[comm_id] = Community(
                id=comm_id,
                members=members,
                size=len(members),
                central_nodes=central,
                summary=self._generate_summary(members),
            )
        
        # Calculate modularity
        if len(self.communities) > 1:
            try:
                modularity = nx.algorithms.community.modularity(
                    self.graph,
                    [set(c.members) for c in self.communities.values()]
                )
                for comm in self.communities.values():
                    comm.modularity = modularity
            except Exception:
                pass
        
        self.logger.info(f"Detected {len(self.communities)} communities")
        return self.communities
    
    def detect_label_propagation(self) -> Dict[int, Community]:
        """Detect communities using label propagation."""
        try:
            communities_iter = nx.algorithms.community.label_propagation_communities(self.graph)
            communities_list = list(communities_iter)
        except Exception:
            return self.detect_louvain()
        
        self.communities = {}
        for i, members in enumerate(communities_list):
            central = self._find_central_nodes(list(members))
            self.communities[i] = Community(
                id=i,
                members=list(members),
                size=len(members),
                central_nodes=central,
                summary=self._generate_summary(list(members)),
            )
        
        return self.communities
    
    def detect_greedy_modularity(self) -> Dict[int, Community]:
        """Detect communities using greedy modularity maximization."""
        try:
            communities_iter = nx.algorithms.community.greedy_modularity_communities(self.graph)
            communities_list = list(communities_iter)
        except Exception:
            return self.detect_louvain()
        
        self.communities = {}
        for i, members in enumerate(communities_list):
            central = self._find_central_nodes(list(members))
            self.communities[i] = Community(
                id=i,
                members=list(members),
                size=len(members),
                central_nodes=central,
                summary=self._generate_summary(list(members)),
            )
        
        return self.communities
    
    def _find_central_nodes(self, members: List[str], top_k: int = 3) -> List[str]:
        """Find most central nodes in a community."""
        if not members:
            return []
        
        subgraph = self.graph.subgraph(members)
        
        try:
            centrality = nx.degree_centrality(subgraph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:top_k]]
        except Exception:
            return members[:top_k]
    
    def _generate_summary(self, members: List[str]) -> str:
        """Generate summary for a community."""
        # Get entity types
        types = defaultdict(int)
        for member in members:
            if member in self.graph.nodes:
                node_data = self.graph.nodes[member]
                etype = node_data.get('entity_type', 'Unknown')
                types[etype] += 1
        
        type_str = ', '.join([f"{t}: {c}" for t, c in types.items()])
        
        return f"Community of {len(members)} members. Types: {type_str}"
    
    def get_community_graph(self, community_id: int) -> nx.Graph:
        """Get subgraph of a community."""
        if community_id not in self.communities:
            return nx.Graph()
        
        members = self.communities[community_id].members
        return self.graph.subgraph(members).copy()
    
    def get_inter_community_edges(self) -> List[Tuple[str, str, Dict]]:
        """Get edges between communities."""
        # Build node to community mapping
        node_to_comm = {}
        for comm_id, comm in self.communities.items():
            for member in comm.members:
                node_to_comm[member] = comm_id
        
        inter_edges = []
        for u, v, data in self.graph.edges(data=True):
            if node_to_comm.get(u) != node_to_comm.get(v):
                inter_edges.append((u, v, data))
        
        return inter_edges
    
    def get_community_bridges(self) -> List[Tuple[str, str]]:
        """Get bridge nodes connecting communities."""
        bridges = list(nx.bridges(self.graph))
        return bridges
    
    def export_communities_json(self) -> str:
        """Export communities as JSON."""
        data = {
            'communities': [],
            'inter_community_edges': [],
        }
        
        for comm_id, comm in self.communities.items():
            data['communities'].append({
                'id': comm.id,
                'members': comm.members,
                'size': comm.size,
                'modularity': comm.modularity,
                'central_nodes': comm.central_nodes,
                'summary': comm.summary,
            })
        
        inter_edges = self.get_inter_community_edges()
        for source, target, attrs in inter_edges:
            data['inter_community_edges'].append({
                'source': source,
                'target': target,
                'attributes': attrs,
            })
        
        return json.dumps(data, indent=2)
    
    def get_community_statistics(self) -> Dict[str, Any]:
        """Get statistics about communities."""
        if not self.communities:
            return {}
        
        sizes = [c.size for c in self.communities.values()]
        
        return {
            'num_communities': len(self.communities),
            'avg_community_size': np.mean(sizes) if sizes else 0,
            'max_community_size': max(sizes) if sizes else 0,
            'min_community_size': min(sizes) if sizes else 0,
            'modularity': list(self.communities.values())[0].modularity if self.communities else 0,
        }
