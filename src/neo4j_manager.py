import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .entity_extractor import Entity, Relation, ExtractionResult

@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]
    
@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]

class Neo4jManager:
    """Manage Neo4j knowledge graph operations."""
    
    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to Neo4j database."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            self.logger.warning("neo4j package not installed, using in-memory graph")
            self.driver = None
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close connection."""
        if self.driver:
            self.driver.close()
    
    def create_constraints(self):
        """Create uniqueness constraints."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Person) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Organization) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Location) REQUIRE n.name IS UNIQUE",
        ]
        
        for query in queries:
            self.run_query(query)
    
    def run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Run a Cypher query."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def import_extraction(self, extraction: ExtractionResult):
        """Import extraction result into Neo4j."""
        # Create entities
        for entity in extraction.entities:
            self.create_entity(entity)
        
        # Create relations
        for relation in extraction.relations:
            self.create_relation(relation)
    
    def create_entity(self, entity: Entity):
        """Create an entity node."""
        query = f"""
        MERGE (n:{entity.entity_type} {{name: $name}})
        SET n += $properties
        SET n.entity_type = $entity_type
        RETURN n
        """
        
        self.run_query(query, {
            'name': entity.name,
            'entity_type': entity.entity_type,
            'properties': entity.properties,
        })
    
    def create_relation(self, relation: Relation):
        """Create a relation edge."""
        query = f"""
        MATCH (a:Entity {{name: $source}})
        MATCH (b:Entity {{name: $target}})
        MERGE (a)-[r:{relation.relation_type}]->(b)
        SET r.confidence = $confidence
        SET r.evidence = $evidence
        RETURN r
        """
        
        self.run_query(query, {
            'source': relation.source,
            'target': relation.target,
            'confidence': relation.confidence,
            'evidence': relation.evidence,
        })
    
    def generate_cypher(self, nl_query: str) -> str:
        """Generate Cypher query from natural language."""
        nl_lower = nl_query.lower()
        
        if 'find' in nl_lower or 'search' in nl_lower:
            return self._generate_search_cypher(nl_query)
        elif 'count' in nl_lower:
            return self._generate_count_cypher(nl_query)
        elif 'path' in nl_lower or 'connect' in nl_query:
            return self._generate_path_cypher(nl_query)
        elif 'community' in nl_lower or 'cluster' in nl_lower:
            return self._generate_community_cypher(nl_query)
        else:
            return self._generate_generic_cypher(nl_query)
    
    def _generate_search_cypher(self, nl_query: str) -> str:
        """Generate search Cypher query."""
        # Extract entity name from query
        words = nl_query.split()
        entity_name = ' '.join(words[words.index('find')+1:]) if 'find' in nl_query else words[-1]
        
        return f"""
        MATCH (n:Entity)
        WHERE n.name CONTAINS '{entity_name}'
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT 50
        """
    
    def _generate_count_cypher(self, nl_query: str) -> str:
        """Generate count Cypher query."""
        if 'person' in nl_query.lower():
            return "MATCH (n:Person) RETURN count(n) as count"
        elif 'organization' in nl_query.lower():
            return "MATCH (n:Organization) RETURN count(n) as count"
        else:
            return "MATCH (n:Entity) RETURN n.entity_type, count(n) as count ORDER BY count DESC"
    
    def _generate_path_cypher(self, nl_query: str) -> str:
        """Generate path-finding Cypher query."""
        return """
        MATCH path = shortestPath(
            (a:Entity {name: 'Source'})-[*]-(b:Entity {name: 'Target'})
        )
        RETURN path
        LIMIT 5
        """
    
    def _generate_community_cypher(self, nl_query: str) -> str:
        """Generate community detection Cypher query."""
        return """
        CALL gds.louvain.stream('knowledge-graph')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).name AS name, communityId
        ORDER BY communityId
        """
    
    def _generate_generic_cypher(self, nl_query: str) -> str:
        """Generate generic Cypher query."""
        return """
        MATCH (n:Entity)-[r]->(m:Entity)
        RETURN n.name, type(r), m.name
        LIMIT 100
        """
    
    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> Dict:
        """Get neighbors of an entity."""
        query = f"""
        MATCH (n:Entity {{name: $name}})
        CALL {{
            WITH n
            MATCH path = (n)-[*1..{depth}]-(m)
            RETURN path
        }}
        RETURN path
        LIMIT 100
        """
        
        results = self.run_query(query, {'name': entity_name})
        
        nodes = {}
        edges = []
        
        for record in results:
            path = record.get('path')
            if path:
                for node in path.nodes:
                    nodes[node['name']] = {
                        'name': node['name'],
                        'type': list(node.labels)[0] if node.labels else 'Entity',
                    }
                for rel in path.relationships:
                    edges.append({
                        'source': rel.start_node['name'],
                        'target': rel.end_node['name'],
                        'type': rel.type,
                    })
        
        return {'nodes': list(nodes.values()), 'edges': edges}
    
    def get_community_summary(self, community_id: int) -> Dict:
        """Get summary of a community."""
        query = """
        CALL gds.louvain.stream('knowledge-graph')
        YIELD nodeId, communityId
        WHERE communityId = $communityId
        RETURN collect(gds.util.asNode(nodeId).name) as members
        """
        
        result = self.run_query(query, {'communityId': community_id})
        
        if result:
            members = result[0].get('members', [])
            return {
                'community_id': community_id,
                'members': members,
                'size': len(members),
            }
        
        return {'community_id': community_id, 'members': [], 'size': 0}
    
    def get_all_communities(self) -> List[Dict]:
        """Get all communities."""
        query = """
        CALL gds.louvain.stream('knowledge-graph')
        YIELD nodeId, communityId
        RETURN communityId, collect(gds.util.asNode(nodeId).name) as members
        ORDER BY communityId
        """
        
        results = self.run_query(query)
        
        communities = []
        for record in results:
            communities.append({
                'community_id': record['communityId'],
                'members': record['members'],
                'size': len(record['members']),
            })
        
        return communities
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'total_edges': "MATCH ()-[r]->() RETURN count(r) as count",
            'node_types': "MATCH (n) RETURN labels(n) as labels, count(n) as count",
        }
        
        stats = {}
        for key, query in queries.items():
            result = self.run_query(query)
            stats[key] = result[0] if result else {}
        
        return stats
