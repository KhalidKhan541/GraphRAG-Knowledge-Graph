import json
import logging
from typing import List

from src.entity_extractor import EntityExtractor, ExtractionResult
from src.neo4j_manager import Neo4jManager
from src.community_detection import CommunityDetector
from src.query_engine import GraphQueryEngine
from src.graphrag_retrieval import GraphRAGRetriever
from src.comparison import GraphRAGBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_extraction() -> ExtractionResult:
    """Create a sample extraction for demonstration."""
    from src.entity_extractor import Entity, Relation
    
    entities = [
        Entity(name="Google", entity_type="ORGANIZATION", properties={"founded": 1998}),
        Entity(name="DeepMind", entity_type="ORGANIZATION", properties={"acquired_by": "Google"}),
        Entity(name="OpenAI", entity_type="ORGANIZATION", properties={"founded": 2015}),
        Entity(name="Sam Altman", entity_type="PERSON", properties={"role": "CEO"}),
        Entity(name="Demis Hassabis", entity_type="PERSON", properties={"role": "CEO"}),
        Entity(name="AlphaGo", entity_type="TECHNOLOGY", properties={"year": 2016}),
        Entity(name="GPT-4", entity_type="TECHNOLOGY", properties={"year": 2023}),
        Entity(name="London", entity_type="LOCATION", properties={"country": "UK"}),
        Entity(name="San Francisco", entity_type="LOCATION", properties={"country": "USA"}),
        Entity(name="Reinforcement Learning", entity_type="CONCEPT", properties={"field": "AI"}),
    ]
    
    relations = [
        Relation(source="Google", target="DeepMind", relation_type="ACQUIRED", confidence=0.95),
        Relation(source="Demis Hassabis", target="DeepMind", relation_type="WORKS_AT", confidence=0.9),
        Relation(source="Demis Hassabis", target="London", relation_type="LOCATED_IN", confidence=0.85),
        Relation(source="DeepMind", target="AlphaGo", relation_type="DEVELOPED", confidence=0.95),
        Relation(source="AlphaGo", target="Reinforcement Learning", relation_type="USES", confidence=0.9),
        Relation(source="Sam Altman", target="OpenAI", relation_type="WORKS_AT", confidence=0.9),
        Relation(source="Sam Altman", target="San Francisco", relation_type="LOCATED_IN", confidence=0.85),
        Relation(source="OpenAI", target="GPT-4", relation_type="DEVELOPED", confidence=0.95),
        Relation(source="Google", target="London", relation_type="LOCATED_IN", confidence=0.8),
    ]
    
    return ExtractionResult(
        entities=entities,
        relations=relations,
        source_text="Sample AI companies knowledge graph",
        metadata={"source": "sample"},
    )

def demo_entity_extraction():
    """Demo entity extraction."""
    print("\n=== Entity Extraction Demo ===")
    
    extractor = EntityExtractor(use_llm=False)
    
    text = """Google acquired DeepMind in 2014. Demis Hassabis, the CEO of DeepMind, 
    is based in London. DeepMind developed AlphaGo using reinforcement learning. 
    OpenAI, founded in 2015, developed GPT-4. Sam Altman is the CEO of OpenAI, 
    based in San Francisco."""
    
    result = extractor.extract(text)
    
    print(f"Entities found: {len(result.entities)}")
    for entity in result.entities:
        print(f"  - {entity.name} ({entity.entity_type})")
    
    print(f"\nRelations found: {len(result.relations)}")
    for rel in result.relations:
        print(f"  - {rel.source} -> {rel.target} ({rel.relation_type})")
    
    return result

def demo_community_detection(extraction: ExtractionResult):
    """Demo community detection."""
    print("\n=== Community Detection Demo ===")
    
    detector = CommunityDetector()
    detector.load_from_extraction(extraction)
    communities = detector.detect_louvain()
    
    print(f"Communities detected: {len(communities)}")
    for comm_id, comm in communities.items():
        print(f"\nCommunity {comm_id}:")
        print(f"  Members: {comm.members}")
        print(f"  Size: {comm.size}")
        print(f"  Central nodes: {comm.central_nodes}")
        print(f"  Summary: {comm.summary}")
    
    return detector

def demo_query_engine(extraction: ExtractionResult):
    """Demo query engine."""
    print("\n=== Query Engine Demo ===")
    
    engine = GraphQueryEngine(extraction)
    
    queries = [
        "What is the relationship between Google and OpenAI?",
        "Find the path between DeepMind and GPT-4",
        "What are the main communities?",
        "Summarize the knowledge graph",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = engine.query(query)
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.reasoning_path:
            print(f"Reasoning: {result.reasoning_path}")

def demo_graphrag_retrieval(extraction: ExtractionResult):
    """Demo GraphRAG retrieval."""
    print("\n=== GraphRAG Retrieval Demo ===")
    
    retriever = GraphRAGRetriever(extraction)
    
    query = "What technologies does DeepMind use?"
    print(f"\nQuery: {query}")
    
    result = retriever.retrieve(query, method="multi_hop")
    print(f"Answer: {result.answer}")
    print(f"Method: {result.retrieval_method}")
    print(f"Context: {result.context[:200]}...")
    
    print("\nGlobal Summary:")
    print(retriever.get_global_summary())

def demo_benchmark(extraction: ExtractionResult):
    """Demo benchmark comparison."""
    print("\n=== Vector RAG vs GraphRAG Benchmark ===")
    
    benchmark = GraphRAGBenchmark(extraction)
    report = benchmark.run_benchmark(max_questions=5)
    
    print("\nBenchmark Report:")
    print(json.dumps(report['summary'], indent=2))
    
    print("\nBy Question Type:")
    for qtype, stats in report.get('by_type', {}).items():
        print(f"  {qtype}: Vector={stats['vector']}/{stats['total']}, "
              f"Graph={stats['graph']}/{stats['total']}")

def demo_neo4j_integration(extraction: ExtractionResult):
    """Demo Neo4j integration (in-memory mode)."""
    print("\n=== Neo4j Integration Demo ===")
    
    manager = Neo4jManager()
    manager.connect()
    
    # Import extraction
    manager.import_extraction(extraction)
    
    # Generate Cypher queries
    queries = [
        "Find all organizations",
        "Count people in the graph",
        "Show all communities",
    ]
    
    for nl_query in queries:
        cypher = manager.generate_cypher(nl_query)
        print(f"\nNL Query: {nl_query}")
        print(f"Cypher: {cypher}")
    
    manager.close()

def main():
    """Run all demos."""
    print("GraphRAG + Knowledge Graph Extraction System")
    print("=" * 50)
    
    extraction = create_sample_extraction()
    
    demo_entity_extraction()
    demo_community_detection(extraction)
    demo_query_engine(extraction)
    demo_graphrag_retrieval(extraction)
    demo_benchmark(extraction)
    demo_neo4j_integration(extraction)
    
    print("\n" + "=" * 50)
    print("All demos completed!")

if __name__ == "__main__":
    main()
