import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .entity_extractor import ExtractionResult
from .graphrag_retrieval import GraphRAGRetriever, RetrievalResult

@dataclass
class BenchmarkQuestion:
    """Benchmark question for comparison."""
    id: str
    question: str
    type: str  # single_hop, multi_hop, aggregation, reasoning
    expected_answer: str
    difficulty: str  # easy, medium, hard
    
@dataclass
class BenchmarkResult:
    """Result of a single benchmark question."""
    question_id: str
    question: str
    vector_rag_answer: str
    graph_rag_answer: str
    vector_rag_correct: bool
    graph_rag_correct: bool
    vector_rag_time: float
    graph_rag_time: float
    vector_rag_reasoning: List[str] = field(default_factory=list)
    graph_rag_reasoning: List[str] = field(default_factory=list)

class VectorRAGSimulator:
    """Simulate vector RAG for comparison."""
    
    def __init__(self, extraction: ExtractionResult):
        self.extraction = extraction
        self.chunks = self._create_chunks()
    
    def _create_chunks(self) -> List[Dict]:
        """Create text chunks for vector retrieval."""
        chunks = []
        
        # Create entity chunks
        for entity in self.extraction.entities:
            chunks.append({
                'text': f"{entity.name} is a {entity.entity_type}",
                'entity': entity.name,
                'type': 'entity',
            })
        
        # Create relation chunks
        for relation in self.extraction.relations:
            chunks.append({
                'text': f"{relation.source} {relation.relation_type.lower().replace('_', ' ')} {relation.target}",
                'source': relation.source,
                'target': relation.target,
                'type': 'relation',
            })
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Simulate vector retrieval."""
        start_time = time.time()
        
        query_lower = query.lower()
        scored_chunks = []
        
        for chunk in self.chunks:
            # Simple keyword matching as proxy for embedding similarity
            text_lower = chunk['text'].lower()
            words = set(query_lower.split())
            chunk_words = set(text_lower.split())
            overlap = len(words.intersection(chunk_words))
            score = overlap / max(len(words), 1)
            scored_chunks.append((score, chunk))
        
        # Get top-k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:top_k]
        
        context = "\n".join([c[1]['text'] for c in top_chunks])
        
        elapsed = time.time() - start_time
        
        return RetrievalResult(
            answer=context if context else "No relevant context found",
            context=context,
            source_nodes=[c[1].get('entity', c[1].get('source', '')) for c in top_chunks],
            reasoning_path=[],
            retrieval_method="vector_rag",
            confidence=top_chunks[0][0] if top_chunks else 0,
            metadata={'time': elapsed},
        )

class GraphRAGBenchmark:
    """Benchmark comparing Vector RAG vs GraphRAG."""
    
    def __init__(self, extraction: ExtractionResult):
        self.extraction = extraction
        self.graphrag = GraphRAGRetriever(extraction)
        self.vectorrag = VectorRAGSimulator(extraction)
        self.logger = logging.getLogger(__name__)
        
        self.benchmark_questions = self._create_benchmark_questions()
        self.results: List[BenchmarkResult] = []
    
    def _create_benchmark_questions(self) -> List[BenchmarkQuestion]:
        """Create benchmark questions."""
        questions = []
        
        # Get some entities from extraction
        entity_names = [e.name for e in self.extraction.entities[:5]]
        
        if len(entity_names) >= 2:
            # Single-hop questions
            questions.append(BenchmarkQuestion(
                id="single_1",
                question=f"What is {entity_names[0]}?",
                type="single_hop",
                expected_answer="",
                difficulty="easy",
            ))
            
            # Multi-hop questions
            questions.append(BenchmarkQuestion(
                id="multi_1",
                question=f"What is the relationship between {entity_names[0]} and {entity_names[1]}?",
                type="multi_hop",
                expected_answer="",
                difficulty="medium",
            ))
            
            questions.append(BenchmarkQuestion(
                id="multi_2",
                question=f"Find the path between {entity_names[0]} and {entity_names[1]}",
                type="multi_hop",
                expected_answer="",
                difficulty="hard",
            ))
        
        # Aggregation questions
        questions.append(BenchmarkQuestion(
            id="agg_1",
            question="What are the main communities in this graph?",
            type="aggregation",
            expected_answer="",
            difficulty="medium",
        ))
        
        questions.append(BenchmarkQuestion(
            id="reason_1",
            question="Summarize the knowledge graph",
            type="reasoning",
            expected_answer="",
            difficulty="medium",
        ))
        
        return questions
    
    def run_benchmark(self, max_questions: int = 10) -> Dict[str, Any]:
        """Run the full benchmark."""
        self.results = []
        
        for question in self.benchmark_questions[:max_questions]:
            result = self._evaluate_question(question)
            self.results.append(result)
        
        return self.generate_report()
    
    def _evaluate_question(self, question: BenchmarkQuestion) -> BenchmarkResult:
        """Evaluate a single question."""
        # Vector RAG
        start = time.time()
        vector_result = self.vectorrag.retrieve(question.question)
        vector_time = time.time() - start
        
        # Graph RAG
        start = time.time()
        graph_result = self.graphrag.retrieve(question.question)
        graph_time = time.time() - start
        
        # Evaluate correctness (simplified)
        vector_correct = len(vector_result.context) > 0
        graph_correct = len(graph_result.context) > 0
        
        return BenchmarkResult(
            question_id=question.id,
            question=question.question,
            vector_rag_answer=vector_result.context[:200],
            graph_rag_answer=graph_result.context[:200],
            vector_rag_correct=vector_correct,
            graph_rag_correct=graph_correct,
            vector_rag_time=vector_time,
            graph_rag_time=graph_time,
            vector_rag_reasoning=[],
            graph_rag_reasoning=graph_result.reasoning_path,
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report."""
        if not self.results:
            return {'error': 'No results'}
        
        vector_correct = sum(1 for r in self.results if r.vector_rag_correct)
        graph_correct = sum(1 for r in self.results if r.graph_rag_correct)
        
        vector_times = [r.vector_rag_time for r in self.results]
        graph_times = [r.graph_rag_time for r in self.results]
        
        report = {
            'summary': {
                'total_questions': len(self.results),
                'vector_rag_accuracy': vector_correct / len(self.results),
                'graph_rag_accuracy': graph_correct / len(self.results),
                'vector_rag_avg_time': np.mean(vector_times),
                'graph_rag_avg_time': np.mean(graph_times),
            },
            'by_type': self._group_by_type(),
            'by_difficulty': self._group_by_difficulty(),
            'questions': [
                {
                    'id': r.question_id,
                    'question': r.question,
                    'vector_correct': r.vector_rag_correct,
                    'graph_correct': r.graph_rag_correct,
                    'vector_time': r.vector_rag_time,
                    'graph_time': r.graph_rag_time,
                }
                for r in self.results
            ],
        }
        
        return report
    
    def _group_by_type(self) -> Dict[str, Dict]:
        """Group results by question type."""
        by_type = {}
        for r in self.results:
            q = next((q for q in self.benchmark_questions if q.id == r.question_id), None)
            if q:
                if q.type not in by_type:
                    by_type[q.type] = {'vector': 0, 'graph': 0, 'total': 0}
                by_type[q.type]['total'] += 1
                if r.vector_rag_correct:
                    by_type[q.type]['vector'] += 1
                if r.graph_rag_correct:
                    by_type[q.type]['graph'] += 1
        
        return by_type
    
    def _group_by_difficulty(self) -> Dict[str, Dict]:
        """Group results by difficulty."""
        by_diff = {}
        for r in self.results:
            q = next((q for q in self.benchmark_questions if q.id == r.question_id), None)
            if q:
                if q.difficulty not in by_diff:
                    by_diff[q.difficulty] = {'vector': 0, 'graph': 0, 'total': 0}
                by_diff[q.difficulty]['total'] += 1
                if r.vector_rag_correct:
                    by_diff[q.difficulty]['vector'] += 1
                if r.graph_rag_correct:
                    by_diff[q.difficulty]['graph'] += 1
        
        return by_diff
    
    def export_results(self, filepath: str):
        """Export results to JSON."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
