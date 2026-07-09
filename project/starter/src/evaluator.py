"""
Evaluation Metrics for Information Retrieval
"""
import numpy as np
from typing import Dict, List


class IRMetrics:
    """Information Retrieval evaluation metrics."""
    
    @staticmethod
    def recall_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Recall@k: fraction of relevant documents found in top-k.
        
        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}
            k: cutoff for top-k evaluation
        """
        recall_scores = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            # YOUR CODE HERE: Calculate recall for this query
            relevant_docs = qrels[q_id]
            retrieved_docs = results[q_id][:k] # Taking only the top k relevant docs
            retrieved_relevant_docs = list(filter(lambda x : x in relevant_docs, retrieved_docs))
            recall = len(retrieved_relevant_docs) / len(relevant_docs)
            recall_scores.append(recall)
            
        return np.mean(recall_scores) if recall_scores else 0.0 # if no results have been returned then recall is zero

    @staticmethod
    def precision_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Precision@k: fraction of top-k that are relevant.
        """
        precision_scores = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            # YOUR CODE HERE: Calculate precision for this query
            relevant_docs = qrels[q_id]
            retrieved_docs = results[q_id][:k] # Taking only the top k relevant docs... should we only take the top k docs or all the docs? I think we should in the end we are setting a threshold of relevance... BUTTTTTT we must keep k the same for all metric
            retrieved_relevant_docs = list(filter(lambda x : x in relevant_docs, retrieved_docs))
            precision = len(retrieved_relevant_docs) / k
            precision_scores.append(precision)
            
        return np.mean(precision_scores) if precision_scores else 0.0

    @staticmethod
    def mrr(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        """
        #MRR is the reciprocal of the first index of a relevant document in the retrieved list. If no relevant document is found, the reciprocal rank is 0 for that query.
        reciprocal_ranks = []
        for q_id in results:
            if q_id not in qrels:
                continue
            
            # YOUR CODE HERE: Calculate MRR for this query
            relevant_docs = qrels[q_id]
            retrieved_docs = results[q_id]
            rank = 0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    rank = i + 1 # add one as indexing starts at 0
                    break
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)

            
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def evaluate_retrieval(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> Dict[str, float]:
        """
        Comprehensive evaluation with standard IR metrics.
        
        Returns:
            Dictionary with metric names and values
        """
        metrics = {
            'Recall@1': IRMetrics.recall_at_k(results, qrels, 1),
            'Recall@5': IRMetrics.recall_at_k(results, qrels, 5), 
            'Recall@10': IRMetrics.recall_at_k(results, qrels, 10),
            'Precision@5': IRMetrics.precision_at_k(results, qrels, 5),
            'MRR': IRMetrics.mrr(results, qrels)
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
        """Pretty print evaluation metrics."""
        print(f"\n📊 {title}")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric:12}: {value:.4f}")
        print("=" * 40)
