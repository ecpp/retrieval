import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class RetrievalEvaluator:
    """
    Evaluates the performance of the retrieval system
    """
    def __init__(self, image_encoder, vector_db, output_dir='data/evaluation'):
        """
        Initialize the evaluator
        
        Args:
            image_encoder: Encoder to use for generating query embeddings
            vector_db: Vector database for retrieval
            output_dir (str): Directory to store evaluation results
        """
        self.image_encoder = image_encoder
        self.vector_db = vector_db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_queries(self, query_images, ground_truth=None, top_k=(1, 5, 10, 20)):
        """
        Evaluate retrieval performance for a set of query images
        
        Args:
            query_images (list): List of query image paths
            ground_truth (dict): Optional mapping of query to relevant results
            top_k (tuple): Tuple of k values for precision/recall calculation
            
        Returns:
            results (dict): Dictionary containing evaluation results
        """
        results = {
            "queries": [],
            "metrics": {k: {"precision": 0, "recall": 0} for k in top_k},
            "avg_query_time": 0,
        }
        
        total_query_time = 0
        
        for query_path in tqdm(query_images, desc="Evaluating queries"):
            # Encode the query image
            query_embedding = self.image_encoder.encode_image(query_path)
            
            if query_embedding is None:
                print(f"Skipping query {query_path} due to encoding error")
                continue
            
            # Perform the search
            import time
            start_time = time.time()
            search_results = self.vector_db.search(query_embedding, k=max(top_k))
            query_time = time.time() - start_time
            total_query_time += query_time
            
            # Calculate metrics if ground truth is available
            query_metrics = {}
            if ground_truth and query_path in ground_truth:
                relevant = set(ground_truth[query_path])
                for k in top_k:
                    retrieved = set(search_results["paths"][:k])
                    relevant_retrieved = relevant.intersection(retrieved)
                    
                    precision = len(relevant_retrieved) / k if k > 0 else 0
                    recall = len(relevant_retrieved) / len(relevant) if len(relevant) > 0 else 0
                    
                    query_metrics[k] = {
                        "precision": precision,
                        "recall": recall
                    }
                    
                    # Update global metrics
                    results["metrics"][k]["precision"] += precision
                    results["metrics"][k]["recall"] += recall
            
            # Store query results
            results["queries"].append({
                "query_path": query_path,
                "results": search_results,
                "query_time": query_time,
                "metrics": query_metrics
            })
        
        # Calculate averages
        num_queries = len(results["queries"])
        if num_queries > 0:
            results["avg_query_time"] = total_query_time / num_queries
            
            for k in top_k:
                results["metrics"][k]["precision"] /= num_queries
                results["metrics"][k]["recall"] /= num_queries
        
        return results
    
    def visualize_retrieval(self, query_results, max_queries=5, max_results=5):
        """
        Generate visualizations of query results
        
        Args:
            query_results (list): List of query result dictionaries
            max_queries (int): Maximum number of queries to visualize
            max_results (int): Maximum number of results to show per query
            
        Returns:
            output_paths (list): List of paths to generated visualizations
        """
        output_paths = []
        
        for i, query in enumerate(query_results[:max_queries]):
            query_path = query["query_path"]
            results = query["results"]
            
            # Create figure
            fig, axes = plt.subplots(1, max_results + 1, figsize=(3 * (max_results + 1), 3))
            
            # Display query
            query_img = Image.open(query_path).convert('RGB')
            axes[0].imshow(query_img)
            axes[0].set_title("Query")
            axes[0].axis('off')
            
            # Display results
            for j, (path, distance) in enumerate(zip(results["paths"][:max_results], results["distances"][:max_results])):
                if j >= max_results:
                    break
                    
                if path and os.path.exists(path):
                    result_img = Image.open(path).convert('RGB')
                    axes[j+1].imshow(result_img)
                    axes[j+1].set_title(f"Dist: {distance:.2f}")
                    axes[j+1].axis('off')
                else:
                    axes[j+1].text(0.5, 0.5, "Image not found", horizontalalignment='center')
                    axes[j+1].axis('off')
            
            # Save visualization
            output_path = os.path.join(self.output_dir, f"query_{i+1}_results.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            output_paths.append(output_path)
        
        return output_paths
    
    def plot_metrics(self, results):
        """
        Generate plots of evaluation metrics
        
        Args:
            results (dict): Evaluation results dictionary
            
        Returns:
            output_path (str): Path to the generated plot
        """
        k_values = sorted(results["metrics"].keys())
        precision = [results["metrics"][k]["precision"] for k in k_values]
        recall = [results["metrics"][k]["recall"] for k in k_values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_values, precision, 'o-', label='Precision@k')
        ax.plot(k_values, recall, 's-', label='Recall@k')
        
        ax.set_xlabel('k')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall at different k values')
        ax.legend()
        ax.grid(True)
        
        output_path = os.path.join(self.output_dir, "precision_recall.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def save_results(self, results, output_file=None):
        """
        Save evaluation results to a JSON file
        
        Args:
            results (dict): Evaluation results dictionary
            output_file (str): Path to the output file
            
        Returns:
            output_file (str): Path to the saved file
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "evaluation_results.json")
        
        # Create a serializable version of the results
        serializable_results = {
            "metrics": results["metrics"],
            "avg_query_time": results["avg_query_time"],
            "queries": []
        }
        
        for query in results["queries"]:
            serializable_query = {
                "query_path": query["query_path"],
                "query_time": query["query_time"],
                "metrics": query["metrics"],
                "results": {
                    "paths": query["results"]["paths"],
                    "distances": query["results"]["distances"]
                }
            }
            serializable_results["queries"].append(serializable_query)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved evaluation results to {output_file}")
        return output_file
