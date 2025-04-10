def visualize_assembly_results(self, query, results, output_path=None):
    """
    Visualize assembly retrieval results
    
    Args:
        query (str): Query STEP ID or graph file path
        results (dict): Search results from retrieve_similar_assemblies
        output_path (str): Optional path to save the visualization
        
    Returns:
        output_path (str): Path to the saved visualization
    """
    if not self.use_assembly:
        print("Assembly similarity not enabled")
        return None
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.config["data"]["output_dir"], "results/assembly_searches")
        os.makedirs(output_dir, exist_ok=True)

        # Get query STEP ID
        if "query_step_id" in results:
            query_step_id = results["query_step_id"]
        elif os.path.exists(query) and query.endswith("_hierarchical.graphml"):
            query_step_id = self.assembly_encoder.extract_step_id(query)
        else:
            query_step_id = query
        
        # Create default output path if none provided
        if output_path is None:
            results_dir = os.path.join(self.config["data"]["output_dir"], "results", "assembly_searches")
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, f"assembly_search_{query_step_id}.html")
        
        # Create an HTML visualization
        # Determine if fusion is enabled
        fusion_enabled = self.config.get("assembly", {}).get("fusion", {}).get("enabled", False)
        fusion_method = self.config.get("assembly", {}).get("fusion", {}).get("method", "weighted")
        
        fusion_info = ""
        if fusion_enabled and self.assembly_fusion is not None:
            # Get fusion weights
            graph_weight = self.config.get("assembly", {}).get("fusion", {}).get("graph_weight", 0.6)
            part_weight = self.config.get("assembly", {}).get("fusion", {}).get("part_weight", 0.4)
            part_aggregation = self.config.get("assembly", {}).get("fusion", {}).get("part_aggregation", "mean")
            
            fusion_info = f"<p>Using fusion approach: {fusion_method} (Graph: {graph_weight:.1f}, Parts: {part_weight:.1f}, Aggregation: {part_aggregation})</p>"
        else:
            fusion_info = "<p>Using graph-only approach (no fusion)</p>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Assembly Search Results for {query_step_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .results {{ display: flex; flex-direction: column; }}
                .result {{ margin: 10px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }}
                .step-id {{ font-weight: bold; font-size: 18px; }}
                .similarity {{ 
                    padding: 5px 10px;
                    border-radius: 10px;
                    font-weight: bold;
                }}
                .similarity-high {{ background-color: #d4edda; color: #155724; }}
                .similarity-medium {{ background-color: #fff3cd; color: #856404; }}
                .similarity-low {{ background-color: #f8d7da; color: #721c24; }}
                .details {{ margin-top: 10px; }}
                .info-box {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Assembly Search Results</h1>
            <h2>Query: {query_step_id}</h2>
            
            <div class="info-box">
                {fusion_info}
            </div>
            
            <div class="results">
        """
        
        # Add results
        for i, (path, similarity) in enumerate(zip(
                results["paths"],
                results.get("similarities", [None] * len(results["paths"]))
            )):
            
            # Get assembly info
            step_id = self.assembly_encoder.extract_step_id(path) if path else "unknown"
            
            # Format similarity score and determine class
            similarity_str = f"{similarity:.1f}%" if similarity is not None else "N/A"
            if similarity >= 70:
                similarity_class = "similarity-high"
            elif similarity >= 50:
                similarity_class = "similarity-medium"
            else:
                similarity_class = "similarity-low"
            
            # Add this result to the HTML
            html_content += f"""
                <div class="result">
                    <div class="result-header">
                        <div class="step-id">#{i+1}: STEP ID: {step_id}</div>
                        <div class="similarity {similarity_class}">Similarity: {similarity_str}</div>
                    </div>
                    <div class="details">
                        <p>Graph file: {os.path.basename(path) if path else "unknown"}</p>
                    </div>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Assembly visualization saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error visualizing assembly results: {e}")
        import traceback
        traceback.print_exc()
        return None
