import os
import json
import shutil
from PIL import Image
import torch
import numpy as np

class DataProcessor:
    """
    Process CAD data from the STEP file outputs
    """
    def __init__(self, output_dir='data/output'):
        """
        Initialize the data processor
        
        Args:
            output_dir (str): Directory to store processed data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_step_output(self, step_output_dir):
        """
        Process a single STEP file output directory
        
        Args:
            step_output_dir (str): Directory containing STEP output files
            
        Returns:
            part_info (list): List of dictionaries with part information
        """
        step_id = os.path.basename(step_output_dir)
        bom_file = os.path.join(step_output_dir, f"{step_id}_bom.json")
        
        if not os.path.exists(bom_file):
            print(f"BOM file not found: {bom_file}")
            return []
        
        # Load BOM data
        try:
            with open(bom_file, 'r') as f:
                bom_data = json.load(f)
                
            # Verify we got a dictionary
            if not isinstance(bom_data, dict):
                print(f"Warning: BOM file {bom_file} did not contain a valid dictionary")
                bom_data = {}
        except Exception as e:
            print(f"Error loading BOM file {bom_file}: {e}")
            bom_data = {}
        
        # Process part images and metadata
        part_info = []
        
        images_dir = os.path.join(step_output_dir, "images")
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return []
        
        # Collect all part images
        for file_name in os.listdir(images_dir):
            # Skip full assembly images
            if file_name.endswith("_full_assembly.png"):
                continue
            
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(images_dir, file_name)
                
                # Extract part ID from filename
                part_id = os.path.splitext(file_name)[0]
                
                # Find corresponding metadata in BOM
                part_metadata = None
                
                # Handle different BOM structures
                if "parts" in bom_data and isinstance(bom_data.get("parts"), list):
                    # Standard structure with parts list
                    for part in bom_data.get("parts", []):
                        # Check if part is a dictionary
                        if not isinstance(part, dict):
                            continue
                            
                        if part.get("id") == part_id or part.get("name") == part_id:
                            part_metadata = part
                            break
                else:
                    # Alternative structure where BOM data is directly key-value pairs
                    # This handles the case where the file structure is different
                    if part_id in bom_data and isinstance(bom_data[part_id], dict):
                        part_metadata = bom_data[part_id]
                    elif isinstance(bom_data, dict):
                        # Try to find the part by iterating through all items
                        for key, value in bom_data.items():
                            if isinstance(value, dict):
                                # Check if this item matches our part_id
                                if key == part_id or value.get("id") == part_id or value.get("name") == part_id:
                                    part_metadata = value
                                    break
                
                # Create part info with parent STEP file and part name for easy retrieval
                part_info.append({
                    "step_id": step_id,
                    "part_id": part_id,
                    "image_path": image_path,
                    "metadata": part_metadata,
                    "parent_step": step_id,  # Store parent STEP file name
                    "part_name": part_id if part_metadata is None else part_metadata.get("name", part_id)  # Get part name from metadata or use ID
                })
        
        return part_info
    
    def process_dataset(self, dataset_dir):
        """
        Process the entire dataset of STEP output directories
        
        Args:
            dataset_dir (str): Directory containing multiple STEP output directories
            
        Returns:
            all_parts (list): List of dictionaries with all part information
        """
        all_parts = []
        
        # Get list of files to process
        items = sorted(os.listdir(dataset_dir))
        total_items = len(items)
        
        for idx, item in enumerate(items):
            item_path = os.path.join(dataset_dir, item)
            
            if os.path.isdir(item_path):
                # Check if this looks like a STEP output directory
                if os.path.exists(os.path.join(item_path, f"{item}_bom.json")):
                    print(f"Processing STEP output: {item} [{idx+1}/{total_items}]")
                    try:
                        parts = self.process_step_output(item_path)
                        all_parts.extend(parts)
                    except Exception as e:
                        print(f"Error processing STEP output {item}: {e}")
                        # Continue with next file instead of crashing
                        continue
        
        print(f"Processed {len(all_parts)} parts from {dataset_dir}")
        return all_parts

    def copy_to_flat_structure(self, all_parts, dest_dir=None):
        """
        Copy all part images to a flat directory structure for easier processing
        
        Args:
            all_parts (list): List of dictionaries with part information
            dest_dir (str): Destination directory (default: self.output_dir/images)
            
        Returns:
            image_mapping (dict): Mapping of original to new image paths
        """
        if dest_dir is None:
            dest_dir = os.path.join(self.output_dir, "images")
        
        os.makedirs(dest_dir, exist_ok=True)
        
        image_mapping = {}
        
        for part in all_parts:
            src_path = part["image_path"]
            
            if not os.path.exists(src_path):
                print(f"Warning: Image not found: {src_path}")
                continue
            
            # Create a unique filename
            filename = f"{part['step_id']}_{part['part_id']}.png"
            dest_path = os.path.join(dest_dir, filename)
            
            # Copy the file
            shutil.copy2(src_path, dest_path)
            
            # Store the mapping
            image_mapping[src_path] = dest_path
            
            # Update the part info
            part["flat_image_path"] = dest_path
        
        print(f"Copied {len(image_mapping)} images to {dest_dir}")
        return image_mapping

    def save_processed_data(self, all_parts, output_file=None):
        """
        Save processed part data to a JSON file
        
        Args:
            all_parts (list): List of dictionaries with part information
            output_file (str): Output JSON file path
            
        Returns:
            output_file (str): Path to the saved file
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "processed_parts.json")
        
        # Create a serializable version of the data
        serializable_parts = []
        for part in all_parts:
            # Create a copy with only serializable fields
            part_data = {
                "step_id": part["step_id"],
                "part_id": part["part_id"],
                "image_path": part["image_path"],
                "flat_image_path": part.get("flat_image_path"),
            }
            
            # Add metadata if available
            if part.get("metadata"):
                part_data["metadata"] = part["metadata"]
            
            serializable_parts.append(part_data)
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(serializable_parts, f, indent=2)
        
        print(f"Saved processed data to {output_file}")
        return output_file
