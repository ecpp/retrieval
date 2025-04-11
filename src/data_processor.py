import os
import json
import shutil
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import concurrent.futures
import multiprocessing

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

        # Check for full assembly image
        full_assembly_path = os.path.join(images_dir, f"{step_id}_full_assembly.png")
        has_full_assembly = os.path.exists(full_assembly_path)

        # Collect all part images
        for file_name in os.listdir(images_dir):
            # Skip full assembly images - they will be handled separately
            if "_full_assembly" in file_name:
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
                    # Alternative structure where part data is directly in the BOM
                    for key, value in bom_data.items():
                        if not isinstance(value, dict):
                            continue

                        if key == part_id or value.get("id") == part_id or value.get("name") == part_id:
                            part_metadata = value
                            break

                # Append part info
                part_info.append({
                    "step_id": step_id,
                    "part_id": part_id,
                    "image_path": image_path,
                    "bom_file": bom_file,
                    "metadata": part_metadata,
                    "has_full_assembly": has_full_assembly,
                    "full_assembly_path": full_assembly_path if has_full_assembly else None
                })

        return part_info

    def process_dataset(self, dataset_dir):
        """
        Process the entire dataset

        Args:
            dataset_dir (str): Root directory of the dataset

        Returns:
            all_parts (list): List of dictionaries with part information
        """
        all_parts = []

        # Get list of files to process
        items = sorted(os.listdir(dataset_dir))
        total_items = len(items)

        for item in tqdm(items, desc="Processing STEP outputs", unit="file"):
            item_path = os.path.join(dataset_dir, item)

            if os.path.isdir(item_path):
                # Check if this looks like a STEP output directory
                if os.path.exists(os.path.join(item_path, f"{item}_bom.json")):
                    try:
                        parts = self.process_step_output(item_path)
                        all_parts.extend(parts)
                    except Exception as e:
                        print(f"Error processing STEP output {item}: {e}")
                        # Continue with next file instead of crashing
                        continue

        print(f"Processed {len(all_parts)} parts from {dataset_dir}")
        return all_parts

    def copy_to_flat_structure(self, all_parts, dest_dir=None, max_workers=None):
        """
        Copy all part images to a flat directory structure for easier processing

        Args:
            all_parts (list): List of dictionaries with part information
            dest_dir (str): Destination directory (default: self.output_dir/images)
            max_workers (int): Maximum number of threads to use (default: CPU count / 2)

        Returns:
            image_mapping (dict): Mapping of original to new image paths
        """
        if dest_dir is None:
            dest_dir = os.path.join(self.output_dir, "images")

        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() // 2)

        os.makedirs(dest_dir, exist_ok=True)

        # Create directory for full assembly images
        full_assembly_dir = os.path.join(self.output_dir, "full_assembly_images")
        os.makedirs(full_assembly_dir, exist_ok=True)

        def copy_single_file(part):
            src_path = part["image_path"]
            result = {"success": False}

            if not os.path.exists(src_path):
                return {
                    "success": False,
                    "src_path": src_path,
                    "error": "Source file not found"
                }

            # Create a unique filename
            filename = f"{part['step_id']}_{part['part_id']}.png"
            dest_path = os.path.join(dest_dir, filename)

            try:
                # Copy the file
                shutil.copy2(src_path, dest_path)

                # Copy full assembly image if available
                if part["has_full_assembly"] and part["full_assembly_path"]:
                    assembly_filename = f"{part['step_id']}_full_assembly.png"
                    assembly_dest_path = os.path.join(full_assembly_dir, assembly_filename)

                    # Only copy if it doesn't exist yet (to avoid duplicates)
                    if not os.path.exists(assembly_dest_path):
                        shutil.copy2(part["full_assembly_path"], assembly_dest_path)

                # Return the result
                return {
                    "success": True,
                    "src_path": src_path,
                    "dest_path": dest_path,
                    "part": part
                }
            except Exception as e:
                return {
                    "success": False,
                    "src_path": src_path,
                    "error": str(e)
                }

        # Copy files in parallel
        copied_count = 0
        error_count = 0
        image_mapping = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_part = {executor.submit(copy_single_file, part): part for part in all_parts}

            with tqdm(total=len(future_to_part), desc="Copying images", unit="file") as pbar:
                for future in concurrent.futures.as_completed(future_to_part):
                    result = future.result()
                    pbar.update(1)

                    if result["success"]:
                        copied_count += 1
                        image_mapping[result["src_path"]] = result["dest_path"]
                    else:
                        error_count += 1
                        print(f"Error copying {result['src_path']}: {result.get('error', 'Unknown error')}")

        # Look for any existing full assembly images that might have been missed
        assembly_copied = 0
        for item in os.listdir(os.path.join(self.output_dir, "images")):
            if "_full_assembly" in item:
                src_path = os.path.join(self.output_dir, "images", item)
                dest_path = os.path.join(full_assembly_dir, item)
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(src_path, dest_path)
                        assembly_copied += 1
                    except Exception as e:
                        print(f"Error copying full assembly image {item}: {e}")

        print(f"Copied {copied_count} part images with {error_count} errors")
        if assembly_copied > 0:
            print(f"Copied {assembly_copied} additional full assembly images")
        return image_mapping

    def save_processed_data(self, all_parts, output_file=None):
        """
        Save processed data to a JSON file

        Args:
            all_parts (list): List of dictionaries with part information
            output_file (str): Output file path (default: self.output_dir/processed_data.json)

        Returns:
            output_file (str): Path to the output file
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "processed_data.json")

        # Prepare data for serialization
        serializable_data = []
        for part in all_parts:
            # Copy the part dictionary
            part_copy = part.copy()

            # Remove any non-serializable data
            if "metadata" in part_copy and isinstance(part_copy["metadata"], dict):
                # Keep only non-complex metadata
                part_copy["metadata"] = {k: v for k, v in part_copy["metadata"].items()
                                       if isinstance(v, (str, int, float, bool, list, dict))}

            serializable_data.append(part_copy)

        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            print(f"Saved processed data to {output_file}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

        return output_file
