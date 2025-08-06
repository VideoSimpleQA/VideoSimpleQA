import os
import json
import glob
from pathlib import Path
from collections import defaultdict

def calculate_metrics(results):
    """
    Calculate evaluation metrics, convert to percentages (multiply by 100) 
    and round to one decimal place.
    
    Args:
        results (list): List of evaluation results where:
            - "A" represents correct answers
            - "B" represents incorrect answers  
            - "C" represents not attempted
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    total_samples = len(results)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "is_correct": 0.0,
            "is_incorrect": 0.0,
            "is_not_attempted": 0.0,
            "is_given_attempted": 0.0,
            "accuracy_given_attempted": 0.0,
            "f1_score": 0.0
        }
    
    # Calculate raw metrics
    correct_count = sum(1 for x in results if x == "A")
    incorrect_count = sum(1 for x in results if x == "B")
    not_attempted_count = sum(1 for x in results if x == "C")
    
    metrics = {
        "total_samples": total_samples,
        "is_correct": round((correct_count / total_samples) * 100, 1),
        "is_incorrect": round((incorrect_count / total_samples) * 100, 1),
        "is_not_attempted": round((not_attempted_count / total_samples) * 100, 1)
    }
    
    # Calculate attempt rate (correct + incorrect)
    attempted_rate = (correct_count + incorrect_count) / total_samples
    metrics["is_given_attempted"] = round(attempted_rate * 100, 1)
    
    # Calculate accuracy given attempts were made
    if (correct_count + incorrect_count) > 0:
        accuracy_given_attempted = correct_count / (correct_count + incorrect_count)
        metrics["accuracy_given_attempted"] = round(accuracy_given_attempted * 100, 1)
    else:
        metrics["accuracy_given_attempted"] = 0.0
    
    # Calculate F1 score
    correct_rate = correct_count / total_samples
    if (metrics["accuracy_given_attempted"] / 100 + correct_rate) > 0:
        f1_score = (2 * (metrics["accuracy_given_attempted"] / 100) * correct_rate
                   / ((metrics["accuracy_given_attempted"] / 100) + correct_rate))
        metrics["f1_score"] = round(f1_score * 100, 1)
    else:
        metrics["f1_score"] = 0.0
    
    return metrics

def extract_qa_grades(data):
    """
    Extract grades from each QA round and Multi_hop evaluation from the data.
    
    Args:
        data (list): List of evaluation data items
    
    Returns:
        tuple: (qa_grades_dict, multi_hop_grades_list)
            - qa_grades_dict: Dictionary mapping QA pair keys to grade lists
            - multi_hop_grades_list: List of multi-hop evaluation grades
    """
    qa_grades = {}
    multi_hop_grades = []
    
    for item in data:
        # Extract Multi_hop grade
        multi_hop_grade = item.get("Multi_hop_grade", "C")
        multi_hop_grades.append(multi_hop_grade)
        
        # Extract grades from each QA round
        qa_pair_num = 1
        while f"QA_Pair_{qa_pair_num}_grade" in item:
            qa_key = f"QA_Pair_{qa_pair_num}"
            grade = item.get(f"{qa_key}_grade", "C")
            
            if qa_key not in qa_grades:
                qa_grades[qa_key] = []
            qa_grades[qa_key].append(grade)
            qa_pair_num += 1
    
    return qa_grades, multi_hop_grades

def extract_model_name(filename):
    """
    Extract model name from filename.
    
    Example: evaluation_results_claude-sonnet4_30frames_round4.json -> claude-sonnet4
    
    Args:
        filename (str): Input filename
    
    Returns:
        str: Extracted model name
    """
    # Remove .json extension
    if filename.endswith('.json'):
        name = filename[:-5]
    else:
        name = filename
    
    # Remove evaluation_results_ prefix if present
    if name.startswith('evaluation_results_'):
        name = name[len('evaluation_results_'):]
    
    # Split by underscore
    parts = name.split('_')
    
    # Find model name parts (usually the first part, but handle special cases)
    model_parts = []
    
    for i, part in enumerate(parts):
        # Stop collecting model name parts when encountering:
        # 1. Parts starting with 'round' (e.g., round4, round1)
        # 2. Parts ending with 'frames' preceded by digits (e.g., 30frames)
        # 3. Standalone digit parts after model name
        if (part.startswith('round') and (len(part) == 5 or part[5:].isdigit())) or \
           (part.endswith('frames') and part[:-6].isdigit()) or \
           (part.isdigit() and len(model_parts) > 0):
            break
        else:
            model_parts.append(part)
    
    # If no model parts found, return first part
    if not model_parts:
        model_parts = [parts[0]] if parts else ['unknown']
    
    return '_'.join(model_parts)

def load_evaluation_results(file_path):
    """
    Load evaluation results from JSON file.
    
    Args:
        file_path (str): Path to the evaluation results file
    
    Returns:
        list or None: Loaded data or None if error occurred
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def process_single_file(file_path):
    """
    Process a single evaluation results file.
    
    Args:
        file_path (str): Path to the evaluation file
    
    Returns:
        dict or None: Dictionary with model metrics or None if error occurred
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist!")
        return None
    
    filename = os.path.basename(file_path)
    model_name = extract_model_name(filename)
    
    print(f"Processing {filename}...")
    
    # Load evaluation results
    data = load_evaluation_results(file_path)
    if data is None:
        return None
    
    # Extract grades from QA rounds and Multi_hop evaluation
    qa_grades, multi_hop_grades = extract_qa_grades(data)
    
    # Build simplified metrics structure
    model_metrics = {
        "total_samples": len(data)
    }
    
    # Add F-score for each QA round
    for qa_key, grades in qa_grades.items():
        metrics = calculate_metrics(grades)
        model_metrics[qa_key] = metrics['f1_score']
        print(f"  {qa_key}: F1 = {metrics['f1_score']}%")
    
    # Add Multi_hop F-score
    multi_hop_metrics = calculate_metrics(multi_hop_grades)
    model_metrics["Multi_hop"] = multi_hop_metrics['f1_score']
    print(f"  Multi_hop: F1 = {multi_hop_metrics['f1_score']}%")
    
    return {model_name: model_metrics}

def process_all_evaluation_files(eval_results_dir="evaluation_results"):
    """
    Process all evaluation result files and calculate metrics.
    
    Args:
        eval_results_dir (str): Directory containing evaluation result files
    
    Returns:
        dict: Dictionary containing metrics for all models
    """
    if not os.path.exists(eval_results_dir):
        print(f"Directory {eval_results_dir} does not exist!")
        return {}
    
    # Find all JSON files
    pattern = os.path.join(eval_results_dir, "*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No JSON files found in {eval_results_dir}")
        return {}
    
    print(f"Found {len(result_files)} evaluation result files:")
    for file in result_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    all_models_metrics = {}
    
    for file_path in result_files:
        model_metrics = process_single_file(file_path)
        if model_metrics:
            all_models_metrics.update(model_metrics)
    
    # Save all metrics to file
    output_file = os.path.join(eval_results_dir, "all_models_fscore.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_models_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\nAll F-scores saved to: {output_file}")
    
    # Create simplified summary table
    print_summary_table(all_models_metrics)
    
    return all_models_metrics

def print_summary_table(all_models_metrics):
    """
    Print a simplified summary table of F-scores for all models.
    
    Args:
        all_models_metrics (dict): Dictionary containing metrics for all models
    """
    if not all_models_metrics:
        print("No metrics to display")
        return
    
    # Get all QA pair keys
    all_qa_keys = set()
    for metrics in all_models_metrics.values():
        qa_keys = [k for k in metrics.keys() if k.startswith("QA_Pair_")]
        all_qa_keys.update(qa_keys)
    all_qa_keys = sorted(all_qa_keys)
    
    print("\n" + "="*100)
    print("MODEL F-SCORE SUMMARY (F1 Scores in %)")
    print("="*100)
    
    # Print header
    header = f"{'Model':<30} {'Samples':<8}"
    for qa_key in all_qa_keys:
        header += f" {qa_key:<12}"
    header += f" {'Multi_hop':<12}"
    print(header)
    print("-" * len(header))
    
    # Print F-scores for each model
    for model_name, metrics in all_models_metrics.items():
        row = f"{model_name:<30} {metrics.get('total_samples', 0):<8}"
        
        # Add F-score for each QA round
        for qa_key in all_qa_keys:
            f1_score = metrics.get(qa_key, 0.0)
            row += f" {f1_score:<12}"
        
        # Add Multi_hop F-score
        multi_hop_f1 = metrics.get("Multi_hop", 0.0)
        row += f" {multi_hop_f1:<12}"
        
        print(row)

def main():
    """
    Main function to run the F-Score calculator.
    """
    print("Multi-Round QA F-Score Calculator")
    print("="*70)
    
    # Process all evaluation result files
    results_directory = "evaluation_results"  # Default directory name
    print(f"Processing all JSON files in '{results_directory}' directory...")
    all_metrics = process_all_evaluation_files(results_directory)
    
    if all_metrics:
        print(f"\nProcessed {len(all_metrics)} models successfully!")
        
        # Statistics
        total_qa_rounds = 0
        for metrics in all_metrics.values():
            qa_pairs_count = len([k for k in metrics.keys() if k.startswith("QA_Pair_")])
            total_qa_rounds = max(total_qa_rounds, qa_pairs_count)
        
        print(f"Maximum QA rounds found: {total_qa_rounds}")
        print("All models have Multi_hop evaluation.")
    else:
        print(f"\nNo files processed. Please check if '{results_directory}' directory exists and contains JSON files.")

if __name__ == "__main__":
    main()