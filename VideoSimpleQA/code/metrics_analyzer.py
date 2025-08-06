import os
import json
import glob
from pathlib import Path
from collections import defaultdict

def calculate_metrics(results):
    """
    Calculate evaluation metrics, convert to percentages and round to one decimal place.
    
    Args:
        results (list): List of evaluation results with 'grade' field
        
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
    
    # Calculate raw metrics based on grades
    # Grade A: Correct, Grade B: Incorrect, Grade C: Not attempted
    correct_count = sum(1 for x in results if x.get("grade") == "A")
    incorrect_count = sum(1 for x in results if x.get("grade") == "B")
    not_attempted_count = sum(1 for x in results if x.get("grade") == "C")
    
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

def calculate_category_fscore(results, category_mapping):
    """
    Calculate F-score for each category.
    
    Args:
        results (list): List of evaluation results
        category_mapping (dict): Mapping from unique keys to categories
        
    Returns:
        dict: F-scores for each category
    """
    # Group results by category
    category_results = defaultdict(list)
    
    for result in results:
        # Find corresponding category using date and question
        key = create_unique_key(result)
        category = category_mapping.get(key)
        if category:
            category_results[category].append(result)
        else:
            # If category not found, put in unknown category
            category_results["Unknown"].append(result)
    
    # Calculate F-score for each category
    category_fscores = {}
    for category, cat_results in category_results.items():
        metrics = calculate_metrics(cat_results)
        category_fscores[category] = metrics["f1_score"]
    
    return category_fscores

def calculate_round_fscore(results, round_mapping):
    """
    Calculate F-score for each round.
    
    Args:
        results (list): List of evaluation results
        round_mapping (dict): Mapping from unique keys to rounds
        
    Returns:
        dict: F-scores for each round
    """
    # Group results by round
    round_results = defaultdict(list)
    
    for result in results:
        # Find corresponding round using date and question
        key = create_unique_key(result)
        round_info = round_mapping.get(key)
        if round_info:
            round_results[round_info].append(result)
        else:
            # If round not found, put in unknown round
            round_results["Unknown"].append(result)
    
    # Calculate F-score for each round
    round_fscores = {}
    for round_name, round_res in round_results.items():
        metrics = calculate_metrics(round_res)
        round_fscores[round_name] = metrics["f1_score"]
    
    return round_fscores

def create_unique_key(item):
    """
    Create unique identifier based on date and Multi_hop_Question.
    
    Args:
        item (dict): Data item containing date and question
        
    Returns:
        str: Unique key string
    """
    date = item.get("date", "")
    question = item.get("Multi_hop_Question", "")
    return f"{date}|{question}"

def load_category_mapping(category_file_path):
    """
    Load category mapping from classification file.
    
    Args:
        category_file_path (str): Path to category classification file
        
    Returns:
        dict: Mapping from unique keys to categories
    """
    try:
        with open(category_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        category_mapping = {}
        for item in data:
            key = create_unique_key(item)
            category = item.get("category", "Unknown")
            category_mapping[key] = category
        
        print(f"Loaded category mapping for {len(category_mapping)} items")
        print(f"Categories found: {set(category_mapping.values())}")
        return category_mapping
    except Exception as e:
        print(f"Error loading category file {category_file_path}: {str(e)}")
        return {}

def load_round_mapping(round_file_path):
    """
    Load round mapping from round number file.
    
    Args:
        round_file_path (str): Path to round number file
        
    Returns:
        dict: Mapping from unique keys to rounds
    """
    try:
        with open(round_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        round_mapping = {}
        for item in data:
            key = create_unique_key(item)
            round_info = item.get("round", "Unknown")
            round_mapping[key] = f"Round_{round_info}"
        
        print(f"Loaded round mapping for {len(round_mapping)} items")
        print(f"Rounds found: {set(round_mapping.values())}")
        return round_mapping
    except Exception as e:
        print(f"Error loading round file {round_file_path}: {str(e)}")
        return {}

def load_evaluation_results(file_path):
    """
    Load evaluation results from JSON file.
    
    Args:
        file_path (str): Path to evaluation results file
        
    Returns:
        list or None: Evaluation results data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def extract_model_name(filename):
    """
    Extract model name from filename.
    
    Example: evaluation_results_gemini-2.5-pro-06-17_30frames.json -> gemini-2.5-pro-06-17
    
    Args:
        filename (str): Input filename
        
    Returns:
        str: Extracted model name
    """
    if filename.startswith("evaluation_results_"):
        # Remove prefix
        name_part = filename[len("evaluation_results_"):]
        # Find the last underscore to remove frame count and extension
        last_underscore = name_part.rfind("_")
        if last_underscore != -1:
            return name_part[:last_underscore]
    return filename

def print_category_summary(category_fscores):
    """
    Print category-wise F-score summary table.
    
    Args:
        category_fscores (dict): F-scores for each category
    """
    if not category_fscores:
        print("No category F-scores to display")
        return
    
    print("\n" + "="*50)
    print("CATEGORY-WISE F-SCORES SUMMARY")
    print("="*50)
    print(f"{'Category':<20} {'F1 Score (%)':<12}")
    print("-"*50)
    
    # Sort by category name
    for category in sorted(category_fscores.keys()):
        f1_score = category_fscores[category]
        print(f"{category:<20} {f1_score:<12}")

def print_round_summary(round_fscores):
    """
    Print round-wise F-score summary table.
    
    Args:
        round_fscores (dict): F-scores for each round
    """
    if not round_fscores:
        print("No round F-scores to display")
        return
    
    print("\n" + "="*50)
    print("ROUND-WISE F-SCORES SUMMARY")
    print("="*50)
    print(f"{'Round':<20} {'F1 Score (%)':<12}")
    print("-"*50)
    
    # Sort by round name
    for round_name in sorted(round_fscores.keys()):
        f1_score = round_fscores[round_name]
        print(f"{round_name:<20} {f1_score:<12}")

def process_all_evaluation_files(eval_results_dir="evaluation_results", 
                                category_file="category_mapping.json", 
                                round_file="round_mapping.json"):
    """
    Process all evaluation result files and calculate metrics.
    
    Args:
        eval_results_dir (str): Directory containing evaluation result files
        category_file (str): Path to category mapping file
        round_file (str): Path to round mapping file
        
    Returns:
        dict: All calculated metrics for all models
    """
    if not os.path.exists(eval_results_dir):
        print(f"Directory {eval_results_dir} does not exist!")
        return
    
    # Load category mapping
    category_mapping = load_category_mapping(category_file) if category_file and os.path.exists(category_file) else {}
    if not category_mapping:
        print("Warning: No category mapping loaded. Category-wise metrics will not be available.")
    
    # Load round mapping
    round_mapping = load_round_mapping(round_file) if round_file and os.path.exists(round_file) else {}
    if not round_mapping:
        print("Warning: No round mapping loaded. Round-wise metrics will not be available.")
    
    # Find all evaluation result files
    pattern = os.path.join(eval_results_dir, "evaluation_results_*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No evaluation result files found in {eval_results_dir}")
        return
    
    print(f"Found {len(result_files)} evaluation result files:")
    for file in result_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    all_metrics = {}
    all_category_fscores = {}
    all_round_fscores = {}
    
    for file_path in result_files:
        filename = os.path.basename(file_path)
        model_name = extract_model_name(filename)
        
        print(f"Processing {filename}...")
        
        # Load evaluation results
        results = load_evaluation_results(file_path)
        if results is None:
            continue
        
        # Calculate overall metrics
        metrics = calculate_metrics(results)
        
        # Calculate category-wise F-scores
        category_fscores = {}
        if category_mapping:
            category_fscores = calculate_category_fscore(results, category_mapping)
            all_category_fscores[model_name] = category_fscores
        
        # Calculate round-wise F-scores
        round_fscores = {}
        if round_mapping:
            round_fscores = calculate_round_fscore(results, round_mapping)
            all_round_fscores[model_name] = round_fscores
        
        # Combine overall metrics with category and round F-scores
        combined_data = metrics.copy()  # Keep all overall metrics
        if category_fscores:
            # Add prefix to distinguish categories from rounds
            for cat, score in category_fscores.items():
                combined_data[f"category_{cat}"] = score
        if round_fscores:
            # Add prefix to distinguish rounds from categories
            for round_name, score in round_fscores.items():
                combined_data[f"round_{round_name}"] = score
        
        all_metrics[model_name] = combined_data
        
        # Print individual model metrics
        print(f"Metrics for {model_name}:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Correct rate: {metrics['is_correct']}%")
        print(f"  Incorrect rate: {metrics['is_incorrect']}%")
        print(f"  Not attempted rate: {metrics['is_not_attempted']}%")
        print(f"  Attempted rate: {metrics['is_given_attempted']}%")
        print(f"  Accuracy (given attempted): {metrics['accuracy_given_attempted']}%")
        print(f"  Overall F1 score: {metrics['f1_score']}%")
        
        # Print category-wise F-scores
        if category_fscores:
            print_category_summary(category_fscores)
        
        # Print round-wise F-scores
        if round_fscores:
            print_round_summary(round_fscores)
        
        print()
    
    # Save all metrics to file
    output_file = os.path.join(eval_results_dir, "all_model_metrics.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"All metrics saved to: {output_file}")
    
    # Create overall metrics summary table
    print("\n" + "="*80)
    print("OVERALL SUMMARY TABLE (All values in %)")
    print("="*80)
    print(f"{'Model':<30} {'Total':<8} {'Correct':<8} {'Incorrect':<10} {'Not Att.':<10} {'Attempted':<10} {'Acc.(Att.)':<10} {'F1':<8}")
    print("-"*80)
    
    for model_name, model_data in all_metrics.items():
        if 'total_samples' in model_data:  # Ensure overall metrics data exists
            print(f"{model_name:<30} {model_data['total_samples']:<8} {model_data['is_correct']:<8} {model_data['is_incorrect']:<10} {model_data['is_not_attempted']:<10} {model_data['is_given_attempted']:<10} {model_data['accuracy_given_attempted']:<10} {model_data['f1_score']:<8}")
    
    # Create category-wise F-score summary table
    if all_category_fscores:
        print("\n" + "="*80)
        print("CATEGORY F-SCORE SUMMARY TABLE")
        print("="*80)
        
        # Get all categories
        all_categories = set()
        for category_fscores in all_category_fscores.values():
            all_categories.update(category_fscores.keys())
        all_categories = sorted(all_categories)
        
        # Print header
        header = f"{'Model':<30}"
        for category in all_categories:
            header += f" {category:<12}"
        print(header)
        print("-" * len(header))
        
        # Print each model's category-wise F-scores
        for model_name, category_fscores in all_category_fscores.items():
            row = f"{model_name:<30}"
            for category in all_categories:
                fscore = category_fscores.get(category, 0.0)
                row += f" {fscore:<12}"
            print(row)
    
    # Create round-wise F-score summary table
    if all_round_fscores:
        print("\n" + "="*80)
        print("ROUND F-SCORE SUMMARY TABLE")
        print("="*80)
        
        # Get all rounds
        all_rounds = set()
        for round_fscores in all_round_fscores.values():
            all_rounds.update(round_fscores.keys())
        all_rounds = sorted(all_rounds)
        
        # Print header
        header = f"{'Model':<30}"
        for round_name in all_rounds:
            header += f" {round_name:<12}"
        print(header)
        print("-" * len(header))
        
        # Print each model's round-wise F-scores
        for model_name, round_fscores in all_round_fscores.items():
            row = f"{model_name:<30}"
            for round_name in all_rounds:
                fscore = round_fscores.get(round_name, 0.0)
                row += f" {fscore:<12}"
            print(row)
    
    return all_metrics

def process_single_file(file_path, 
                       category_file="category_mapping.json", 
                       round_file="round_mapping.json"):
    """
    Process a single evaluation result file.
    
    Args:
        file_path (str): Path to evaluation result file
        category_file (str): Path to category mapping file
        round_file (str): Path to round mapping file
        
    Returns:
        dict or None: Calculated metrics or None if error
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist!")
        return None
    
    # Load category mapping
    category_mapping = load_category_mapping(category_file) if category_file and os.path.exists(category_file) else {}
    
    # Load round mapping
    round_mapping = load_round_mapping(round_file) if round_file and os.path.exists(round_file) else {}
    
    filename = os.path.basename(file_path)
    model_name = extract_model_name(filename)
    
    print(f"Processing {filename}...")
    
    # Load evaluation results
    results = load_evaluation_results(file_path)
    if results is None:
        return None
    
    # Calculate overall metrics
    metrics = calculate_metrics(results)
    
    # Calculate category-wise F-scores
    category_fscores = {}
    if category_mapping:
        category_fscores = calculate_category_fscore(results, category_mapping)
    
    # Calculate round-wise F-scores
    round_fscores = {}
    if round_mapping:
        round_fscores = calculate_round_fscore(results, round_mapping)
    
    # Combine overall metrics with category and round F-scores
    combined_data = metrics.copy()  # Keep all overall metrics
    if category_fscores:
        for cat, score in category_fscores.items():
            combined_data[f"category_{cat}"] = score
    if round_fscores:
        for round_name, score in round_fscores.items():
            combined_data[f"round_{round_name}"] = score
    
    # Print metrics
    print(f"Metrics for {model_name}:")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Correct rate: {metrics['is_correct']}%")
    print(f"  Incorrect rate: {metrics['is_incorrect']}%")
    print(f"  Not attempted rate: {metrics['is_not_attempted']}%")
    print(f"  Attempted rate: {metrics['is_given_attempted']}%")
    print(f"  Accuracy (given attempted): {metrics['accuracy_given_attempted']}%")
    print(f"  Overall F1 score: {metrics['f1_score']}%")
    
    # Print category-wise F-scores
    if category_fscores:
        print_category_summary(category_fscores)
    
    # Print round-wise F-scores
    if round_fscores:
        print_round_summary(round_fscores)
    
    return {model_name: combined_data}

if __name__ == "__main__":
    print("Video Evaluation Metrics Calculator with Category and Round Analysis")
    print("="*70)
    
    # Check if category file exists
    category_file = "category_mapping.json"
    if not os.path.exists(category_file):
        print(f"Warning: Category file '{category_file}' not found!")
        print("Category-wise analysis will be skipped.")
        category_file = None
    
    # Check if round file exists
    round_file = "round_mapping.json"
    if not os.path.exists(round_file):
        print(f"Warning: Round file '{round_file}' not found!")
        print("Round-wise analysis will be skipped.")
        round_file = None
    
    # Process all evaluation result files
    print("Processing all evaluation files in 'evaluation_results' directory...")
    all_metrics = process_all_evaluation_files(category_file=category_file, round_file=round_file)
    
    if all_metrics:
        print(f"\nProcessed {len(all_metrics)} models successfully!")
        
        # Calculate how many models have category analysis
        models_with_categories = 0
        all_categories = set()
        models_with_rounds = 0
        all_rounds = set()
        
        for model_data in all_metrics.values():
            # Check category data
            categories = [k for k in model_data.keys() if k.startswith("category_")]
            if categories:
                models_with_categories += 1
                all_categories.update([k.replace("category_", "") for k in categories])
            
            # Check round data
            rounds = [k for k in model_data.keys() if k.startswith("round_")]
            if rounds:
                models_with_rounds += 1
                all_rounds.update([k.replace("round_", "") for k in rounds])
        
        if models_with_categories > 0:
            print(f"Category F-score analysis completed for {len(all_categories)} categories across {models_with_categories} models.")
        
        if models_with_rounds > 0:
            print(f"Round F-score analysis completed for {len(all_rounds)} rounds across {models_with_rounds} models.")
    else:
        print("\nNo files processed. Please check if 'evaluation_results' directory exists and contains evaluation result files.")