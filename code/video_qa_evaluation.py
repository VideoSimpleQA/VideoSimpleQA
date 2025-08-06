import os
import json
import base64
import argparse
from datetime import datetime
from functools import partial
from openai import OpenAI
from multiprocessing import Pool, Manager

# Configuration - Update these values with your own API credentials
API_KEY = "your-api-key-here"  # Replace with your OpenAI API key
BASE_URL = "https://api.openai.com/v1"  # Replace with your API base URL if using a custom endpoint

# Grading template for evaluating model responses
GRADER_TEMPLATE = """
Your job is to look at some video frames generated from the video, a question generated from the video, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What is the name of the man's child in the video?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What is the name of the man's child in the video?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What is the name of the man's child in the video?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Grade the predicted answer of the question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
    
Just return the letter "A", "B", or "C", with no text around it.
"""

# Prompt template for getting answers with confidence scores from the target model
ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
Do not generate any intermediate reasoning process. Based on the video frames, directly output a short, accurate answer to the user's question and include a confidence score (0-100) in the following JSON format:
{"answer": "Your answer here", "confidence_score": number}
Do not include any additional text or explanations outside this JSON format.
"""


def parse_arguments():
    """
    Parse command line arguments for evaluation configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Video QA Evaluation Framework')
    
    # Model configuration
    parser.add_argument('--target-model', '-tm', type=str, required=True,
                        help='Model to be evaluated (e.g., gpt-4-vision-preview)')
    parser.add_argument('--grader-model', '-gm', type=str, required=True,
                        help='Model used for grading responses (e.g., gpt-4)')
    
    # Data configuration
    parser.add_argument('--frame-num', '-fn', type=int, default=32,
                        help='Number of frames to extract from each video (default: 32)')
    parser.add_argument('--frames-path', '-fp', type=str, default=None,
                        help='Path to video frames directory (default: ./frames_{FRAME_NUM}/)')
    parser.add_argument('--data-file', '-df', type=str, default='VideoSimpleQA.json',
                        help='Path to the evaluation dataset (default: VideoSimpleQA.json)')
    
    # Processing configuration
    parser.add_argument('--max-retry-times', '-mr', type=int, default=10,
                        help='Maximum number of retries for API calls (default: 10)')
    parser.add_argument('--pool-processes', '-pp', type=int, default=20,
                        help='Number of parallel processes for evaluation (default: 20)')
    
    return parser.parse_args()


def clean_json_response(response):
    """
    Clean and parse JSON response from model output.
    
    Args:
        response (str): Raw response string from the model
        
    Returns:
        dict or None: Parsed JSON object or None if parsing fails
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            return json.loads(json_str)
        return None
    except Exception:
        return None


def save_metrics(metrics_data, output_file):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics_data (dict): Dictionary containing evaluation metrics
        output_file (str): Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)


def save_results(results, output_file):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results (list): List of evaluation results
        output_file (str): Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


def calculate_metrics(results):
    """
    Calculate evaluation metrics from grading results.
    
    Args:
        results (list): List of results with 'grade' field
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    total_samples = len(results)
    if total_samples == 0:
        return {
            "is_correct": 0,
            "is_incorrect": 0,
            "is_not_attempted": 0,
            "is_given_attempted": 0,
            "accuracy_given_attempted": 0,
            "f1_score": 0
        }
    
    metrics = {
        "is_correct": sum(1 for x in results if x["grade"] == "A") / total_samples,
        "is_incorrect": sum(1 for x in results if x["grade"] == "B") / total_samples,
        "is_not_attempted": sum(1 for x in results if x["grade"] == "C") / total_samples
    }
    
    metrics["is_given_attempted"] = metrics["is_correct"] + metrics["is_incorrect"]
    
    metrics["accuracy_given_attempted"] = (
        metrics["is_correct"] / metrics["is_given_attempted"]
        if metrics["is_given_attempted"] > 0
        else 0
    )
    
    metrics["f1_score"] = (
        2 * metrics["accuracy_given_attempted"] * metrics["is_correct"]
        / (metrics["accuracy_given_attempted"] + metrics["is_correct"])
        if (metrics["accuracy_given_attempted"] + metrics["is_correct"]) > 0
        else 0
    )
    
    return metrics


def call_single_model(client, messages, model, item_id, max_retry_times):
    """
    Make a single API call to the specified model with retry logic.
    
    Args:
        client: OpenAI client instance
        messages (list): List of messages for the API call
        model (str): Model name to use
        item_id (str): ID of the item being processed (for error logging)
        max_retry_times (int): Maximum number of retries
        
    Returns:
        str or None: Model response or None if all retries failed
    """
    retry_times = 0
    while retry_times < max_retry_times:
        try:
            if model == "gpt-4-vision-preview":
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096
                )
            else:    
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
            return completion.choices[0].message.content
        except Exception as e:
            retry_times += 1
            if retry_times == max_retry_times:
                with open(f'error_log_{model.replace("/", "_")}.txt', 'a') as f:
                    f.write(f"Error processing item {item_id} with model {model}: {str(e)}\n")
                return None
            print(f"Retrying model {model} after error: {str(e)}")
            import time
            time.sleep(10)
            continue


def evaluate_single_model(data_item, frames, target_model, grader_model, api_key, base_url, max_retry_times):
    """
    Evaluate a single data item using the target model and grade the response.
    
    Args:
        data_item (dict): Dictionary containing question and answer data
        frames (list): List of encoded video frames
        target_model (str): Model to be evaluated
        grader_model (str): Model used for grading
        api_key (str): API key
        base_url (str): API base URL
        max_retry_times (int): Maximum number of retries
        
    Returns:
        dict or None: Evaluation result or None if evaluation failed
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Step 1: Get model answer
    answer_messages = [{"role": "system", "content": ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE}]
    answer_messages.append({"role": "user", "content": "Here are the video frames:"})
    answer_messages.append({"role": "user", "content": frames})
    answer_messages.append({"role": "user", "content": f"This is the question: {data_item['Multi_hop_Question']}"})
    
    response = call_single_model(client, answer_messages, target_model, data_item["date"], max_retry_times)
    if response is None:
        return None
        
    # Parse answer and confidence score
    parsed_response = clean_json_response(response)
    if parsed_response is None:
        answer = response  # Use raw response if parsing fails
        confidence = None
    else:
        answer = parsed_response.get("answer", response)
        confidence = parsed_response.get("confidence_score")
    
    # Step 2: Grade the answer
    grade_messages = [{"role": "system", "content": GRADER_TEMPLATE}]
    grade_messages.append({"role": "user", "content": "Here are the video frames:"})
    grade_messages.append({"role": "user", "content": frames})
    grade_messages.append({"role": "user", "content": f"Question: {data_item['Multi_hop_Question']}"})
    grade_messages.append({"role": "user", "content": f"Gold target: {data_item['Multi_hop_Answer']}"})
    grade_messages.append({"role": "user", "content": f"Predicted answer: {answer}"})
    
    grade = call_single_model(client, grade_messages, grader_model, data_item["date"], max_retry_times)
    
    # Create result dictionary with original data plus new fields
    result = {
        **data_item,  # Expand all original data
        "model_answer": answer,
        "confidence": confidence,
        "grade": grade
    }
    
    return result


def encode_image(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_frames(frames_path):
    """
    Process video frames from a directory and encode them for API usage.
    
    Args:
        frames_path (str): Path to the directory containing video frames
        
    Returns:
        list: List of encoded frame objects for API consumption
    """
    frame_path_list = []
    for filename in os.listdir(frames_path):
        full_path = os.path.join(frames_path, filename)
        if os.path.isfile(full_path):
            frame_path_list.append(full_path)
    
    frame_path_list = sorted(frame_path_list)
    N = len(frame_path_list)
    
    # Encode all frames to base64
    base64_image_list = []
    for idx, name in enumerate(frame_path_list):
        base64_image_list.append(encode_image(name))
    
    # Create frame objects for API
    frames = []
    for idx in range(N):
        frames.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image_list[idx]}",
                "detail": "low" 
            },
        })
    
    return frames


def process_single_data(data_item, args, shared_results, shared_metrics, results_lock, 
                       metrics_lock, file_lock, counter_lock, counter, total):
    """
    Process a single data item in a multiprocessing context.
    
    Args:
        data_item (dict): Single data item to process
        args: Command line arguments
        shared_results: Shared list for storing results
        shared_metrics: Shared list for storing metrics
        results_lock: Lock for results access
        metrics_lock: Lock for metrics access
        file_lock: Lock for file operations
        counter_lock: Lock for counter access
        counter: Shared counter for progress tracking
        total (int): Total number of items to process
    """
    try:
        frames_path = os.path.join(args.frames_path, data_item["date"])
        frames = process_frames(frames_path)
        
        result = evaluate_single_model(
            data_item, frames, args.target_model, args.grader_model, 
            args.api_key, args.base_url, args.max_retry_times
        )
        
        if result is not None:
            # Save result to shared list and file
            with results_lock:
                shared_results.append(result)
                all_results = list(shared_results)
                save_results(all_results, f"evaluation_results_{args.target_model.replace('/', '_')}_{args.frame_num}frames.json")
            
            # Update metrics
            with metrics_lock:
                shared_metrics.append({
                    "grade": result["grade"]
                })
        
        print(f"Processed ID: {data_item['date']}")
        
        # Update progress counter
        with counter_lock:
            counter.value += 1
            print(f"\rProcessed: {counter.value}/{total} videos")
        
    except Exception as e:
        print(f"Error processing video {data_item['date']}: {str(e)}")
        
        # Update counter even on error
        with counter_lock:
            counter.value += 1
            print(f"\rProcessed: {counter.value}/{total} videos")
        
        # Log error to file
        with file_lock:
            with open(f'error_log_{args.target_model.replace("/", "_")}.txt', 'a') as f:
                f.write(f"Error processing video {data_item['date']}: {str(e)}\n")


def load_test_data(json_file):
    """
    Load test data from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing test data
        
    Returns:
        list: List of test data items
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    """
    Main function to run the video QA evaluation framework.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up API key (priority: command line > environment variable > hardcoded)
    if args.api_key:
        api_key = args.api_key
    elif os.getenv('OPENAI_API_KEY'):
        api_key = os.getenv('OPENAI_API_KEY')
    else:
        api_key = API_KEY
    
    # Update args with resolved API key
    args.api_key = api_key
    
    # Set frames path if not provided
    if args.frames_path is None:
        args.frames_path = f"./frames_{args.frame_num}/"
    
    print(f"Processing with model: {args.target_model}")
    print(f"Grading with model: {args.grader_model}")
    print(f"Frame number: {args.frame_num}")
    print(f"Frames path: {args.frames_path}")
    print(f"Data file: {args.data_file}")
    print(f"Pool processes: {args.pool_processes}")
    
    # Initialize error log
    error_log_file = f'error_log_{args.target_model.replace("/", "_")}.txt'
    with open(error_log_file, 'w') as f:
        f.write(f"=== Error Log Started at {datetime.now()} ===\n")
    
    # Define output files
    output_file = f"evaluation_results_{args.target_model.replace('/', '_')}_{args.frame_num}frames.json"
    metrics_output_file = f"model_metrics_{args.target_model.replace('/', '_')}.json"
    
    # Load test data
    test_data = load_test_data(args.data_file)
    total_videos = len(test_data)
    print(f"Total videos to process: {total_videos}")
    
    # Set up multiprocessing with shared data structures
    with Manager() as manager:
        shared_results = manager.list()
        shared_metrics = manager.list()
        counter = manager.Value('i', 0)

        # Create locks for thread-safe operations
        results_lock = manager.Lock()
        metrics_lock = manager.Lock()
        file_lock = manager.Lock()
        counter_lock = manager.Lock()
        
        # Process data in parallel
        with Pool(processes=args.pool_processes) as pool:
            process_func = partial(
                process_single_data,
                args=args,
                shared_results=shared_results,
                shared_metrics=shared_metrics,
                results_lock=results_lock,
                metrics_lock=metrics_lock,
                file_lock=file_lock,
                counter_lock=counter_lock,
                counter=counter,
                total=total_videos
            )
            
            pool.map(process_func, test_data)
        
        # Convert shared data to regular lists
        all_results = list(shared_results)
        all_metrics = list(shared_metrics)

    
    # Save final results
    save_results(all_results, output_file)
    
    print(f"Processing complete for model: {args.target_model}")
    print(f"Results saved to: {output_file}")
    # print(f"Metrics saved to: {metrics_output_file}")


if __name__ == "__main__":
    main()