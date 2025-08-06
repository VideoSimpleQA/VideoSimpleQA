import os
import json
import base64
import argparse
from datetime import datetime
from functools import partial
from openai import OpenAI
from multiprocessing import Pool, Manager

# Initialize OpenAI client - Replace with your own API configuration
API_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"  # Replace with your API endpoint

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

# Prompt template for getting answers with confidence scores
ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
Do not generate any intermediate reasoning process. Based on the video frames, directly output a short, accurate answer to the user's question and include a confidence score (0-100) in the following JSON format:
{"answer": "Your answer here", "confidence_score": number}
Do not include any additional text or explanations outside this JSON format.
"""

def parse_arguments():
    """
    Parse command line arguments for configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Video QA Evaluation Script')
    
    parser.add_argument('--frame-num', type=int, default=32,
                        help='Number of frames to use (default: 32)')
    
    parser.add_argument('--frames-path', type=str, default=None,
                        help='Path to frames directory (default: ./frames_{frame_num}/)')
    
    parser.add_argument('--target-model', type=str, required=True,
                        help='Model to be evaluated (required)')
    
    parser.add_argument('--grader-model', type=str, required=True,
                        help='Model used for grading responses (required)')
    
    parser.add_argument('--output-file', type=str, default='./results/evaluation_results.json',
                        help='Path to output file (default: ./results/evaluation_results.json)')
    
    parser.add_argument('--data-file', type=str, default='./data/test_data.json',
                        help='Path to test data file (default: ./data/test_data.json)')
    
    parser.add_argument('--processes', type=int, default=20,
                        help='Number of parallel processes (default: 20)')
    
    args = parser.parse_args()
    
    # Set frames_path if not provided
    if args.frames_path is None:
        args.frames_path = f"./frames_{args.frame_num}/"
    
    return args

def encode_image(image_path):
    """
    Encode image file to base64 string for API consumption.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_frames(frames_path):
    """
    Process video frames from a directory and encode them for API use.
    
    Args:
        frames_path (str): Path to directory containing video frames
        
    Returns:
        list: List of frame objects formatted for OpenAI API
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
    
    # Format frames for API
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

def clean_json_response(response):
    """
    Clean and parse JSON response from model output.
    
    Args:
        response (str): Raw response string from model
        
    Returns:
        dict or None: Parsed JSON object, or None if parsing fails
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

def save_results(results, output_file):
    """
    Save evaluation results to JSON file.
    
    Args:
        results (list): List of evaluation results
        output_file (str): Path to output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

def call_single_model(client, messages, model, id, target_model):
    """
    Make a single API call to the specified model with retry logic.
    
    Args:
        client: OpenAI client instance
        messages (list): List of messages for the API call
        model (str): Model name to use
        id (str): Identifier for logging purposes
        target_model (str): Target model name for error logging
        
    Returns:
        str or None: Model response content, or None if all retries failed
    """
    max_retry_times = 10
    retry_times = 0
    
    while retry_times < max_retry_times:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            retry_times += 1
            if retry_times == max_retry_times:
                # Log error if all retries failed
                with open(f'error_log_{target_model}.txt', 'a') as f:
                    f.write(f"Error processing question {id} by using {model}: {str(e)}\n")
                return None
            print(f"Retrying model {model} after error: {str(e)}")
            import time
            time.sleep(10)
            continue

def answer_and_grade_qa(client, question, gold_answer, qa_id, data_id, frames, target_model, grader_model):
    """
    Get model answer for a question and grade it against the gold answer.
    
    Args:
        client: OpenAI client instance
        question (str): Question to ask
        gold_answer (str): Ground truth answer
        qa_id (str): QA pair identifier
        data_id (str): Data item identifier
        frames (list): Video frames for context
        target_model (str): Target model name
        grader_model (str): Grader model name
        
    Returns:
        tuple: (answer, confidence_score, grade)
    """
    # Step 1: Get model answer
    answer_messages = [
        {"role": "system", "content": ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE},
        {"role": "user", "content": "Here are the video frames:"},
        {"role": "user", "content": frames},
        {"role": "user", "content": f"This is the question: {question}"}
    ]
    
    response = call_single_model(client, answer_messages, target_model, f"{data_id}_{qa_id}", target_model)
    
    if response is None:
        return None, None, None
        
    # Parse answer and confidence score
    parsed_response = clean_json_response(response)
    if parsed_response is None:
        answer = response  # Use raw response if parsing fails
        confidence = None
    else:
        answer = parsed_response.get("answer", response)
        confidence = parsed_response.get("confidence_score")
    
    # Step 2: Grade the answer
    grade_messages = [
        {"role": "system", "content": GRADER_TEMPLATE},
        {"role": "user", "content": "Here are the video frames:"},
        {"role": "user", "content": frames},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "user", "content": f"Gold target: {gold_answer}"},
        {"role": "user", "content": f"Predicted answer: {answer}"}
    ]
    
    grade = call_single_model(client, grade_messages, grader_model, f"{data_id}_{qa_id}_grade", target_model)
    
    return answer, confidence, grade

def evaluate_single_data_item(data_item, args):
    """
    Evaluate a single data item containing multiple QA pairs.
    
    Args:
        data_item (dict): Data item containing questions and answers
        args: Parsed command line arguments
        
    Returns:
        dict: Evaluation results for the data item
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    # Process video frames for this data item
    frames_path = os.path.join(args.frames_path, data_item["date"])
    frames = process_frames(frames_path)
    
    # Initialize result structure
    result = {
        "date": data_item["date"],
        "Multi_hop_Question": data_item["Multi_hop_Question"],
        "Multi_hop_Answer": data_item["Multi_hop_Answer"],
        "Multi_hop_model_answer": data_item["Multi_hop_model_answer"],
        "Multi_hop_confidence": data_item["Multi_hop_confidence"],
        "Multi_hop_grade": data_item["Multi_hop_grade"]
    }
    
    # Process 4 QA pairs
    qa_pairs = [
        ("QA_Pair_1_Question", "QA_Pair_1_Answer"),
        ("QA_Pair_2_Question", "QA_Pair_2_Answer"),
        ("QA_Pair_3_Question", "QA_Pair_3_Answer"),
        ("QA_Pair_4_Question", "QA_Pair_4_Answer")
    ]
    
    for i, (q_key, a_key) in enumerate(qa_pairs, 1):
        if q_key in data_item and a_key in data_item:
            question = data_item[q_key]
            gold_answer = data_item[a_key]
            
            # Preserve original question and answer
            result[q_key] = question
            result[a_key] = gold_answer
            
            # Get model answer and evaluation
            answer, confidence, grade = answer_and_grade_qa(
                client, question, gold_answer, f"qa{i}", data_item["date"], frames,
                args.target_model, args.grader_model
            )
            
            # Save model response, confidence score, and grade
            result[f"QA_Pair_{i}_model_answer"] = answer
            result[f"QA_Pair_{i}_confidence"] = confidence
            result[f"QA_Pair_{i}_grade"] = grade
    
    return result

def process_single_data(data_item, shared_results, results_lock, counter_lock, counter, total, args):
    """
    Process a single data item in multiprocessing context.
    
    Args:
        data_item (dict): Data item to process
        shared_results: Shared list for storing results
        results_lock: Lock for accessing shared results
        counter_lock: Lock for accessing counter
        counter: Shared counter for progress tracking
        total (int): Total number of items to process
        args: Parsed command line arguments
    """
    try:
        result = evaluate_single_data_item(data_item, args)
        
        if result is not None:
            # Save results with thread safety
            with results_lock:
                shared_results.append(result)
                all_results = list(shared_results)
                save_results(all_results, args.output_file)
        
        print(f"Processed ID: {data_item['date']}")
        
        # Update progress counter
        with counter_lock:
            counter.value += 1
            print(f"\rProcessed: {counter.value}/{total} items")
        
    except Exception as e:
        print(f"Error processing item {data_item['date']}: {str(e)}")
        
        # Update counter even on error
        with counter_lock:
            counter.value += 1
            print(f"\rProcessed: {counter.value}/{total} items")
        
        # Log error
        with open(f'error_log_{args.target_model}.txt', 'a') as f:
            f.write(f"Error processing item {data_item['date']}: {str(e)}\n")

def load_test_data(json_file):
    """
    Load test data from JSON file.
    
    Args:
        json_file (str): Path to JSON data file
        
    Returns:
        list: List of test data items
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    """
    Main function to run the evaluation pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Frame number: {args.frame_num}")
    print(f"  Frames path: {args.frames_path}")
    print(f"  Target model: {args.target_model}")
    print(f"  Grader model: {args.grader_model}")
    print(f"  Output file: {args.output_file}")
    print(f"  Data file: {args.data_file}")
    print(f"  Processes: {args.processes}")
    
    # Initialize error log
    with open(f'error_log_{args.target_model}.txt', 'w') as f:
        f.write(f"=== Error Log Started at {datetime.now()} ===\n")
    
    # Load test data
    test_data = load_test_data(args.data_file)
    total_items = len(test_data)
    print(f"Total items to process: {total_items}")
    
    # Process data using multiprocessing
    with Manager() as manager:
        shared_results = manager.list()
        counter = manager.Value('i', 0)
        results_lock = manager.Lock()
        counter_lock = manager.Lock()
        
        # Create partial function with shared variables
        process_func = partial(
            process_single_data,
            shared_results=shared_results,
            results_lock=results_lock,
            counter_lock=counter_lock,
            counter=counter,
            total=total_items,
            args=args
        )
        
        # Use multiprocessing pool for parallel processing
        with Pool(processes=args.processes) as pool:
            pool.map(process_func, test_data)
        
        # Convert shared results to regular list
        all_results = list(shared_results)
    
    # Save final results
    save_results(all_results, args.output_file)
    print(f"Processing complete for model: {args.target_model}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()