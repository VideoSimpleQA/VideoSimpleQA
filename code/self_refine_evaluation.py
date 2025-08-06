import os
import json
import base64
import asyncio
from tqdm import tqdm
from datetime import datetime
from openai import AsyncOpenAI


# Configuration - Replace with your own API credentials
API_KEY = "your_openai_api_key_here"
BASE_URL = "https://api.openai.com/v1"  # Replace with your API endpoint

# Template for grading answers against gold standard
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

# Prompt for generating initial answers
INITIAL_ANSWER_PROMPT = """
Based on the video frames, provide a concise and accurate answer to the user's question.
Return your answer in the following JSON format:
{"answer": "Your answer here"}
Do not include any additional text or explanations outside this JSON format.
"""

# Prompt for generating feedback on answers
FEEDBACK_PROMPT = """
You are an expert evaluator. Review the following answer to the question based on the video frames.
Provide specific, actionable feedback on how to improve the answer. Focus on:
1. Factual accuracy
2. Completeness of information
3. Clarity and conciseness

Return your feedback in the following JSON format:
{"feedback": "Your detailed feedback here"}
Do not include any additional text or explanations outside this JSON format.
"""

# Prompt for refining answers based on feedback
REFINE_PROMPT = """
Based on the video frames, the question, your previous answer, and the feedback provided, generate an improved answer.
Consider the feedback carefully and address all the issues mentioned.

Return your improved, short and accurate answer in the following JSON format:
{"answer": "Your improved answer here"}
Do not include any additional text or explanations outside this JSON format.
"""

# Model configuration: maps model names to their maximum supported frame counts and frame directories
MODEL_FRAMES_CONFIG = {
    "gpt-4o-0513": {"frames_dir": "frames_30/", "max_frames": 50},
}


def clean_json_response(response):
    """
    Clean and parse JSON response from model output.
    
    Args:
        response (str): Raw response from the model
        
    Returns:
        dict: Parsed JSON with answer field, or empty answer if parsing fails
    """
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end != 0:
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except:
            # If parsing fails, try to extract answer directly
            if "answer" in response:
                try:
                    match = response.split('"answer": "')[1].split('"')[0]
                    return {"answer": match}
                except:
                    return {"answer": ""}
            else:
                return {"answer": ""}  
    return {"answer": ""}


def encode_image(image_path):
    """
    Encode image to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_frames(frames_path):
    """
    Process video frames from a directory and convert to base64 format for API calls.
    
    Args:
        frames_path (str): Path to directory containing video frames
        
    Returns:
        list: List of formatted frame objects for API consumption
    """
    frame_path_list = []
    for filename in os.listdir(frames_path):
        full_path = os.path.join(frames_path, filename)
        if os.path.isfile(full_path):
            frame_path_list.append(full_path)
    
    # Sort frames to maintain temporal order
    frame_path_list = sorted(frame_path_list)
    N = len(frame_path_list)
    
    # Convert frames to base64
    base64_image_list = []
    for idx, name in enumerate(frame_path_list):
        base64_image_list.append(encode_image(name))
    
    # Format frames for API
    frames = []
    for idx in range(N):
        frames.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_list[idx]}",
                    "detail": "low" 
                },
            }
        )
    return frames


def load_test_data(json_file):
    """
    Load test data from JSON file.
    
    Args:
        json_file (str): Path to JSON file containing test data
        
    Returns:
        list: List of test data items
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


async def call_single_model(client, messages, model):
    """
    Make API call to a single model with retry logic.
    
    Args:
        client: AsyncOpenAI client instance
        messages (list): List of message objects for the API call
        model (str): Model name to use
        
    Returns:
        Completion object or None if all retries failed
    """
    max_retry_times = 10
    retry_times = 0
    
    while retry_times < max_retry_times:
        try:
            if model == "gpt-4-vision-preview":
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=1.0
                )
            else:    
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1.0
                )
            return completion
        except Exception as e:
            retry_times += 1
            if retry_times == max_retry_times:
                with open('error_log_self_refine.txt', 'a') as f:
                    f.write(f"Failed to call model {model} after {max_retry_times} retries: {str(e)}\n")
                return None
            print(f"Retrying model {model} after error: {str(e)}")
            await asyncio.sleep(10)
            continue


async def grade_answer(client, data_item, answer, frames):
    """
    Grade an answer against the gold standard using o3.
    
    Args:
        client: AsyncOpenAI client instance
        data_item (dict): Test data item containing question and gold answer
        answer (str): Predicted answer to grade
        frames (list): Video frames for context
        
    Returns:
        bool: True if answer is correct, False otherwise
    """
    try:
        grade_messages = [
            {"role": "system", "content": GRADER_TEMPLATE},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames},
            {"role": "user", "content": f"Question: {data_item['Question']}"},
            {"role": "user", "content": f"Gold target: {data_item['Answer']}"},
            {"role": "user", "content": f"Predicted answer: {answer}"}
        ]
        
        response = await call_single_model(client, grade_messages, "")
        
        if response is None:
            return False
            
        grade = response.choices[0].message.content.strip()[0]
        return grade == "A"
    except Exception as e:
        with open('error_log_self_refine.txt', 'a') as f:
            f.write(f"Error grading answer: {str(e)}\n")
        return False


async def generate_initial_answer(client, model, data_item, frames):
    """
    Generate initial answer for a question based on video frames.
    
    Args:
        client: AsyncOpenAI client instance
        model (str): Model name to use
        data_item (dict): Test data item containing the question
        frames (list): Video frames for context
        
    Returns:
        str: Generated answer
    """
    try:
        messages = [
            {"role": "system", "content": INITIAL_ANSWER_PROMPT},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames},
            {"role": "user", "content": f"Question: {data_item['Question']}"}
        ]
        
        response = await call_single_model(client, messages, model)
        if response is None:
            return ""
            
        answer_json = clean_json_response(response.choices[0].message.content)
        return answer_json.get("answer", "")
    except Exception as e:
        with open('error_log_self_refine.txt', 'a') as f:
            f.write(f"Error generating initial answer with {model}: {str(e)}\n")
        return ""


async def generate_feedback(client, model, data_item, answer, frames):
    """
    Generate feedback for an answer to help improve it.
    
    Args:
        client: AsyncOpenAI client instance
        model (str): Model name to use
        data_item (dict): Test data item containing the question
        answer (str): Answer to provide feedback on
        frames (list): Video frames for context
        
    Returns:
        str: Generated feedback
    """
    try:
        messages = [
            {"role": "system", "content": FEEDBACK_PROMPT},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames},
            {"role": "user", "content": f"Question: {data_item['Question']}"},
            {"role": "user", "content": f"Answer: {answer}"}
        ]
        
        response = await call_single_model(client, messages, model)
        if response is None:
            return "Unable to provide feedback."
        
        return response.choices[0].message.content
    except Exception as e:
        with open('error_log_self_refine.txt', 'a') as f:
            f.write(f"Error generating feedback with {model}: {str(e)}\n")
        return "Unable to provide feedback."


async def refine_answer(client, model, data_item, previous_answer, feedback, frames):
    """
    Refine an answer based on provided feedback.
    
    Args:
        client: AsyncOpenAI client instance
        model (str): Model name to use
        data_item (dict): Test data item containing the question
        previous_answer (str): Previous answer to improve
        feedback (str): Feedback on the previous answer
        frames (list): Video frames for context
        
    Returns:
        str: Refined answer
    """
    try:
        messages = [
            {"role": "system", "content": REFINE_PROMPT},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames},
            {"role": "user", "content": f"Question: {data_item['Question']}"},
            {"role": "user", "content": f"Previous answer: {previous_answer}"},
            {"role": "user", "content": f"Feedback: {feedback}"}
        ]
        
        response = await call_single_model(client, messages, model)
        if response is None:
            return previous_answer
            
        refined_json = clean_json_response(response.choices[0].message.content)
        return refined_json.get("answer", previous_answer)
    except Exception as e:
        with open('error_log_self_refine.txt', 'a') as f:
            f.write(f"Error refining answer with {model}: {str(e)}\n")
        return previous_answer


async def process_single_item_with_self_refine(client, model, data_item, frames, max_iterations=3):
    """
    Process a single test item using the self-refine approach.
    
    Args:
        client: AsyncOpenAI client instance
        model (str): Model name to use
        data_item (dict): Test data item
        frames (list): Video frames for context
        max_iterations (int): Maximum number of refinement iterations
        
    Returns:
        dict: Dictionary containing initial answer, final answer, all answers, and feedbacks
    """
    try:
        # Generate initial answer
        initial_answer = await generate_initial_answer(client, model, data_item, frames)
        
        answers = [initial_answer]
        feedbacks = []
        
        # Iterative refinement
        for i in range(max_iterations):
            # Generate feedback
            feedback = await generate_feedback(client, model, data_item, answers[-1], frames)
            feedbacks.append(feedback)
            
            # Stop if feedback indicates the answer is already good
            if "good" in feedback.lower() and "no improvement" in feedback.lower():
                break
                
            # Refine answer based on feedback
            refined_answer = await refine_answer(client, model, data_item, answers[-1], feedback, frames)
            answers.append(refined_answer)
        
        return {
            "initial_answer": initial_answer,
            "final_answer": answers[-1],
            "all_answers": answers,
            "feedbacks": feedbacks
        }
    except Exception as e:
        with open('error_log_self_refine.txt', 'a') as f:
            f.write(f"Error in self-refine process with {model}: {str(e)}\n")
        return {
            "initial_answer": "",
            "final_answer": "",
            "all_answers": [],
            "feedbacks": []
        }


def save_intermediate_results(model_results, filename="self_refine_intermediate_results.json"):
    """
    Save intermediate results to JSON file for recovery purposes.
    
    Args:
        model_results (dict): Dictionary containing model results
        filename (str): Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=4, ensure_ascii=False)
        print(f"Intermediate results successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving intermediate results: {str(e)}")
        with open('error_log_self_refine.txt', 'a') as log:
            log.write(f"[{datetime.now()}] Save Intermediate Results Error: {str(e)}\n")


async def analyze_test_time_compute_self_refine(client, models, test_data, max_iterations=3):
    """
    Analyze test-time compute effectiveness using self-refine method.
    
    Args:
        client: AsyncOpenAI client instance
        models (list): List of model names to evaluate
        test_data (list): Test dataset
        max_iterations (int): Maximum refinement iterations
        
    Returns:
        dict: Results containing iterations and accuracies for each model
    """
    model_results = {model: {} for model in models}
    
    async def process_single_combination(model, data_item):
        """Process single model-data combination."""
        try:
            frames_path = os.path.join(MODEL_FRAMES_CONFIG[model]["frames_dir"], data_item["ID"])
            frames = process_frames(frames_path)
            
            result = await process_single_item_with_self_refine(
                client, 
                model, 
                data_item, 
                frames, 
                max_iterations
            )
            
            return {
                "model": model,
                "data_index": data_item["index"],
                "result": result
            }
        except Exception as e:
            with open('error_log_self_refine.txt', 'a') as f:
                f.write(f"Error processing item {data_item['index']} with model {model}: {str(e)}\n")
            return {
                "model": model,
                "data_index": data_item["index"],
                "result": {
                    "initial_answer": "",
                    "final_answer": "",
                    "all_answers": [],
                    "feedbacks": []
                }
            }
    
    # Control concurrency with semaphore
    semaphore = asyncio.Semaphore(20)
    
    async def process_with_semaphore(model, data_item):
        """Process with semaphore to limit concurrency."""
        async with semaphore:
            return await process_single_combination(model, data_item)
    
    # Build all tasks
    all_tasks = []
    for data_item in test_data:
        for model in models:
            all_tasks.append(process_with_semaphore(model, data_item))
    
    total_combinations = len(test_data) * len(models)
    completed = 0
    
    # Process completed tasks
    for future in asyncio.as_completed(all_tasks):
        result = await future
        if result:
            model = result["model"]
            data_index = result["data_index"]
            model_results[model][data_index] = result["result"]
            
            # Save intermediate results and print progress every 10 completions
            completed += 1
            if completed % 10 == 0:
                save_intermediate_results(model_results)
                print(f"Progress: {completed}/{total_combinations} combinations processed ({(completed/total_combinations)*100:.2f}%)")
    
    print(f"All {total_combinations} combinations processed.")
    
    # Save final intermediate results
    save_intermediate_results(model_results)
    
    # Calculate accuracy for each iteration
    iterations = list(range(max_iterations + 1))  # Include initial answer and all iterations
    final_results = {model: {i: {"correct": 0, "total": 0} for i in iterations} for model in models}
    
    eval_semaphore = asyncio.Semaphore(20)  # Limit evaluation concurrency
    
    async def evaluate_iteration(model, data_item, iteration):
        """Evaluate a specific iteration for a model-data combination."""
        async with eval_semaphore:
            try:
                frames_path = os.path.join(MODEL_FRAMES_CONFIG[model]["frames_dir"], data_item["ID"])
                frames = process_frames(frames_path)
                result = model_results[model][data_item["index"]]
                all_answers = result["all_answers"]
                
                if iteration >= len(all_answers):
                    return None
                
                answer = all_answers[iteration]
                is_correct = await grade_answer(client, data_item, answer, frames)

                return {
                    "model": model,
                    "iteration": iteration,
                    "is_correct": is_correct
                }
            except Exception as e:
                with open('error_log_self_refine.txt', 'a') as f:
                    f.write(f"Error in evaluation: {str(e)}\n")
                return {
                    "model": model,
                    "iteration": iteration,
                    "is_correct": False
                }
    
    # Create concurrent evaluation tasks
    eval_tasks = []
    for model in models:
        for data_item in test_data:
            for i in iterations:
                eval_tasks.append(evaluate_iteration(model, data_item, i))
    
    # Execute all evaluation tasks concurrently
    eval_results = await asyncio.gather(*eval_tasks)
    
    # Aggregate results
    for result in eval_results:
        if result is not None:
            model = result["model"]
            iteration = result["iteration"]
            final_results[model][iteration]["total"] += 1
            if result["is_correct"]:
                final_results[model][iteration]["correct"] += 1
    
    # Format results
    results = {}
    for model in models:
        results[model] = []
        for i in iterations:
            total = final_results[model][i]["total"]
            correct = final_results[model][i]["correct"]
            accuracy = correct / total if total > 0 else 0
            results[model].append(round(accuracy, 4))
    
    return {
        "iterations": iterations,
        "accuracies": results
    }


async def run_self_refine_analysis(models, test_data, max_iterations=3):
    """
    Run the complete self-refine analysis.
    
    Args:
        models (list): List of model names to evaluate
        test_data (list): Test dataset
        max_iterations (int): Maximum refinement iterations
        
    Returns:
        dict: Analysis results
    """
    async with AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    ) as client:
        results = await analyze_test_time_compute_self_refine(
            client,
            models,
            test_data,
            max_iterations=max_iterations
        )
        save_results(results)
        return results


def save_results(results, filename="self_refine_results.json"):
    """
    Save final results to JSON file.
    
    Args:
        results (dict): Results dictionary to save
        filename (str): Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        with open('error_log_self_refine.txt', 'a') as log:
            log.write(f"[{datetime.now()}] Save Error: {str(e)}\n")


def main():
    """Main function to run the self-refine video QA analysis."""
    print("Starting Self-Refine Video QA Analysis...")
    
    # Initialize error log
    with open('error_log_self_refine.txt', 'w') as f:
        f.write(f"=== Error Log Started at {datetime.now()} ===\n")
    
    # Configuration
    models = list(MODEL_FRAMES_CONFIG.keys())
    data_file = "VideoSimpleQA.json"  # Update with your data file path
    test_data = load_test_data(data_file)
    max_iterations = 3  # Maximum refinement iterations
    
    # Run analysis
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(
        run_self_refine_analysis(models, test_data, max_iterations)
    )
    
    print("Analysis completed successfully!")
    return results


if __name__ == "__main__":
    main()