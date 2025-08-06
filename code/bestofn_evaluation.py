import os
import json
import base64
import asyncio
from tqdm import tqdm
from datetime import datetime
from openai import AsyncOpenAI

# Configuration - Replace with your own API credentials
API_KEY = "your_api_key_here"
BASE_URL = "your_base_url_here"  # e.g., "https://api.openai.com/v1"

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

# Template for generating answers with confidence
ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
Do not generate any intermediate reasoning process. Based on the video frames, directly output a short, accurate answer to the user's question in the following JSON format:
{"answer": "Your answer here"}
Do not include any additional text or explanations outside this JSON format.
"""

# Template for selecting the best answer from candidates
SELECTOR_PROMPT = """You are an expert evaluator. Based on the video frames and question, select the most correct answer from the candidates. Output only the selected answer in the following JSON format:
{"answer": "Your answer here"}
Do not include any additional text or explanations outside this JSON format.
"""

# Model configuration mapping model names to their frame directories and maximum supported frames
MODEL_FRAMES_MAP = {
    "claude_sonnet4": "frames_32/",  # Max 32 frames
}


def clean_json_response(response):
    """
    Extract and parse JSON from model response.
    
    Args:
        response (str): Raw response from the model
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end != 0:
        json_str = response[start:end]
        return json.loads(json_str)
    return None


def encode_image(image_path):
    """
    Encode image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_frames(frames_path):
    """
    Process video frames from a directory and encode them for API consumption.
    
    Args:
        frames_path (str): Path to directory containing video frames
        
    Returns:
        list: List of frame objects formatted for API consumption
    """
    frame_path_list = []
    for filename in os.listdir(frames_path):
        full_path = os.path.join(frames_path, filename)
        if os.path.isfile(full_path):
            frame_path_list.append(full_path)
    
    # Sort frames to maintain temporal order
    frame_path_list = sorted(frame_path_list)
    N = len(frame_path_list)
    
    # Encode all frames to base64
    base64_image_list = []
    for idx, name in enumerate(frame_path_list):
        base64_image_list.append(encode_image(name))
    
    # Format frames for API consumption
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


async def call_single_model(client, messages, model, n):
    """
    Make API call to a single model with retry logic.
    
    Args:
        client: AsyncOpenAI client instance
        messages (list): List of messages for the API call
        model (str): Model name
        n (int): Number of completions to generate
        
    Returns:
        Completion object or None if all retries fail
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
                with open('error_log_BoN.txt', 'a') as f:
                    f.write(f"Retrying model {model} after error: {str(e)}\n")
                return None
            print(f"Retrying model {model} after error: {str(e)}")
            await asyncio.sleep(10)
            continue


async def select_best_answer(client, data_item, candidates, frames):
    """
    Select the best answer from multiple candidates using a selector model.
    
    Args:
        client: AsyncOpenAI client instance
        data_item (dict): Test data item containing question and answer
        candidates (list): List of candidate answers
        frames (list): Video frames
        
    Returns:
        str: Selected best answer
    """
    try:
        formatted = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(candidates)])

        messages = [
            {"role": "system", "content": SELECTOR_PROMPT},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames},
            {"role": "user", "content": f"Question: {data_item['Multi_hop_Question']}"},
            {"role": "user", "content": f"Candidate answers:\n{formatted}"}
        ]
        
        response = await call_single_model(client, messages, "o3-0416-global", 1)
        
        answer = clean_json_response(response.choices[0].message.content).get("answer")
        if answer == "":
            return candidates[0]
        return answer
    except Exception as e:
        with open('error_log_BoN.txt', 'a') as f:
            f.write(f"Error selecting best answer: {str(e)}\n")
        return candidates[0]  


async def grade_answer(client, data_item, answer, frames):
    """
    Grade an answer using the grader model.
    
    Args:
        client: AsyncOpenAI client instance
        data_item (dict): Test data item containing question and gold answer
        answer (str): Answer to grade
        frames (list): Video frames
        
    Returns:
        bool: True if answer is correct, False otherwise
    """
    try:
        grade_messages = [
            {"role": "system", "content": GRADER_TEMPLATE},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames},
            {"role": "user", "content": f"Question: {data_item['Multi_hop_Question']}"},
            {"role": "user", "content": f"Gold target: {data_item['Multi_hop_Answer']}"},
            {"role": "user", "content": f"Predicted answer: {answer}"}
        ]
        
        response = await call_single_model(client, grade_messages, "o3-0416-global", 1)
        
        grade = response.choices[0].message.content.strip()[0]
        return grade == "A"
    except Exception as e:
        with open('error_log_BoN.txt', 'a') as f:
            f.write(f"Error grading answer: {str(e)}\n")
        return False


async def process_single_model_bestofn(client, model, data_item, frames_dict, n_inferences):
    """
    Generate multiple answers from a single model for Best-of-N evaluation.
    
    Args:
        client: AsyncOpenAI client instance
        model (str): Model name
        data_item (dict): Test data item
        frames_dict (dict): Dictionary mapping models to their processed frames
        n_inferences (int): Number of inference attempts
        
    Returns:
        tuple: (model_name, results_dict)
    """
    try:
        messages = [
            {"role": "system", "content": ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE},
            {"role": "user", "content": "Video frames:"},
            {"role": "user", "content": frames_dict[model]},
            {"role": "user", "content": f"Question: {data_item['Multi_hop_Question']}"}
        ]
        
        tasks = []
        semaphore = asyncio.Semaphore(30)  # Control concurrency to 30

        async def call_with_semaphore():
            async with semaphore:
                try:
                    response = await call_single_model(client, messages, model, 1)
                    answer_json = clean_json_response(response.choices[0].message.content)
                    return answer_json.get("answer", "") if answer_json else ""
                except Exception:
                    return ""

        for _ in range(n_inferences):
            tasks.append(call_with_semaphore())
        answers = await asyncio.gather(*tasks)
            
        return model, {"answers": answers}
    except Exception as e:
        with open('error_log_BoN.txt', 'a') as f:
            f.write(f"Error in {model}: {str(e)}\n")
        return model, {"answers": []}


def save_intermediate_results(model_results, filename="intermediate_results.json"):
    """
    Save intermediate results to JSON file.
    
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
        with open('error_log_BoN.txt', 'a') as log:
            log.write(f"[{datetime.now()}] Save Intermediate Results Error: {str(e)}\n")


async def analyze_test_time_compute_bestofn(client, models, test_data, n_inferences):
    """
    Analyze test-time compute using Best-of-N strategy.
    
    Args:
        client: AsyncOpenAI client instance
        models (list): List of model names to evaluate
        test_data (list): List of test data items
        n_inferences (int): Number of inferences per model per question
        
    Returns:
        dict: Results containing accuracies for different N values
    """
    model_results = {model: {} for model in models}
    
    # Generate candidates for all models and questions
    for data_item in tqdm(test_data, desc="Generating candidates"):
        frames_dict = {}
        for model in models:
            frames_path = os.path.join(MODEL_FRAMES_MAP[model], data_item["date"])
            frames_dict[model] = process_frames(frames_path)
        
        tasks = [
            process_single_model_bestofn(client, model, data_item, frames_dict, n_inferences)
            for model in models
        ]
        results = await asyncio.gather(*tasks)
        
        for model, result in results:
            model_results[model][data_item["id"]] = result["answers"]
    
    # Save intermediate results
    save_intermediate_results(model_results)
    
    # Load intermediate results for evaluation
    filename = "intermediate_results.json"
    with open(filename, 'r', encoding='utf-8') as f:
        model_results = json.load(f)

    # Evaluate different N values
    ns = [1, 2, 4, 8, 16]
    final_results = {model: {n: {"correct": 0, "total": 0} for n in ns} for model in models}

    eval_semaphore = asyncio.Semaphore(20)  # Control evaluation concurrency

    async def evaluate_single_item(model, data_item, n):
        """Evaluate a single item for a specific model and N value."""
        async with eval_semaphore:
            try:
                frames_path = os.path.join(MODEL_FRAMES_MAP[model], data_item["date"])
                frames = process_frames(frames_path)
                all_answers = model_results[model][str(data_item["id"])]
                
                if n > len(all_answers):
                    return None
                
                candidates = all_answers[:n]
                best_answer = await select_best_answer(client, data_item, candidates, frames)
                is_correct = await grade_answer(client, data_item, best_answer, frames)
                
                return {
                    "model": model,
                    "n": n,
                    "is_correct": is_correct
                }
            except Exception as e:
                with open('error_log_BoN.txt', 'a') as f:
                    f.write(f"Error in evaluation: {str(e)}\n")
                return {
                    "model": model,
                    "n": n,
                    "is_correct": False
                }

    # Create concurrent evaluation tasks
    eval_tasks = []
    for model in models:
        for data_item in test_data:
            for n in ns:
                eval_tasks.append(evaluate_single_item(model, data_item, n))

    # Execute all evaluation tasks concurrently
    eval_results = await asyncio.gather(*eval_tasks)

    # Aggregate results
    for result in eval_results:
        if result is not None:
            model = result["model"]
            n = result["n"]
            final_results[model][n]["total"] += 1
            if result["is_correct"]:
                final_results[model][n]["correct"] += 1
    
    # Calculate accuracies
    results = {}
    for model in models:
        results[model] = []
        for n in ns:
            total = final_results[model][n]["total"]
            correct = final_results[model][n]["correct"]
            accuracy = correct / total if total > 0 else 0
            results[model].append(round(accuracy, 4))
    
    return {
        "ns": ns,
        "accuracies": results
    }


async def run_bestofn_analysis(models, test_data):
    """
    Run Best-of-N analysis for all models.
    
    Args:
        models (list): List of model names
        test_data (list): Test dataset
        
    Returns:
        dict: Analysis results
    """
    async with AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    ) as client:
        results = await analyze_test_time_compute_bestofn(
            client,
            models,
            test_data,
            n_inferences=16  # Adjust N size as needed
        )
        save_results(results)
        return results


def save_results(results, filename="best_of_n_results.json"):
    """
    Save final results to JSON file.
    
    Args:
        results (dict): Results dictionary
        filename (str): Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        with open('error_log_BoN.txt', 'a') as log:
            log.write(f"[{datetime.now()}] Save Error: {str(e)}\n")


if __name__ == "__main__":
    print("Processing with Best of N method...")
    
    # Initialize error log
    with open('error_log_BoN.txt', 'w') as f:
        f.write(f"=== Error Log Started at {datetime.now()} ===\n")
    
    # Configuration
    models = list(MODEL_FRAMES_MAP.keys())
    data_file = "VideoSimpleQA.json"
    test_data = load_test_data(data_file)

    # Run Best-of-N analysis
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(run_bestofn_analysis(models, test_data))