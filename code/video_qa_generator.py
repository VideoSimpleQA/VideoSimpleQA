import os
import json
import time
import base64
import asyncio
from multiprocessing import Pool, Manager
from openai import OpenAI, AsyncOpenAI

# Configuration - Replace with your own API credentials
API_KEY = "your_api_key_here"
BASE_URL = "your_base_url_here"

# Prompt for generating Question-Answer pairs from video content
GENERATE_QA_PROMPT = """\
You will be provided with video frames extracted from a video and a related document. Your task is to generate factual Questions and corresponding standard Answers based on these materials. The Questions must be derivable from the video frames and require additional knowledge to be answered, while Answers need to be synthesized from the video content and the document knowledge provided. If the video frames are not related to the document content, you may generate factual questions based on the visual content alone.
The generated Question needs to meet the following requirements:
1. Questions must relate to visible content in the video frames. Do not generate questions solely based on document information not visible in frames. For example, If frames show cooking steak but document contains filmmaker's biography, don't generate the question about the filmmaker.
2. Questions must relate to objective, verifiable facts, for example, you can ask "Who is the winner of the 2024 Nobel Prize in Physics?" You must not construct subjective questions related to personal opinions or feelings, such as "What do you think of xxx?".
3. Each question must have a single, indisputable answer. Avoid ambiguous or vague questions. For example, do not ask "Which is Zhou Ruchang's most well-known work?" because "most well-known" may be controversial.
4. Answers must not change over time. For example, "Who is the current president of the United States? " is not a suitable Question, because the identity of the president will change with the election results.
5. Questions should be challenging enough to reflect domain knowledge. For example: The movie "Striptease" is adapted from the novel of the same name. Who is the author of the novel?
6. Answers should be concise and use accurate but minimal wording
7. Questions and responses should not contain the words "frames", "document" and "images". Use "video" instead of "frames" and "images".
8. Use Arabic numerals instead of English words for numbers. For example: Use "3" instead of "Three".
9. When specifying the date, please use the format YYYY-MM-DD. For example: 2024-12-15.
10. You must generate exactly 3 questions, each starting with a different question word.
11. The question words must be selected from this list: what, who, when, where, how, why, whom, whose, which
12. No two questions should use the same question word.
13. The questions should focus on different aspects of the content to maintain variety.
14. All questions and answers MUST be in English, regardless of the language in the provided document.

Please return exactly three question-answer pairs in this specific JSON format. Do not include any other text, explanations, or multiple answers:
{
    "QA1": {"Question": "Your first question here", "Answer": "Your first answer here"},
    "QA2": {"Question": "Your second question here", "Answer": "Your second answer here"},
    "QA3": {"Question": "Your third question here", "Answer": "Your third answer here"}
}

The following are some examples:
Example 1:
{
    "QA1": {"Question": "What type of microscope was used to capture the cell division process shown in the video?", "Answer": "Phase-contrast microscope"},
    "QA2": {"Question": "How does the Venus flytrap shown in the video capture its prey?", "Answer": "By rapidly closing its modified leaf lobes"},
    "QA3": {"Question": "When did Marie Curie discover radium, as demonstrated in the video recreation?", "Answer": "1898"}
}

Example 2:
{
    "QA1": {"Question": "Who patented this specific phonograph design shown in the video?", "Answer": "Thomas Edison"},
    "QA2": {"Question": "Whose theory of general relativity is being demonstrated in the video through the gravity well experiment?", "Answer": "Albert Einstein"},
    "QA3": {"Question": "Where was the first Large Hadron Collider experiment shown in the video conducted?", "Answer": "CERN, Geneva"}
}

Example 3:
{
    "QA1": {"Question": "Why does liquid nitrogen make the rubber ball in the video shatter upon impact?", "Answer": "Because it freezes the molecular bonds making the rubber brittle"},
    "QA2": {"Question": "Which chemical element creates the distinctive blue flame color demonstrated in the video?", "Answer": "Copper"},
    "QA3": {"Question": "To whom did Niels Bohr write the letter about quantum mechanics that appears in the video archive?", "Answer": "Albert Einstein"}
}

Let's get started!
"""

# Prompt for validating question quality
CHECK_QUESTION_PROMPT = """\
You are a data quality checker responsible for evaluating questions and answers generated from video content and accompanying documents. Your task is to ensure each QA pair meets strict quality standards:
1. Each question must have a single, indisputable answer.
2. Question must relate to visible content in the video frames. 
3. Answers can draw from external knowledge sources that provide factual, verifiable information beyond what's shown in the video.
4. No subjective opinions or personal preferences.
5. Answers must not change over time.

Please evaluate the question and return exactly one JSON response in this format:
If the question meets all requirements, return {"Verification": "Yes", "Reason": ""}
If the question does not meet any requirement, return {"Verification": "No", "Reason": "Specific reason why the question fails to meet requirements"}
Do not include any additional text or explanations outside this JSON format.

The following are some examples:
Example 1:
Question: What's the most impressive scene in the video?
Answer: The mountain climbing sequence
Return results: {"Verification": "No", "Reason": "Question is subjective and relies on personal opinion. Terms like 'most impressive' cannot have a single, indisputable answer."}

Example 2:
Question: Who is the world record holder for the event shown in the video?
Answer: Usain Bolt
Return results: {"Verification": "No", "Reason": "Answer may change over time."}

Example 3:
Question: Which two cities does the Han-Shi Expressway connect Wuhan City with?
Answer: Xiaogan City
Return results: {"Verification": "Yes", "Reason": ""}

Let's get started!
"""

# Prompt for evaluating model response quality and correctness
JUDGE_PROMPT = """
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
    
Just return the letters "A", "B", or "C", with no text around it.
"""


def clean_json_response(response):
    """
    Clean and parse JSON response from model output.
    
    Args:
        response (str): Raw response from the model
        
    Returns:
        dict or None: Parsed JSON object or None if parsing fails
    """
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end != 0:
        json_str = response[start:end]
        return json.loads(json_str)
    return None


def call_model(messages, model, video_id):
    """
    Call OpenAI model with retry mechanism for robustness.
    
    Args:
        messages (list): List of message dictionaries for the conversation
        model (str): Model identifier to use
        video_id (str): Video ID for error tracking
        
    Returns:
        str or None: Model response or None if all retries fail
    """
    response = None
    max_retry_times = 10
    retry_times = 0

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    while response is None and retry_times < max_retry_times:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages
            )
            response = completion.choices[0].message.content
        except Exception as e:
            retry_times += 1
            print(f"Unexpected error for {video_id}: {str(e)}")
            print(f"Retrying {video_id} ({retry_times}/{max_retry_times})...")
            time.sleep(10)
            continue
    
    return response


def llm_verification(question, frames, description, video_id):
    """
    Verify question quality using LLM-based validation.
    
    Args:
        question (str): Generated question to verify
        frames (list): Video frames as base64 encoded images
        description (str): Document description associated with the video
        video_id (str): Video ID for error tracking
        
    Returns:
        dict or None: Verification result or None if verification fails
    """
    if question == "":
        print(f"Unexpected error for {video_id}: Question is empty")
        return None
        
    messages = [{"role": "system", "content": CHECK_QUESTION_PROMPT}]
    messages.append({"role": "user", "content": f"Question: {question}"})
    messages.append({"role": "user", "content": "Here are the video frames:"})
    messages.append({"role": "user", "content": frames})
    messages.append({"role": "user", "content": f"This is the document file : {description}"})
    
    return clean_json_response(call_model(messages, "", video_id))


async def call_single_model(client, messages, model, video_id):
    """
    Asynchronously call a single model with retry mechanism.
    
    Args:
        client: AsyncOpenAI client instance
        messages (list): Conversation messages
        model (str): Model identifier
        video_id (str): Video ID for error tracking
        
    Returns:
        str or None: Model response or None if all retries fail
    """
    max_retry_times = 10
    retry_times = 0
    
    while retry_times < max_retry_times:
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            retry_times += 1
            if retry_times == max_retry_times:
                print(f"Failed to call model {model} after {max_retry_times} retries. Error: {str(e)}")
                return None
            print(f"Retrying {video_id} ({retry_times}/{max_retry_times})...")
            await asyncio.sleep(10)
            continue


async def diff_filtering_async(messages, models, question, target, frames, video_id):
    """
    Asynchronously evaluate question difficulty using multiple models.
    First generates answers from multiple models, then judges their correctness.
    
    Args:
        messages (list): Messages for answer generation
        models (list): List of model identifiers to use
        question (str): Question to evaluate
        target (str): Expected correct answer
        frames (list): Video frames
        video_id (str): Video ID for error tracking
        
    Returns:
        list: List of binary results (1 = correct, 0 = incorrect) for each model
    """
    async with AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    ) as client:
        # Generate answers from multiple models
        tasks1 = [
            call_single_model(client, messages, model, video_id)
            for model in models
        ]
        responses1 = await asyncio.gather(*tasks1)
        answers = [response for response in responses1]

        # Prepare judgment messages for each answer
        new_messages = []
        for answer in answers:
            new_message = [{"role": "system", "content": JUDGE_PROMPT}]
            new_message.append({"role": "user", "content": "Here are the video frames:"})
            new_message.append({"role": "user", "content": frames})
            new_message.append({"role": "user", "content": f"Question: {question}"})
            new_message.append({"role": "user", "content": f"Gold target: {target}"})
            new_message.append({"role": "user", "content": f"Predicted answer: {answer}"})
            new_messages.append(new_message)

        # Judge each answer's correctness
        tasks2 = [
            call_single_model(client, message, "", video_id)
            for message in new_messages
        ]
        responses2 = await asyncio.gather(*tasks2)
        results = [1 if response == "A" else 0 for response in responses2]
        
        return results


def call_models(question, answer, frames, video_id):
    """
    Evaluate QA difficulty using multiple models.
    
    Args:
        question (str): Question to evaluate
        answer (str): Expected answer
        frames (list): Video frames
        video_id (str): Video ID for error tracking
        
    Returns:
        int: 0 if question is too easy, 1 if appropriately difficult, -1 if error occurred
    """
    models = [""]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    messages = [{"role": "system", "content": "Please answer the user's question accurately based on the video frames in the short format"}]
    messages.append({"role": "user", "content": "Here are the video frames:"})
    messages.append({"role": "user", "content": frames})
    messages.append({"role": "user", "content": f"This is the question: {question}"})
    
    results = loop.run_until_complete(diff_filtering_async(messages, models, question, answer, frames, video_id))

    flag = 0
    try:
        for result in results:
            try:
                if result == 0:  # If any model got it wrong, question is appropriately difficult
                    flag = 1
                    break
            except:
                continue
    except:
        return -1
    return flag


def diff_filtering(question, answer, frames, video_id):
    """
    Filter out questions that are too easy by testing with multiple models.
    
    Args:
        question (str): Question to evaluate
        answer (str): Expected answer
        frames (list): Video frames
        video_id (str): Video ID for error tracking
        
    Returns:
        int: 0 if too easy, 1 if appropriately difficult, -1 if error occurred
    """
    try:
        return call_models(question, answer, frames, video_id) 
    except Exception as e:
        print(f"Unexpected error for {video_id}: {str(e)}")
        return -1


def parse_to_json(response, video_id, description, frames, error_ids):
    """
    Parse and validate QA response, converting to structured JSON format.
    
    Args:
        response (dict): Raw QA response from model
        video_id (str): Video identifier
        description (str): Video description
        frames (list): Video frames
        error_ids (list): List to track error cases
        
    Returns:
        dict: Structured output with validated QA pairs
    """
    if response is None:
        error_ids.append(video_id)
        output = {
            "ID": video_id,
            "Response": f"Unexpected error for {video_id}"
        }
        return output
        
    try:
        qa_pairs = []
        
        # Process each of the 3 expected QA pairs
        for i in range(1, 4):
            qa_key = f"QA{i}"
            if qa_key not in response:
                continue
            
            current_qa = response[qa_key]
            
            # Check question difficulty
            flag = diff_filtering(current_qa["Question"], current_qa["Answer"], frames, video_id)
            
            if flag == 0:
                qa_pairs.append({
                    "Question": current_qa["Question"],
                    "Answer": current_qa["Answer"],
                    "Status": "Too Simple"
                })
                continue
            
            if flag == -1:
                qa_pairs.append({
                    "Question": current_qa["Question"],
                    "Answer": current_qa["Answer"],
                    "Status": "Error in Processing"
                })
                continue
            
            # Verify question quality
            verify_resp = llm_verification(current_qa["Question"], frames, description, video_id)
            if verify_resp is None:
                qa_pairs.append({
                    "Question": current_qa["Question"],
                    "Answer": current_qa["Answer"],
                    "Status": "Verification Failed"
                })
                continue
            
            qa_pair = {
                "Question": current_qa["Question"],
                "Answer": current_qa["Answer"],
                "Verification": verify_resp.get("Verification")
            }
            
            if verify_resp.get("Verification") == "No":
                qa_pair["Reason"] = verify_resp.get("Reason")
            
            qa_pairs.append(qa_pair)
        
        output = {
            "ID": video_id,
            "URL": f"https://commons.wikimedia.org/wiki/Template:Motd/{video_id}",
            "Description": description,
            "QAPairs": qa_pairs
        }
        
        return output
            
    except Exception as e:
        print(f"Unexpected error for {video_id}: {str(e)}")
        error_ids.append(video_id)
        output = {
            "ID": video_id,
            "Response": str(e)
        }
        return output


def generate_qa(generate_qa_prompt, frames, description, video_id):
    """
    Generate question-answer pairs from video frames and description.
    
    Args:
        generate_qa_prompt (str): System prompt for QA generation
        frames (list): Processed video frames
        description (str): Video description text
        video_id (str): Video identifier
        
    Returns:
        dict or None: Generated QA pairs or None if generation fails
    """
    messages = []
    messages.append({"role": "system", "content": generate_qa_prompt})
    messages.append({"role": "user", "content": "Here are the video frames:"})
    messages.append({"role": "user", "content": frames})
    messages.append({"role": "user", "content": f"This is the document file: {description}"})
    
    return clean_json_response(call_model(messages, "", video_id))


def encode_image(image_path):
    """
    Encode image file to base64 string.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_frames(n, frame_path_list):
    """
    Process video frames into format suitable for vision models.
    
    Args:
        n (int): Number of frames to process
        frame_path_list (list): List of frame file paths
        
    Returns:
        list: List of formatted frame objects for model input
    """
    base64_image_list = []
    for idx, name in enumerate(frame_path_list):
        base64_image_list.append(encode_image(name))
        
    frames = []
    for idx in range(n):
        frames.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image_list[idx]}",
                "detail": "low" 
            },
        })
    return frames


def load_descriptions(descriptions_dict):
    """
    Load video descriptions from JSON file into shared dictionary.
    
    Args:
        descriptions_dict (dict): Shared dictionary to store descriptions
    """
    with open("../../data/wiki_videos/descriptions.json", 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
        
    for item in descriptions:
        video_id = item['id']
        rag_text = ' '.join(item['rag_res']) if isinstance(item['rag_res'], list) else item['rag_res']
        combined_text = item['des'] + '\n\n' + item['add_desc'] + '\n\n' + rag_text
        descriptions_dict[video_id] = combined_text


def process_video(video_name, descriptions_dict, error_ids, processed_ids):
    """
    Process a single video to generate QA pairs.
    
    Args:
        video_name (str): Video filename
        descriptions_dict (dict): Dictionary of video descriptions
        error_ids (list): Shared list to track processing errors
        processed_ids (dict): Shared dictionary to track processed videos
        
    Returns:
        dict or None: Processed QA result or None if processing fails
    """
    current_id = video_name.strip()
    video_id = current_id.split('.')[0]

    # Skip if already processed
    if video_id in processed_ids:
        print(f"Skipping {video_id}: Already processed")
        return None
    processed_ids[video_id] = True 

    print(f"Processing {video_id}")
    description = descriptions_dict.get(video_id, "")

    # Load and process video frames
    frames_path = f'../../data/wiki_videos/frames_15/{video_id}'
    frame_path_list = []
    for filename in os.listdir(frames_path):
        full_path = os.path.join(frames_path, filename)
        if os.path.isfile(full_path):
            frame_path_list.append(full_path)
    frame_path_list = sorted(frame_path_list)
    n = len(frame_path_list)
    frames = process_frames(n, frame_path_list)

    # Generate QA pairs
    try:
        response = generate_qa(GENERATE_QA_PROMPT, frames, description, video_id)
        if response is None:
            error_ids.append(video_id)
            return None
    except Exception as e:
        print(f"Unexpected error for {video_id}: {str(e)}")
        error_ids.append(video_id)
        return None

    # Parse and validate results
    resp_json = parse_to_json(response, video_id, description, frames, error_ids)
    if resp_json:
        with open('output.json', 'a', encoding='utf-8') as f:
            json.dump(resp_json, f, ensure_ascii=False, indent=4)
            f.write(",\n")
    print(f"Finished processing {video_id}")

    return resp_json


def main():
    """
    Main function to orchestrate the video QA generation process.
    Uses multiprocessing to handle multiple videos concurrently.
    """
    start_time = time.time()

    # Initialize shared data structures for multiprocessing
    manager = Manager()
    descriptions_dict = manager.dict()  
    error_ids = manager.list()  
    processed_ids = manager.dict()  

    # Load video descriptions
    load_descriptions(descriptions_dict)

    # Read list of video files to process
    txt_file_path = 'videos_name.txt'
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    # Initialize output file
    with open('output.json', 'w', encoding='utf-8') as f:
        f.write("[\n")

    # Process videos using multiprocessing
    with Pool(processes=10) as pool:
        pool.starmap(process_video, [(video_name, descriptions_dict, error_ids, processed_ids) for video_name in data])

    # Finalize output file
    with open('output.json', 'rb+') as f:
        f.seek(-2, os.SEEK_END)
        f.truncate()
        f.write(b"\n]")
    
    # Write error log
    with open('error_output.txt', 'w', encoding='utf-8') as f:
        f.write("Error IDs:\n")
        for error_id in error_ids:
            f.write(f"{error_id}\n")

    end_time = time.time()
    print("Total running time: {:.2f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    main()