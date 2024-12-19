import openai
from openai import OpenAI
import os
import re
import math

openai.api_key = os.environ.get('OPENAI_API_KEY')

IMPLICIT_QUESTION_GENERATION = """
Given the claim, context and evidence list. Please generate additional implicit questions.
Implicit question is the question raised beyond the literal meaning of the claim, and is related to Domain Knowledge, Context, Implicit Meaning, or Statistical Rigor.

Steps:

1. Does the claim need extra domain knowledge to verify? 
For example, Claim: “When President Obama was elected, the market crashed ... Trump was up 9%, President Obama was down 14.8% and President Bush was down almost 4%. There is an instant reaction on Wall Street.”
An implicit question as such is "Did Obama cause the stock market crash when he was elected?" Because we need to know domain knowledge of whether the stock market is correlated with the election.

2. Do we need extra context to understand the claim?
For example, Claim: With voting by mail, “you get thousands and thousands of people ... signing ballots all over the place.”
An implicit question as such is "Is there a greater risk of voting fraud with mail-in ballots?" Because we need to know the background that the claim is about the potential risks of mail-in ballots.

3. Does the claim implies a checkworthy deeper meaning?
For example, Claim: Nancy Pelosi bought $1.25 million in Tesla stock the day before Joe Biden signed an order “for all federal vehicles” to be electric.
An implicit question as such is "Were the stock purchases improper insider trading?" Because the claim implies this purchase is insider trading, which is also checkworthy.

4. Does the claim related to some statistics, and its statistical rigor needs checking?
For example, Claim: “No other country witnesses the number of gun deaths that we do here in the U.S., and it's not even close.”
An implicit question as such is "Is the United States the country with the the highest percentage of gun deaths?" Because highest number of gun deaths does not entail highest percentage of gun deaths.

5. Does the question answerable by the given evidence? If not, remove it.

Context: [CONTEXT]
Claim: [CLAIM]
Evidence:
[EVIDENCE]

# Follow steps above. At the end of output, generate at least 1, at most [QUESTION_NUMBER] most checkworthy yes-no implicit questions inside <Answer></Answer>. Questions are separated by '||'. Ensure that no literal question is generated.
"""

INTENT_DISCOVERY = """
You are required to discover the intended conclusion based on the claim, evidence, and a yes-no question.

Step:
1. Convert the yes-no question to statement and its negative statement.
2. Compare the 2 statements to the claim, based on your understanding of the claim and the evidence, which one is more likely the intended conclusion of the claim?
3. Output intented conclusino inside <INTENT></INTENT>, and negative intended conclusion inside <NEGATIVE></NEGATIVE>.

# Example
Claim: Hassen is out of state for 30 days in the last 3 months.
Evidence: Hassen's reponsibility is to address political issues. Hassen had a trip to California for a political meeting.
Question: Does Hassen commit her responsibility?

1. The statement could be "Hassen committed her responsibility.". The negative statement could be "Hassen did not commit her responsibility."
2. The claim itself seem to critizing Hassen not commit her responsibility because out of state for too long. Therefore, the intended conclusion is "Hassen did not commit her responsibility.", and the negative intended conclusion is "Hassen committed her responsibility.".

<INTENT>Hassen did not commit her responsibility.</INTENT>
<NEGATIVE>Hassen committed her responsibility.</NEGATIVE>

# Your practice
Claim: [CLAIM]
Evidence: [EVIDENCE]
Question: [QUESTION]
"""

TABLE_MERGE = """
You will be given a claim, its context and evidence dict. The evidence dict consist of a passage where consecutive key means consecutive content. 
You are required to merge the **table data** into a single value accessed by a single key, and then output your solution with code.
You should follow the steps:
1. Does the evidence dict contain table data consist of multiple lines?
2. If contain, then merge then with the given function and its number. You should both merge the data and its title and necessary explanation. And then set 'has_table' as True.
3. If no, set 'has_table' as False. Do not merge any data.
4. You should finish the Python code and your code should only contain multipel call of merge_execute() or assignment to 'has_table', do not explain your rationale and do not write annotation.

def merge_execute(numbers, paragraph_dict):
    # Merge the paragraphs corresponding to the specified numbers. Keep the key of the first number and delete the keys of other merged paragraphs.
    # You must ensure the given numbers are consecutive.
    if not numbers:
        return paragraph_dict
    numbers.sort()
    print(numbers)
    if numbers[-1] - numbers[0] + 1!= len(numbers):
        raise ValueError("The numbers in the list must be consecutive.")
    main_key = numbers[0]
    merged_item = []

    for key in numbers:
        merged_item.append(paragraph_dict[key])
        if key!= main_key:
            del paragraph_dict[key]

    merge_content = '\n'.join(merged_item)

    paragraph_dict[main_key] = merge_content
    return paragraph_dict


claim = [CLAIM]
evidence = [EVIDENCE]

# Dose the evidence have table data? Your answer should be inside <has_table></has_table>
<has_table>
</has_table>
# If has_table == True, then merging data inside below <Answer></Answer> The merging index **must** be consecutive!!
<Answer>
</Answer>
"""

CONTEXT_TEMPLATE = "Date: [DATE]   Originator: [SPEAKER]   Source: [SOURCE]\n"

PROMPT_RELEVANT = """
You are required to determine whether an evidence is related to the fact checking of the claim based on the claim and justification.
The evidence is related to the fact checking of a claim when it tells the context, background, neccessary domain knowledge, implicit meaning or even just a rephrase of the claim.

Claim: [CLAIM]
Context: [CONTEXT]
Justification: [RULING]
Evidence: [EVIDENCE]

Does the evidence related to the fact checking of the claim? From {Yes, No}, give your answer inside <Answer></Answer>.

<Answer></Answer>
"""

PROMPT_PRESENTED = """
Based on the claim, context and evidence, you are required to determine whether the given information of a piece of evidence presented in a claim.

Claim: [CLAIM]
Context: [CONTEXT]
Evidence: [EVIDENCE]

Does the evidence express the information that have been presented in the claim? From {Yes, No}, give your answer inside <Answer></Answer>.

<Answer></Answer>
"""


def merge_execute(numbers, paragraph_dict):
    # Merge the paragraphs corresponding to the specified numbers. Keep the key of the first number and delete the keys of other merged paragraphs.
    # You must ensure the given numbers are consecutive.
    if not numbers:
        return paragraph_dict
    numbers.sort()
    if numbers[-1] - numbers[0] + 1 - len(numbers) > 2:
        raise ValueError("The numbers in the list must be consecutive.")
    main_key = numbers[0]
    merged_item = []

    for key in numbers:
        merged_item.append(paragraph_dict[key])
        if key!= main_key:
            del paragraph_dict[key]

    merge_content = '\n'.join(merged_item)

    paragraph_dict[main_key] = merge_content
    return paragraph_dict


def evidence_str(evidence_list):
    content = "{\n"
    for i, evi in enumerate(evidence_list):
        content += f'\t{i+1} : \"{evi}\"\n'
    content += "}\n"
    return content

def extract_lists(s):
    result = []
    pattern_1 = r'\[(.*?)\]'
    matches_1 = re.findall(pattern_1, s)
    for match in matches_1:
        try:
            elements = [int(x.strip()) for x in match.split(',')]
            result.append(elements)
        except ValueError:
            continue

    pattern_2 = r'list\(range\((.*?)\)\)'
    matches_2 = re.findall(pattern_2, s)
    for match in matches_2:
        parts = match.split(',')
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                result.append(list(range(start, end)))
            except ValueError:
                continue
    return result


def call_gpt(cur_prompt, stop=None, model="gpt-4o-mini"):
    reasoner_messages = [
        {
            "role": "user",
            "content": cur_prompt
        },
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=reasoner_messages,
    )
    returned = completion.choices[0].message.content
    return returned


def generate_request(example_id, prompt):
    request = {
        "custom_id": example_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            ],
            "max_tokens": 1000
        }
    }
    return request


def openai_batch_api(batch_data_path, desc):
    client = OpenAI()

    batch_input_file = client.files.create(
    file=open(batch_data_path, "rb"),
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": desc
        }
    )


def split_multiple_batch(input_file, output_dir, max_size=85 * 1024 * 1024):
    os.makedirs(output_dir, exist_ok=True)
    
    current_size = 0
    file_count = 1
    output_file = open(os.path.join(output_dir, f'batch_{file_count}.jsonl'), 'w', encoding='utf-8')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_size = len(line.encode('utf-8'))

            if current_size + line_size > max_size:
                output_file.close()
                file_count += 1
                output_file = open(os.path.join(output_dir, f'batch_{file_count}.jsonl'), 'w', encoding='utf-8')
                current_size = 0
            
            output_file.write(line)
            current_size += line_size

    output_file.close()