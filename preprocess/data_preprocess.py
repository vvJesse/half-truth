import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tqdm import tqdm
import argparse
from utils import *
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

random.seed(42)

def main_merge_data():
    data = json.loads(open('./data/politifact-20241214.json', 'r').read())
    with open('./data/requests_merge.jsonl', 'w') as write_file:
        for idx, record in tqdm(enumerate(data)):
            claim = f"\"{record['claim']}\""
            example_id = str(record['example_id'])
            prompt = TABLE_MERGE.replace('[CLAIM]', claim).replace('[EVIDENCE]', evidence_str(record['evidence']))
            request = generate_request(example_id, prompt)
            json.dump(request, write_file)
            write_file.write('\n')
            
    openai_batch_api('./data/requests_merge.jsonl', desc='merge table data')


def code_2_result(code, evidence):
    merge_list = extract_lists(code)
    evidence_dict = {}
    for index, item in enumerate(evidence, start=1):
        evidence_dict[index] = item
    for merge_list_item in merge_list:
        evidence_dict = merge_execute(merge_list_item, evidence_dict)
    result_list = [evidence_dict[key] for key in sorted(evidence_dict.keys())]
    return result_list


def main_exec():
    data = json.loads(open('./data/politifact-20241214.json', 'r').read())
    exception_index = []
    with open('./data/requests_merge_res.jsonl', 'r') as res_f:
        for line in res_f:
            response_dict = json.loads(line)
            id = response_dict['custom_id']
            code_content = response_dict['response']['body']['choices'][0]['message']['content']
            if 'false' in code_content.lower():
                continue
            else:
                evidence = data[int(id)]['evidence']
                try:
                    data[int(id)]['evidence'] = code_2_result(code_content, evidence)
                except Exception as e:
                    exception_index.append(id)
    with open('./data/politifact-20241215.json', 'w') as write_file:
        json.dump(data, write_file, indent=4)


def main_human_correction():
    data = json.loads(open('./data/politifact-20241215.json', 'r').read())
    extra_code = json.loads(open('./data/code-extra.json', 'r').read())
    extra_code_map = {record['example_id']: record['code'] for record in extra_code}
    num_evidence = []
    for item in data:
        evidence = item['evidence']
        num_evidence.append(len(evidence))
        if len(evidence) > 60:
            id = item['example_id']
            if id in extra_code_map:
                evidence_dict = {}
                for index, obj in enumerate(evidence, start=1):
                    evidence_dict[index] = obj
                code = extra_code_map[id]
                evidence_new = code_2_result(code, item['evidence'])
                item['evidence'] = evidence_new
    with open('./data/politifact-20241215-2.json', 'w') as write_f:
        json.dump(data, write_f, indent=4)


def main_relevant():
    data = json.loads(open('./data/politifact-20241215-4.json', 'r').read())
    batch = []
    with open('./data/relevant-request.jsonl', 'w') as relevant_request_file:
        for item in tqdm(data):
            claim = item['claim']
            ruling = '\n'.join(item['ruling'])
            item_id = item['example_id']
            context = CONTEXT_TEMPLATE.replace('[DATE]', item['date']).replace('[SPEAKER]', item['speaker']).replace('[SOURCE]', item['source'])
            evidence = item['evidence']
            for idx, evi in enumerate(evidence):
                prompt = PROMPT_RELEVANT.replace('CLAIM', claim).replace('[CONTEXT]', context).replace('[RULING]', ruling).replace('[EVIDENCE]', evi)
                content = {
                    'id': f"{item_id}-{idx}",
                    'claim': claim,
                    'context': context,
                    'ruling': ruling,
                    'evidence': evi,
                    'prmopt': prompt,
                }
                request = generate_request(content['id'], prompt)
                batch.append(content)
                json.dump(request, relevant_request_file)
                relevant_request_file.write('\n')

    selected_batch = random.sample(batch, 50)
    
    with open('./data/human-annotation.json', 'w') as write_f:
        json.dump(selected_batch, write_f, indent=4)


def main_split():
    split_multiple_batch('./data/present-request.jsonl', './data/present_split', max_size=45 * 1024 * 1024)


def main_present():
    data = json.loads(open('./data/politifact-20241215-4.json', 'r').read())
    batch = []
    with open('./data/present-request.jsonl', 'w') as present_file:
        for item in tqdm(data):
            claim = item['claim']
            item_id = item['example_id']
            context = CONTEXT_TEMPLATE.replace('[DATE]', item['date']).replace('[SPEAKER]', item['speaker']).replace('[SOURCE]', item['source'])
            evidence = item['evidence']
            for idx, evi in enumerate(evidence):
                prompt = PROMPT_PRESENTED.replace('CLAIM', claim).replace('[CONTEXT]', context).replace('[EVIDENCE]', evi)
                content = {
                    'id': f"{item_id}-{idx}",
                    'claim': claim,
                    'context': context,
                    'evidence': evi,
                    'prmopt': prompt,
                }
                request = generate_request(content['id'], prompt)
                batch.append(content)
                json.dump(request, present_file)
                present_file.write('\n')


def bilingual_embedding_similarity(args):
    read_path = args.read_path
    write_path = args.write_path
    model = SentenceTransformer("Lajavaness/bilingual-embedding-large", trust_remote_code=True)
    with open(read_path, 'r') as fread:
        data = json.loads(fread.read())
        for item in tqdm(data):
            claim = [item['claim']]
            evidence = item['evidence']

            evidence_embeddings = model.encode(evidence)
            claim_embeddings = model.encode(claim)

            similarities = model.similarity(claim_embeddings, evidence_embeddings)
            bilingual_similarity = similarities.flatten().tolist()
            item['bilingual_similarity'] = bilingual_similarity
    
    with open(write_path, 'w') as fwrite:
        json.dump(data, fwrite, indent=4)

def bge_rerank(args):
    read_path = args.read_path
    write_path = args.write_path
    # model = SentenceTransformer('BAAI/bge-reranker-large')
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    with open(read_path, 'r') as fread:
        data = json.loads(fread.read())
        for item in tqdm(data):
            claim = [item['claim']]
            evidence = item['evidence']
            sentence_pairs = [
                [claim, evi] for evi in evidence
            ]
            bge_similarity = reranker.compute_score(sentence_pairs).flatten().tolist()

            item['bge_similarity'] = bge_similarity

    with open(write_path, 'w') as fwrite:
        json.dump(data, fwrite, indent=4)


def main_rank(args):
    if args.phase == 'bilingual':
        bilingual_embedding_similarity(args)
    if args.phase == 'rerank':
        bge_rerank(args)
                

def main(args):
    if args.task == 'merge':
        main_merge_data()
    if args.task == 'exec':
        main_exec()
    if args.task == 'correct':
        main_human_correction()
    if args.task == 'relevant':
        main_relevant()
    if args.task == 'split':
        main_split()
    if args.task == 'upload':
        # Remember to upload right file!
        openai_batch_api('data/present_split/batch_2.jsonl', 'present-2')
    if args.task == 'present':
        main_present()
    if args.task == 'rank':
        main_rank(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--phase', type=str)
    parser.add_argument('--read_path', type=str)
    parser.add_argument('--write_path', type=str)
    # parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    # parser.add_argument('--num_question', type=int, default=3)
    args = parser.parse_args()
    # main(args)
    main(args)