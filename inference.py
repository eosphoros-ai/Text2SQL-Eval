"""
# @author qumu
# @date 2023/8/15
# @module hf_inference.py
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import traceback

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    StoppingCriteria,
)

from utils.extract_sql_meta import isConstCanFind, convert_schema, fetch_column_all_value, is_number

MODEL_TYPES = {
    "llama": LlamaForCausalLM,
}

TOKENIZERS = {
    "llama": LlamaTokenizer,
}

SYSTEM_ROLE_START_TAG = "<s>system\n"
HUMAN_ROLE_START_TAG = "<s>human\n"
BOT_ROLE_START_TAG = "<s>bot\n"
SYSTEM = 'You are a professional SQL engineer and you are writing SQL queries for data query tasks.\n'


class EotOrPadStopping(StoppingCriteria):
    """
    Args:
        start_length (:obj:`int`):
            The number of initial tokens.
    """

    def __init__(self, stop_token_id, pad_token_id):
        self.stop_token_id = stop_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return torch.logical_or(input_ids[:, -1] == self.stop_token_id,
                                input_ids[:, -1] == self.pad_token_id).all().item()


def load_model_tokenizer(path, model_type=None, peft_path=None, quantization=None, torch_dtype=torch.bfloat16, eos_token=None, pad_token=None, batch_size=1):
    """
        load model and tokenizer by transfromers
    """
    if model_type:
        ModelClass = MODEL_TYPES.get(model_type, AutoModelForCausalLM)
    else:
        ModelClass = AutoModelForCausalLM
    TokenizerClass = TOKENIZERS.get(model_type, AutoTokenizer)
    print(f"Tokenizer Class: {TokenizerClass}, Model Class: {ModelClass}")

    config, unused_kwargs = AutoConfig.from_pretrained(
        path,
        use_flash_attn=batch_size==1,
        use_xformers=batch_size==1,
        trust_remote_code=True,
        return_unused_kwargs=True)

    config_dict = config.to_dict()

    tokenizer = TokenizerClass.from_pretrained(path, trust_remote_code=True, use_fast=False, legacy=False)
    if eos_token:
        print("input eos_token: ", eos_token)
        try:
            tokenizer.eos_token = eos_token
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        except:
            print(tokenizer.eos_token, tokenizer.eos_token_id)

    elif "eos_token_id" in config_dict:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
        tokenizer.eos_token_id = config.eos_token_id
    elif "eos_token" in config_dict:
        print(config.eos_token)
        tokenizer.eos_token = config.eos_token
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    if pad_token:
        print("input pad_token: ", pad_token)
        try:
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        except:
            print(tokenizer.pad_token, tokenizer.pad_token_id)

    elif "pad_token_id" in config_dict:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config.pad_token_id)
        tokenizer.pad_token_id = config.pad_token_id
    elif "pad_token" in config_dict:
        print(config.eos_token)
        tokenizer.pad_token = config.pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)

    tokenizer.padding_side = "left"
    print(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    print(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
    print(tokenizer)

    base_model = ModelClass.from_pretrained(
        path,
        config=config,
        load_in_8bit=(quantization=='8bit'),
        load_in_4bit=(quantization=='4bit'),
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        # use_safetensors=False,
    )

    print("Loading Original MODEL...")
    model = base_model

    model.eval()

    print("=======================================MODEL Configs=====================================")
    print(model.config)
    print("=========================================================================================")
    print("=======================================MODEL Archetecture================================")
    print(model)
    print("=========================================================================================")

    return model, tokenizer


def hf_inference(model, tokenizer, text_list, max_new_tokens=512, do_sample=True, **kwargs):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
    logging.info("================================Prompts and Generations=============================")

    outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )

    gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    for i in range(len(text_list)):
        logging.info('=========' * 10)
        logging.info(f'Prompt:\n{text_list[i]}')
        gen_text[i] = gen_text[i].replace(tokenizer.pad_token, '')
        logging.info(f'Generation:\n{gen_text[i]}')
    sys.stdout.flush()
    return gen_text


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = int(ls_len / n)
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def load_test_data(valid_dataset_path):
    content_list, database_list = [], []
    eval_datas = []
    if valid_dataset_path.endswith(".jsonl"):
        with open(valid_dataset_path) as f:
            for line in f:
                eval_datas.append(json.loads(line))
    else:
        with open(valid_dataset_path) as f:
            eval_datas = json.load(f)
    for eval_data in eval_datas:
        content = eval_data['chat_rounds'][1]['content']
        database = eval_data['db_id']
        if not content.endswith("\n"):
            content += "\n"
        content_list.append(content)
        database_list.append(database)
    return content_list, database_list


def isValidSQL(sql, db_list):
    for db in db_list:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
        except Exception as e:
            return e
    return None


def second_round_prompt_check_error(sql, e):
    return "An ERROR in the SQL. You must fix it." \
           f"\n ERROR : {str(e)}" \
           f"\n SQL : {str(sql)}"


def second_round_check(content, db_list, predict_sql):
    e = isValidSQL(predict_sql, db_list)
    second_prompt = None
    if e is not None:
        second_prompt = second_round_prompt_check_error(predict_sql, e)
    else:
        tables, _ = convert_schema(content)
        may_be_used_column = isConstCanFind(predict_sql, db_list, tables)
        if len(may_be_used_column) != 0:
            second_prompt = second_round_prompt_check_constrain(predict_sql, may_be_used_column, db_list)
    return second_prompt


def is_same_value(database_value, compare_value):
    database_value = database_value.strip()
    if database_value.lower() == compare_value.lower():
        return True
    if is_abbreviation(database_value, compare_value):
        return True
    if is_abbreviation(compare_value, database_value):
        return True
    return False


def is_abbreviation(word, abbreviation):
    i, j = 0, 0
    word, abbreviation = word.lower(), abbreviation.lower()
    while i < len(word) and j < len(abbreviation):
        if word[i] == abbreviation[j]:
            i += 1
            j += 1
        elif abbreviation[j].isdigit() and abbreviation[j] != "0":
            count = 0
            while j < len(abbreviation) and abbreviation[j].isdigit():
                count = count * 10 + int(abbreviation[j])
                j += 1
            i += count
        else:
            i += 1

    return j == len(abbreviation)


def second_round_prompt_check_constrain(sql, may_be_other_fields, db_list):
    prompt_str_list = []
    for may_be_other_field in may_be_other_fields:
        table = may_be_other_field["table"]
        column = may_be_other_field["not_right_column"]
        compare_type = may_be_other_field["compare_type"]
        compare_value = str(may_be_other_field["compare_value"])
        may_in_columns = may_be_other_field["may_in_columns"]
        if len(may_in_columns) == 0:
            if is_number(compare_value):
                continue
            column_values = fetch_column_all_value(column, db_list, table)
            abbreviation_value = ""
            for value in column_values:
                if is_number(value):
                    continue
                if is_same_value(compare_value, value):
                    abbreviation_value = value
            if abbreviation_value != "":
                prompt_str = f"The variable \"{compare_value}\" has a case error. It should be written as \"{abbreviation_value}\"" \
                             f"\nPlease confirm that SQL have used the correct constants and Return the SQL after check!" \
                             f"\nSQL: {sql}" \
                             f"\nShould Use Value: {abbreviation_value}"
                prompt_str_list.append(prompt_str)
        else:
            prompt_str = f"No value in column {column}  of table {table} {compare_type} {compare_value}," \
                         f"\nBut, there are values in columns {','.join(may_in_columns)} of table {table}" \
                         f"\nPlease make sure you are using the correct columns in SQL !" \
                         f"\nSQL : {sql}" \
                         f"\nNo Value Compare: {table}.{column} {compare_type} {compare_value}" \
                         f"""\nValue Exists Compare: {','.join([table + "." + c + " " + compare_type + " " + compare_value
                                                                for c in may_in_columns])}"""
            prompt_str_list.append(prompt_str)
    if len(prompt_str_list) == 0:
        return None
    return '\n'.join(prompt_str_list)


def start_inference(base_model_path, valid_file_path, db_dir):
    content_list, database_list = load_test_data(valid_file_path)
    model, tokenizer = load_model_tokenizer(base_model_path, model_type='deepseek',
                                            eos_token='<｜end▁of▁sentence｜>', pad_token='<｜end▁of▁sentence｜>')
    cnt, err = 0, 0
    predict_result = []

    for content, database in zip(content_list, database_list):
        cnt += 1
        try:
            prompt = f"{SYSTEM_ROLE_START_TAG}{SYSTEM}{HUMAN_ROLE_START_TAG}{content}{BOT_ROLE_START_TAG}"
            predict_res = hf_inference(model, tokenizer, [prompt], do_sample=False, num_beams=1,
                                       num_return_sequences=1)
            curr_predict = predict_res[0].split('\n')[0]
            db_list = [os.path.join(db_dir, database, database + ".sqlite")]
            second_prompt = second_round_check(content, db_list, curr_predict)
            if second_prompt is not None:
                prompt = [
                    f"{SYSTEM_ROLE_START_TAG}{SYSTEM}{HUMAN_ROLE_START_TAG}{content}{BOT_ROLE_START_TAG}"
                    f"{curr_predict}{HUMAN_ROLE_START_TAG}{second_prompt}{BOT_ROLE_START_TAG}"]
                second_predict_res = hf_inference(model, tokenizer, prompt, do_sample=False,
                                                  num_beams=1,
                                                  num_return_sequences=1)
                curr_predict = second_predict_res[0].split('\n')[0]
            predict_result.append(curr_predict)
        except Exception as e:
            logging.error(f'error: {e}')
            logging.error(traceback.format_exc())
            err += 1

    return predict_result


def main(opt):
    predict_result = start_inference(opt.model_path, opt.eval_file, opt.base_dir)
    with open(opt.output, 'w') as f:
        f.write("\n".join(predict_result))
        f.flush()


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser("")
    parser_arg.add_argument('--model_path', type=str, default="deepseek")
    parser_arg.add_argument('--eval_file', type=str, default="./data/preprocessed_data/resdsql_dev.json")
    parser_arg.add_argument('--base_dir', type=str, default="./data/preprocessed_data/spider/database")
    parser_arg.add_argument('--output', type=str, default="./predict_result/sqlgpt.sql")
    opt = parser_arg.parse_args()
    main(opt)
