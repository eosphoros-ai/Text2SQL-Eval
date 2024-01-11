import argparse
import copy
import json

prompt_temple = {
    "data_name": "spider",
    "id": 0,
    "db_id": "",
    "chat_rounds": [
        {
            "role": "system",
            "content": "",
            "chat_round_id": 0
        },
        {
            "role": "human",
            "content": "",
            "chat_round_id": 1
        },
        {
            "role": "bot",
            "content": "",
            "chat_round_id": 2
        }
    ]
}

def resdsql_insider(schemalinking_result):
    results = []
    with open(schemalinking_result) as f:
        redsql_trains = json.load(f)
        for redsql_train in redsql_trains:
            _, sql = redsql_train["output_sequence"].split("|", 1)
            prompt_temple["db_id"] = redsql_train["db_id"]
            prompt_temple["chat_rounds"][1]["content"] = redsql_train["input_sequence"]
            prompt_temple["chat_rounds"][2]["content"] = sql
            results.append(copy.deepcopy(prompt_temple))
    return results


def main(opt):
    prompts = resdsql_insider(opt.schemalinking_result)
    with open(opt.output, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
        f.flush()


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser("")
    parser_arg.add_argument('--schemalinking_result', type=str, default="")
    parser_arg.add_argument('--output', type=str, default="")
    opt = parser_arg.parse_args()
    main(opt)
