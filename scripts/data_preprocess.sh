# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/dev.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev.json" \
    --db_path "./database"\
    --target_type "sql"

python schema_item_classifier.py \
    --batch_size 32 \
    --device "0" \
    --seed 42 \
    --save_path "model/text2sql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_dev.json" \
    --output_filepath "./data/preprocessed_data/dev_with_probs.json" \
    --use_contents \
    --add_fk_info \
    --mode "eval"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/dev_with_probs.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_dev.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "eval" \
    --use_contents \
    --add_fk_info \
    --output_skeleton \
    --target_type "sql"

python gen_inference_data.py \
    --schemalinking_result "./data/preprocessed_data/resdsql_dev.jsonl" \
    --output "./data/preprocessed_data/inference_data.json"
