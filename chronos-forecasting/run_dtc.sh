CONFIG_FILE=evaluation/configs/zero-shot.yaml 
OUTPUT_FILE=evaluation/results/chronos-t5-small-zero-shot.csv
python3 evaluation/evaluate.py $CONFIG_FILE $OUTPUT_FILE --chronos-model-id "amazon/chronos-t5-small"  --batch-size=32 --device=cuda:0 --num-samples 20
