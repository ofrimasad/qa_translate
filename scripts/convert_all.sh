# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate
export CUDA_VISIBLE_DEVICES=2,3

for lang in "de" "el" "ru" "tr" "ar" "vi" "th" "hi" "zh"
do
  python ${PYTHONPATH}/src/utils/squad2_to_squad1.py \
  -i ${PYTHONPATH}/data/squad/v2_translated_aug/train-v2.0_${lang}_aug.json \
  -o ${PYTHONPATH}/data/squad/v1.1_translated_aug/train-v1.1_${lang}_aug.json
done

