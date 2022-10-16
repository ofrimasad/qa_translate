# batch (total on all devices = 16)
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=2,3

for lang in "de" "el" "ru" "tr" "ar" "vi" "th" "hi" "zh"
do
  python ${PYTHONPATH}/utils/squad2_to_squad1.py \
  -i ${DATAPATH}/squad/v2_translated_aug/train-v2.0_${lang}_aug.json \
  -o ${DATAPATH}/squad/v1.1_translated_aug/train-v1.1_${lang}_aug.json
done

