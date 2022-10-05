export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/ofri/qa_translate

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw"
do
  python ${PYTHONPATH}/src/utils/translate_squad_from_base.py ${PYTHONPATH}/data/squad/base_datasets/train-v2.0_${lang}_base.json ${lang} ${PYTHONPATH}/matcher_exp/enq_new/train_matcher_${lang} --output_dir ${PYTHONPATH}/data/squad/enq03/ --match_thresh 0.3;
  python ${PYTHONPATH}/src/utils/translate_squad_from_base.py ${PYTHONPATH}/data/squad/base_datasets/dev-v2.0_${lang}_base.json ${lang} ${PYTHONPATH}/matcher_exp/enq_new/train_matcher_${lang} --output_dir ${PYTHONPATH}/data/squad/enq03/ --original_dataset_path ${PYTHONPATH}/data/squad/dev-v2.0.json --match_thresh 0.3;
done