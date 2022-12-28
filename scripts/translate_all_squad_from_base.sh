export CUDA_VISIBLE_DEVICES=0
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN"
do
  python ${PYTHONPATH}/utils/translate_squad_from_base.py ${DATAPATH}/squad/base_v1.1/train-v1.1_${lang}_base_n.json ${lang} ${BASEPATH}/matcher_exp/enq_25_12_2022/train_matcher_${lang} --output_dir ${DATAPATH}/squad/enq_25_12_2022/ --match_thresh 0.05 --from_en --both;
  python ${PYTHONPATH}/utils/translate_squad_from_base.py ${DATAPATH}/squad/base_v1.1/dev-v1.1_${lang}_base_n.json ${lang} ${BASEPATH}/matcher_exp/enq_25_12_2022/train_matcher_${lang} --output_dir ${DATAPATH}/squad/enq_25_12_2022/ --original_dataset_path ${DATAPATH}/squad/dev-v1.1.json --match_thresh 0.05 --from_en --both;
done