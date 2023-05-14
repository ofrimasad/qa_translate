export CUDA_VISIBLE_DEVICES=0
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=0,1

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw" "sv" "cs"
do
  python ${PYTHONPATH}/utils/translate_squad_from_base.py ${DATAPATH}/squad/base_v2.0/train-v2.0_${lang}_base.json ${lang} ${BASEPATH}/matcher_exp/enq_25_12_2022/train_matcher_${lang} --output_dir ${DATAPATH}/squad/enq_25_12_2022_v2.0/ --match_thresh 0.05 --from_en --both;
  python ${PYTHONPATH}/utils/translate_squad_from_base.py ${DATAPATH}/squad/base_v2.0/dev-v2.0_${lang}_base.json ${lang} ${BASEPATH}/matcher_exp/enq_25_12_2022/train_matcher_${lang} --output_dir ${DATAPATH}/squad/enq_25_12_2022_v2.0/ --match_thresh 0.05 --from_en --both;
done