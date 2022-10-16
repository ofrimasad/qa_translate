
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw"
do
  python ${PYTHONPATH}/matcher/generate_matcher_dataset.py ${DATAPATH}/squad/base_datasets/train-v2.0_${lang}_base.json ${lang} --out_dir ${DATAPATH}/matcher_datasets_new --enq --num_phrases_in_sentence=10 --translated --hf;
  python ${PYTHONPATH}/matcher/generate_matcher_dataset.py ${DATAPATH}/squad/base_datasets/dev-v2.0_${lang}_base.json ${lang} --out_dir ${DATAPATH}/matcher_datasets_new --enq --num_phrases_in_sentence=10 --translated --hf --max_possible=20000;
done