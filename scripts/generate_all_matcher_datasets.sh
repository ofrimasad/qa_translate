
export PYTHONPATH=/home/ofri/qa_translate

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw"
do
  python ${PYTHONPATH}/src/matcher/generate_matcher_dataset.py ${PYTHONPATH}/data/squad/base_datasets/train-v2.0_${lang}_base.json ${lang} --out_dir ${PYTHONPATH}/data/matcher_datasets_new --enq --num_phrases_in_sentence=10 --translated --hf;
  python ${PYTHONPATH}/src/matcher/generate_matcher_dataset.py ${PYTHONPATH}/data/squad/base_datasets/dev-v2.0_${lang}_base.json ${lang} --out_dir ${PYTHONPATH}/data/matcher_datasets_new --enq --num_phrases_in_sentence=10 --translated --hf --max_possible=20000;
done