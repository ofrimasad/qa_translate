
export PYTHONPATH=/home/ofri/qa_translate

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw"
do
  python ${PYTHONPATH}/src/generate_matcher_dataset.py ${PYTHONPATH}/data/squad/train-v2.0_${lang}_base.json ${lang} --out_dir ${PYTHONPATH}/data/matcher_datasets_new --enq --num_phrases_in_sentence=10 --translated --hf;
  python ${PYTHONPATH}/src/generate_matcher_dataset.py ${PYTHONPATH}/data/squad/dev-v2.0_${lang}_base.json ${lang} --out_dir ${PYTHONPATH}/data/matcher_datasets_new --enq --num_phrases_in_sentence=10 --translated --hf --max_possible=20000;
done