# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate
export CUDA_VISIBLE_DEVICES=2,3

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN"
do
  python ${PYTHONPATH}/src/translate/translate_squad.py /home/ofri/qa_translate/data/squad/train-v2.0.json ${lang} --replace --skip_impossible --base_mode
  python ${PYTHONPATH}/src/translate/translate_squad.py /home/ofri/qa_translate/data/squad/dev-v2.0.json ${lang} --replace --skip_impossible --base_mode
done

