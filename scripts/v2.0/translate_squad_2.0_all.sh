# batch (total on all devices = 16)
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=2,3

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw" "sv" "cs"
do
  python ${PYTHONPATH}/translate/translate_squad.py ${DATAPATH}/squad/train-v2.0.json ${lang} --replace --base_mode
  python ${PYTHONPATH}/translate/translate_squad.py ${DATAPATH}/squad/dev-v2.0.json ${lang} --replace --base_mode
done
