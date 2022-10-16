export CUDA_VISIBLE_DEVICES=0
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data

for lang in "ar" "de" "el" "es" "hi"
do
  python ${PYTHONPATH}/utils/translate_squad_from_base.py ${DATAPATH}/squad/base_datasets/train-v2.0_${lang}_base.json ${lang} ${BASEPATH}/matcher_exp/non_enq_new/train_matcher_${lang} --output_dir ${DATAPATH}/squad/no_enq05_aug/ --match_thresh 0.5;
  python ${PYTHONPATH}/utils/translate_squad_from_base.py ${DATAPATH}/squad/base_datasets/dev-v2.0_${lang}_base.json ${lang} ${BASEPATH}/matcher_exp/non_enq_new/train_matcher_${lang} --output_dir ${DATAPATH}/squad/no_enq05_aug/ --original_dataset_path ${DATAPATH}/squad/dev-v2.0.json --match_thresh 0.5;
done