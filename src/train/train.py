# fmt: off
import argparse
import logging
import os
import pprint
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.utils import write_squad_predictions
from farm.eval import Evaluator
from farm.infer import QAInferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from transformers import AutoModelForQuestionAnswering, AutoConfig

from utils.utils import AlephBertTokenizerFast

BASE_PATH = '/home/ofri/qa_translate/'


def question_answering(run_name: str,
                       lang_model: str,
                       train_filename: str,
                       dev_filename: str,
                       n_epochs: int,
                       batch_size: int,
                       lr: float,
                       evaluate_every: int,
                       no_token_types: bool):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="SQuAD_tr", run_name=run_name)
    ml_logger.log_params({'train_data': train_filename.split('/')[-1], 'dev_data': dev_filename.split('/')[-1]})

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    do_lower_case = False  # roberta is a cased model

    # 1.Create a tokenizer

    if no_token_types:
        tokenizer = AlephBertTokenizerFast.from_pretrained(lang_model)
    else:
        tokenizer = Tokenizer.load(tokenizer_class='BertTokenizer',
            pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    label_list = ["start_token", "end_token"]
    metric = "squad"
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=512,
        label_list=label_list,
        metric=metric,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=None,
        data_dir=Path(BASE_PATH + "data/squad"),
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    # NOTE: In FARM, the dev set metrics differ from test set metrics in that they are calculated on a token level instead of a word level
    data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Question Answering
    prediction_head = QuestionAnsweringHead()

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )


    optimizer_params = {"name": "TransformersAdamW", 'eps': 1e-08, "weight_decay": 0.00}
    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        optimizer_opts=optimizer_params,
        learning_rate=lr,
        schedule_opts={"name": "LinearWarmup", "warmup_proportion": 0.06},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device,
        grad_acc_steps=8
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )

    # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path(f'{BASE_PATH}runs/{run_name}')
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    # QA_input = [
    #         {
    #             "questions": ["Who counted the game among the best ever made?"],
    #             "text":  "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created."
    #         }]
    #
    # model = QAInferencer.load(save_dir, batch_size=40, gpu=True)
    # result = model.inference_from_dicts(dicts=QA_input)[0]
    #
    # pprint.pprint(result)
    # model.close_multiprocessing_pool()

    # 10. Do Inference on whole SQuAD Dataset & write the predictions file to disk
    # filename = os.path.join(processor.data_dir,processor.dev_filename)
    # result = model.inference_from_file(file=filename, return_json=False)
    # result_squad = [x.to_squad_eval() for x in result]

    # write_squad_predictions(
    #     predictions=result_squad,
    #     predictions_filename=filename,
    #     out_filename="predictions.json"
    # )

    # 11. Get final evaluation metric using the official SQuAD evaluation script
    # To evaluate the model's performance on the SQuAD dev set, run the official squad eval script
    # (farm/squad_evaluation.py) in the command line with something like the command below.
    # This is necessary since the FARM evaluation during training is done on the token level.
    # This script performs word level evaluation and will generate metrics that are comparable
    # to the SQuAD leaderboard and most other frameworks:
    #       python squad_evaluation.py path/to/squad20/dev-v2.0.json path/to/predictions.json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--lang_model', type=str, default="roberta-base")
    parser.add_argument('--train_filename', type=str, default="train-v2.0.json")
    parser.add_argument('--dev_filename', type=str, default="dev-v2.0.json")
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--evaluate_every', type=int, default=500)
    parser.add_argument('--no_token_types', action='store_true', help='set all token types to 0 (for alephbert)')

    opt = parser.parse_args()

    question_answering(run_name=opt.run_name,
                       lang_model=opt.lang_model,
                       train_filename=opt.train_filename,
                       dev_filename=opt.dev_filename,
                       n_epochs= opt.n_epochs,
                       batch_size=opt.batch_size,
                       lr=opt.lr,
                       evaluate_every=opt.evaluate_every,
                       no_token_types=opt.no_token_types)
