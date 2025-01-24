import importlib
from itertools import product
from typing import List, Literal, Tuple
import os
import json

from loguru import logger

from core.data.dataloader import DataLoader
from core.data.dataset import Dataset
from core.llm.base import BaseLLM
from core.evaluator.few_shot import FewShotEvaluator
from core.data.sampler.rand import RandomSampler
from core.load_conf import load_yaml_conf


DATA_CONF_PATH = './configs/data.yaml'
EXPERIMENT_CONF_PATH = './configs/experiment.yaml'
CACHE_GRIMOIRE_DIR = './.cache/grimoire_prompts'
CLASSIFIER_PATH = './.cache/classifier_model.pth'

# SentenceTransformer format!
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDINGS_FILENAME = 'ebd_train_all-mpnet-base-v2.pickle'
SIMILARITY_FILENAME = 'sims_all-mpnet-base-v2.pickle'

SEED = 22
EXPERIMENT_BATCH_SIZE = 500
# EXPERIMENT_TIMES = 3
EXPERIMENT_TIMES = 1
PROCESS = 10


def create_dataset_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    records = []
    for result in data['results']:
        text = result['original_data']['text']
        if 'pred_label' in result['pred_result']:
            classification = result['pred_result']['pred_label']
        elif 'LM_pred_label' in result['pred_result']:
            classification = result['pred_result']['LM_pred_label']
        else:
            classification = "None"
        
        if 'why' in result['pred_result']:
            explanation = result['pred_result']['why']
        else:
            explanation = "None"
        records.append({
            'text': text,
            'pred_label': classification,
            'explanation': explanation
        })
    
    return records

def parse_experiment_conf(experiment_conf: dict) -> Tuple[List[BaseLLM], List[str], BaseLLM]:
    """Instantiate LLMs and provide data names based on the provided configuration.

    Args:
        experiment_conf (dict): The configuration dictionary containing 
            information about LLMs and datasets.

    Returns:
        Tuple[List[BaseLLM], List[str], BaseLLM]: A tuple containing a list 
            of instantiated LLMs, the data names and the LLM used to generate grimoires.
    """

    # ─── Get Instantiated Llms ────────────────────────────────────────────

    llm_configs = experiment_conf.get('llm', {})
    instantiated_llms = []

    for llm_type, llm_list in llm_configs.items():
        if llm_list is None:
            continue

        for llm_dict in llm_list:
            if llm_dict is None:
                continue

            generator_name = list(llm_dict.keys())[0]
            parameters = list(llm_dict.values())[0]
            llm_module_name = f"core.llm.{llm_type}"
            llm_module = importlib.import_module(llm_module_name)
            generator_class = getattr(llm_module, generator_name)

            if parameters is not None:
                instantiated_llms.append(generator_class(**parameters))
            else:
                instantiated_llms.append(generator_class())

    # ─── Get Data Names ───────────────────────────────────────────────────

    datanames = experiment_conf.get('data')

    # ─── Get Template Directories ─────────────────────────────────────────
    template_dirs = experiment_conf.get('template_dirs')

    return instantiated_llms, datanames,  template_dirs

def evaluating_explanation_quality(
    model: BaseLLM,
    data_name: str,
    experiment_name: str,
    data_conf: dict,
    template_dirs: List[str],
    seed: int = 22,
) -> None:
    """Generate explain feature for the given model and dataset.

    Args:
        model (BaseLLM): The model to generate explain feature for.
        data_name (str): The name of the dataset.
        data_conf (dict): The configuration dictionary of the dataset.
        template_dirs (List[str]): The list of directories containing templates.
        grimoire_generator (BaseLLM): The model used to generate grimoires.
        seed (int): The random seed.
    """
    # ——— Prepare setting and data ————————————————————————————————————————
    model_name = model.params['model_name']
    setting = f'{dataname}-{experiment_name}-{model_name}-{seed}'
    logger.info(setting)

    # test_dataset_ourmethod = create_dataset_from_json(f'EQ-Evaluating-Input/{data_name}-ourmethod.json')
    test_dataset = create_dataset_from_json(f'EQ-Evaluating-Input/{data_name}-{experiment_name}.json')
    # with open(f'EQ-Evaluating-Input/{data_name}-{experiment_name}.json', 'r', encoding='utf-8') as file:
    #     test_dataset = json.load(file)
    # test_dataset = test_dataset[0:1]
    evaluator = FewShotEvaluator(
        model, data_conf, test_dataset, setting, process_num=PROCESS)
    evaluator.run_evaluating_explanation(template_dirs=template_dirs)
    # # test if input_dir is correct
    # print(f'EQ-Evaluating-Input/{data_name}-{experiment_name}.json is correct')

    


if __name__ == '__main__':
    experiment_conf = load_yaml_conf(EXPERIMENT_CONF_PATH)
    models, datanames, template_dirs = parse_experiment_conf(
        experiment_conf)
    data_confs = load_yaml_conf(DATA_CONF_PATH)

    for model, dataname in product(models, datanames):

        # ─── Add Experiments Here: ────────────────────────────────────

        # main results


        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "ourmethod",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )

        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "gpt-3.5-turbo-explain-then-predict",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        evaluating_explanation_quality(
            model=model,
            data_name=dataname,
            data_conf=data_confs[dataname],
            experiment_name = "gpt-4",#gpt-4/llama-3-8b
            template_dirs=template_dirs,
            seed=SEED,
        )
    print('ok')
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "lime",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "shap",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "gpt-4o-mini-explain",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        # # ablation study
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "dial",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "dial-orig",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "uni",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "orig-uni",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
        #
        # evaluating_explanation_quality(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     experiment_name = "uni-orig",
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
