import importlib
from itertools import product
from typing import List, Literal, Tuple
import os,random

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

def generate_dlexplain_feature(
    model: BaseLLM,
    data_name: str,
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
    setting = f'{dataname}-{model_name}-{seed}'
    logger.info(setting)
    
    train_dataset = Dataset(f'data/{data_name}/train.json', data_name=f'{data_name}_train').load()
    #add now
    train_dataset=random.sample(train_dataset, 9000)

    val_dataset = Dataset(f'data/{data_name}/val.json', data_name=f'{data_name}_val').load()
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').load()
    # void_dev_dataset = Dataset(f'data/{data_name}/void_dev.json', data_name=f'{data_name}_void_dev').load()

    # ─── Prepare Evaluator And Generate Explain Feature───────────────────────────────────────────────
    evaluator = FewShotEvaluator(
        model, data_conf, train_dataset, setting, process_num=PROCESS)
    evaluator.run_dlexplain_feature(template_dirs=template_dirs)
    evaluator = FewShotEvaluator(
        model, data_conf, val_dataset, setting, process_num=PROCESS)
    evaluator.run_dlexplain_feature(template_dirs=template_dirs)
    evaluator = FewShotEvaluator(
        model, data_conf, test_dataset, setting, process_num=PROCESS)
    evaluator.run_dlexplain_feature(template_dirs=template_dirs)

    # evaluator = FewShotEvaluator(
    #     model, data_conf, void_dev_dataset, setting, process_num=PROCESS)
    # evaluator.run_dlexplain_feature(template_dirs=template_dirs)

def expt_zero_shot(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
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
    setting = f'{dataname}-{model_name}-{seed}'
    logger.info(setting)
    
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').load()

    test_sampler = RandomSampler(
            test_dataset, cnt=0, cluster=True, identical=True, seed=seed)

    # ─── Prepare Evaluator And Generate Explain Feature───────────────────────────────────────────────
    evaluator = FewShotEvaluator(
        model, data_conf, test_dataset, setting, process_num=PROCESS)
    
    evaluator.post_init(few_shot_sampler=test_sampler)
    evaluator.run()

def expt_few_shot(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
    cnt: int = 0,
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
    setting = f'{dataname}-{model_name}-{seed}'
    logger.info(setting)

    train_dataset = Dataset(f'data/{data_name}/train.json', data_name=f'{data_name}_train').load()    
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').load()

    train_sampler = RandomSampler(
            train_dataset, cnt=cnt, cluster=True, identical=True, seed=seed)

    # ─── Prepare Evaluator And Generate Explain Feature───────────────────────────────────────────────
    evaluator = FewShotEvaluator(
        model, data_conf, test_dataset, setting, process_num=PROCESS)
    
    evaluator.post_init(few_shot_sampler=train_sampler)
    evaluator.run()

def expt_dl_prompt(
    model: BaseLLM,
    data_name: str,
    data_conf: dict,
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
    setting = f'{dataname}-{model_name}-{seed}'
    logger.info(setting)
    
    train_dataset = Dataset(f'data/{data_name}/train.json', data_name=f'{data_name}_train').load()
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').load()

    train_sampler = RandomSampler(
            train_dataset, cnt=0, cluster=True, identical=True, seed=seed)

    # ─── Prepare Evaluator And Generate Explain Feature───────────────────────────────────────────────
    evaluator = FewShotEvaluator(
        model, data_conf, train_dataset, setting, process_num=PROCESS)
    
    evaluator.post_init(few_shot_sampler=train_sampler)
    evaluator.run_dl_prompt(template_dirs=[])
    
    # evaluator = FewShotEvaluator(
    #     model, data_conf, test_dataset, setting, process_num=PROCESS)
    # evaluator.post_init(few_shot_sampler=train_sampler)
    # evaluator.run()

def generate_uni_explain_feature(
    model: BaseLLM,
    data_name: str,
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
    setting = f'{dataname}-{model_name}-{seed}'
    logger.info(setting)
    
    test_dataset = Dataset(f'data/{data_name}/test.json', data_name=f'{data_name}_test').load()

    # ─── Prepare Evaluator And Generate Explain Feature───────────────────────────────────────────────

    evaluator = FewShotEvaluator(
        model, data_conf, test_dataset, setting, process_num=PROCESS)
    evaluator.run_uni_explain_feature(template_dirs=template_dirs)
    


if __name__ == '__main__':
    experiment_conf = load_yaml_conf(EXPERIMENT_CONF_PATH)
    models, datanames, template_dirs = parse_experiment_conf(
        experiment_conf)
    data_confs = load_yaml_conf(DATA_CONF_PATH)

    for model, dataname in product(models, datanames):

        # ─── Add Experiments Here: ────────────────────────────────────

        # generate dialectical explain explanations
        generate_dlexplain_feature(
            model=model,
            data_name=dataname,
            data_conf=data_confs[dataname],
            template_dirs=template_dirs,
            seed=SEED,
        )

        # generate_uni_explain_feature(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     template_dirs=template_dirs,
        #     seed=SEED,
        # )
    print('ok')
        # # zero shot
        #
        # expt_zero_shot(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     seed=SEED,
        # )
        #
        # # few shot
        #
        # expt_few_shot(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     cnt=4,
        #     seed=SEED,
        # )
        #
        # # use prompt to direct the model to classify the data
        #
        # expt_dl_prompt(
        #     model=model,
        #     data_name=dataname,
        #     data_conf=data_confs[dataname],
        #     seed=SEED,
        # )