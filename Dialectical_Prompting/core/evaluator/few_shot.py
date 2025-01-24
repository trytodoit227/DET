import datetime
import os
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger
from functools import partial

from core.data.sampler.base import BaseSampler
from core.evaluator.base import BaseEvaluator


class FewShotEvaluator(BaseEvaluator):
    def post_init(self, few_shot_sampler: BaseSampler) -> None:
        self.few_shot_sampler = few_shot_sampler

    def evaluator_info(self) -> dict:
        if hasattr(self, 'few_shot_sampler'):
            sampler = self.few_shot_sampler.__class__.__name__
        else:
            sampler = "zero-shot"        
        return {
            'setting': self.setting_name,
            'llm': self.model.params,
            'dataset': self.data_conf,
            'sampler': sampler,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def scoring(self, data_point: dict) -> dict:
        try:
            few_shots = self.few_shot_sampler.draw_examples(data_point)
            result = self.model.classify(self.data_conf, data_point, few_shots)
        except Exception as e:
            result = ''
            logger.warning(repr(e))
        return {
            'correct': result.lower() == data_point['ans_text'].lower(),
            'output': result,
            'valid': result.lower() in [label.lower() for label in self.data_conf['data_label_list']] \
                     or result.lower() in data_point['text'],
        }

    def compute_valid_overall(self, valid_results: List[dict]) -> dict:
        return {
            'accuracy': sum([result['correct'] for result in valid_results]) / len(valid_results),
            'valid_num': len(valid_results),
        }
    
    def compute_overall(self, results: List[dict]) -> dict:
        return {
            'accuracy': sum([result['correct'] for result in results]) / len(results),
            'correct_num': sum([result['correct'] for result in results]),
            'total_num': len(results),
            'valid_num': sum([result['valid'] for result in results]),
        }
    def average_score(self, results: List[dict]) -> dict:
        for result in results:
            # compute the average score for all result sub-dicts
            evaluating_result = result['evaluating_result']
            scores = []
            for key, value in evaluating_result.items():
                if key.startswith('score_'):
                    scores.append(int(value))
            
            if scores:
                average_score = sum(scores) / len(scores)
            else:
                average_score = 1
            result['evaluating_result']['EQ_score'] = average_score
        return {
            'EQ_score' : sum([result['evaluating_result']['EQ_score'] for result in results]) / len(results)}
    
    def generate_dlexplain_feature(self, data_point: dict, template_dirs: List[str]) -> dict:
        """Generate explain feature for the given model and dataset.

        Args:
            data_point (dict): The data point to generate explain feature for.
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        
        try:
            result = self.model.dl_classify(template_dirs[0], self.data_conf, data_point)
            return {
                'correct': result.get('pred_label', '').lower() == data_point['ans_text'].lower(),
                'valid': result.get('pred_label') is not None and result['pred_label'].lower() in [label.lower() for label in self.data_conf['data_label_list']],
                'pred_result': result
            }
        except Exception as e:
            result = ''
            logger.warning(repr(e))
            return {
                'correct': False,
                'valid': False,
                'pred_result': result
            }

    def generate_dlexplain_feature(self, data_point: dict, template_dirs: List[str]) -> dict:
        """Generate explain feature for the given model and dataset.

        Args:
            data_point (dict): The data point to generate explain feature for.
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        
        try:
            result = self.model.dl_classify(template_dirs[0], self.data_conf, data_point)
            return {
                'correct': result.get('pred_label', '').lower() == data_point['ans_text'].lower(),
                'valid': result.get('pred_label') is not None and result['pred_label'].lower() in [label.lower() for label in self.data_conf['data_label_list']],
                'pred_result': result
            }
        except Exception as e:
            result = ''
            logger.warning(repr(e))
            return {
                'correct': False,
                'valid': False,
                'pred_result': result
            }    

    def generate_evaluating_score(self, data_point: dict, template_dirs: List[str]) -> dict:
        """Generate explain feature for the given model and dataset.

        Args:
            data_point (dict): The data point to generate explain feature for.
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        
        try:
            result = self.model.evaluating_score(template_dirs[2], self.data_conf, data_point)
            return {
                'evaluating_result': result
            }
        except Exception as e:
            result = ''
            logger.warning(repr(e))
            return {
                'evaluating_result': result
            } 
            
    def generate_dl_result(self, data_point: dict, template_dirs: List[str]) -> dict:
        """Generate explain feature for the given model and dataset.

        Args:
            data_point (dict): The data point to generate explain feature for.
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        grimoire = "Let's think step-by-step."
        try:
            few_shots = self.few_shot_sampler.draw_examples(data_point)
            result = self.model.classify(self.data_conf, data_point, few_shots, grimoire)
        except Exception as e:
            result = ''
            logger.warning(repr(e))
        return {
            'correct': result.lower() == data_point['ans_text'].lower(),
            'output': result,
            'valid': result.lower() in [label.lower() for label in self.data_conf['data_label_list']] \
                     or result.lower() in data_point['text'],
        }
    
    def generate_uni_explain_feature(self, data_point: dict, template_dirs: List[str]) -> dict:
        """Generate explain feature for the given model and dataset.

        Args:
            data_point (dict): The data point to generate explain feature for.
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        
        try:
            result = self.model.uni_classify(template_dirs[1], self.data_conf, data_point)
            return {
                'correct': result.get('pred_label', '').lower() == data_point['ans_text'].lower(),
                'valid': result.get('pred_label') is not None and result['pred_label'].lower() in [label.lower() for label in self.data_conf['data_label_list']],
                'pred_result': result
            }
        except Exception as e:
            result = ''
            logger.warning(repr(e))
            return {
                'correct': False,
                'valid': False,
                'pred_result': result
            }

            
    def run_dlexplain_feature(self, template_dirs: List[str]) -> list:
        """Generate explain feature for the given model and dataset.

        Args:
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        info = self.evaluator_info()

        generate_explain_feature_partial = partial(self.generate_dlexplain_feature, template_dirs=template_dirs)

        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(generate_explain_feature_partial, self.dataset),
                total = len(self.dataset),
                desc = self.model.params['model_name']
            ))

        results = [
            {**result, 'original_data': data_point}
            for result, data_point
            in zip(results, self.dataset)
        ]

        try:
            overall = self.compute_overall(results) if len(results) > 0 else {}
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()

        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output
    
    

    def run_uni_explain_feature(self, template_dirs: List[str]) -> list:
        """Generate explain feature for the given model and dataset.

        Args:
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        info = self.evaluator_info()

        generate_explain_feature_partial = partial(self.generate_uni_explain_feature, template_dirs=template_dirs)

        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(generate_explain_feature_partial, self.dataset),
                total = len(self.dataset),
                desc = self.model.params['model_name']
            ))

        results = [
            {**result, 'original_data': data_point}
            for result, data_point
            in zip(results, self.dataset)
        ]

        try:
            overall = self.compute_overall(results) if len(results) > 0 else {}
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()

        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output
    
    def run_evaluating_explanation(self, template_dirs: List[str]) -> list:
        info = self.evaluator_info()

        generate_evaluating_score_partial = partial(self.generate_evaluating_score, template_dirs=template_dirs)

        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(generate_evaluating_score_partial, self.dataset),
                total = len(self.dataset),
                desc = self.model.params['model_name']
            ))

        results = [
            {**result, 'original_data': data_point}
            for result, data_point
            in zip(results, self.dataset)
        ]

        try:
            EQ_score = self.average_score(results) if len(results) > 0 else {}
        except Exception as e:
            logger.warning(repr(e))
            EQ_score = dict()

        output_path = os.path.join("EQ-Evaluating-Output", self.output_name)
        self.save_output(output:={'info': info, 'EQ_score': EQ_score, 'results': results}, output_path = output_path)
        print(f'Output saved at {output_path}!')
        return output        
    
    def run_dl_prompt(self, template_dirs: List[str]) -> list:
        """Generate explain feature for the given model and dataset.

        Args:
            template_dirs (List[str]): The list of directories containing templates.
        Returns:
            dict: Output dictionary contains fields such as: info, results, etc.
        """
        info = self.evaluator_info()

        generate_explain_feature_partial = partial(self.generate_dl_result, template_dirs=template_dirs)

        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(generate_explain_feature_partial, self.dataset),
                total = len(self.dataset),
                desc = self.model.params['model_name']
            ))

        results = [
            {**result, 'original_data': data_point}
            for result, data_point
            in zip(results, self.dataset)
        ]

        try:
            overall = self.compute_overall(results) if len(results) > 0 else {}
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()

        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output
    
    
