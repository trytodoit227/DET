import copy
import os
from abc import ABC, abstractmethod
from typing import List
import json
import re

from loguru import logger

from core.load_conf import load_yaml_conf


PROMPT_DIR = './prompts'
LLM_CONF_PATH = './configs/llm.yaml'


class BaseLLM(ABC):
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.01,
        max_new_tokens: int = 64,
        top_p: float = 0.9,
        top_k: int = 5,
        **more_params
    ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }
        # Load secret token / id / key / url of current LLM
        self.conf = load_yaml_conf(LLM_CONF_PATH)[self.__class__.__name__]
        if os.path.exists(os.path.expanduser(self.conf['api_key_save_file'])):
            self.conf['api_key'] = open(os.path.expanduser(self.conf['api_key_save_file']), 'r').read().strip()
        else:
            print(f"API key file not found at {os.path.expanduser(self.conf['api_key_save_file'])}")
        if os.path.exists(os.path.expanduser(self.conf['base_url_save_file'])):
            self.conf['base_url'] = open(os.path.expanduser(self.conf['base_url_save_file']), 'r').read().strip()
        else:
            print(f"Base URL file not found at {os.path.expanduser(self.conf['base_url_save_file'])}")
            

    @abstractmethod
    def _request(self, query: str) -> str:
        """Without further processing the response of the request; simply return the string."""
        return ''

    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join(PROMPT_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''

    def update_params(self, inplace: bool = True, **params):
        """Update parameters either in-place or create a new object with updated parameters.

        Args:
            inplace (bool, optional): If True, update parameters in-place. If False, create a new object with updated parameters. Default is True.
            **params: Keyword arguments representing parameters to be updated.

        Returns:
            BaseLLM: An instance of the class with updated parameters.

        Examples:
            Update parameters in-place:
            ```
            >>> obj.update_params(param1=20)
            >>> print(obj.params)
            {'param1': 20}
            ```

            Create a new object with updated parameters:
            ```
            >>> new_obj = obj.update_params(False, param1=20)
            >>> print(obj.params)
            {'param1': 10}
            >>> print(new_obj.params)
            {'param1': 20}
            ```
        """
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self._request(query)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

    # ─── Prompt Engineering ───────────────────────────────────────────────

    def classify(self, data_conf: dict, test_data: dict, few_shot_data: List[dict] = None, grimoire: str = None):
        """Classify the test data based on the specified task configuration (few-shot / zero-shot / with-grimoire).

        Args:
            data_conf (dict): A dictionary containing task configuration parameters, including 'task_type' and 'task_description'.
            test_data (dict): Test data dictionary with 'text' key representing the real question.
            few_shot_data (List[dict], optional): Few-shot examples used for the classification. Each dictionary in the list should contain 'text' and 'ans_text' keys.
            grimoire (str, optional): A string representing additional information for with-grimoire setting.

        Returns:
            str: The predicted label / result.

        Notes:
            - If 'few_shot_data' is provided, the function operates in the few-shot setting.
            - If 'few_shot_data' is not provided but 'grimoire' is provided, the function operates in the with-grimoire setting.
            - If both 'few_shot_data' and 'grimoire' are not provided, the function operates in the zero-shot setting.

        Raises:
            Any exceptions raised during the API request or processing.

        Example:
            ```
            >>> llm = AnImplementedLLMClass()
            >>> data_config = {'task_type': 'classification', 'task_description': 'Categorize text'}
            >>> test_data = {'text': 'Sample text for classification'}
            >>> few_shot_data = [{'text': 'Example 1', 'ans_text': 'Category A'}, {'text': 'Example 2', 'ans_text': 'Category B'}]
            >>> result = llm.classify(data_config, test_data, few_shot_data)
            ```
        """

        # ─── Construct The Few-shot Examples Or The Grimire ───────────

        if few_shot_data is not None and len(few_shot_data) > 0:  # few-shot setting
            examples_str = 'Here are some examples:\n\n' + '\n\n'.join([
                shot['text'] + '\nAnswer: ' + shot['ans_text']  # Few-shot examples
                for shot in few_shot_data
            ])
            examples_or_grimoire = examples_str
        else:  # zero-shot setting or with-grimoire setting
            grimoire_str = grimoire if grimoire is not None else ''
            examples_or_grimoire = grimoire_str

        # ─── Construct The Prompt From The Template ───────────────────

        prompt_template = self._read_prompt_template('classify.txt')
        prompt = prompt_template.format(
            task_type = data_conf['task_type'],
            task_description = data_conf['task_description'],
            examples_or_grimoire = examples_or_grimoire,
            question = test_data['text'] + '\nAnswer: ',
        )

        # ─── Query And Get The Returned Label ─────────────────────────

        res = self.safe_request(prompt)
        real_res = res.split('Answer: ')[-1].strip().replace('.', '').replace(',', '').lower()
        return real_res

    def exp_classify(self, template_dir: str, data_conf: dict, test_data: dict):

        # ─── Construct The Prompt From The Template ───────────────────

        data_label_list = data_conf['data_label_list']
        data_label_str = "/".join(data_label_list)
        formatted_labels = ['latent_concepts','cues'] + [f'support_{label.replace(" ", "_")}_reasoning' for label in data_label_list]
        formatted_labels.extend(['overall_reasoning', 'pred_label', 'pred_confidence'])
        fields_str = ", ".join(formatted_labels)

        prompt_template = self._read_prompt_template(template_dir)
        prompt = prompt_template.format(
            task_type = data_conf['task_type'],
            task_description = data_conf['task_description'],
            class_types = data_label_str,
            fields = fields_str,
            question = test_data['text'],
        )

        # ─── Query And Get The Returned Label ─────────────────────────
        res = self.safe_request(prompt)
        result = {}
        for label in formatted_labels:
            # Construct the regular expression to match the content within brackets
            pattern = re.compile(rf'"{label}":\s*(\[\[.*?\]\]|\[.*?\]|\{{.*?\}}|\".*?\"|\d+)', re.DOTALL)
            
            # Search for matches
            match = pattern.search(res)
            if match:
                key = label
                value = match.group(1).strip()
                result[key] = value
                if key == 'pred_label':
                    result['pred_label'] = value.replace('"', '')
                elif key == 'pred_confidence':
                    result['pred_confidence'] = float(value)
        return result
    
    def dl_classify(self, template_dir: str, data_conf: dict, test_data: dict):

        # ─── Construct The Prompt From The Template ───────────────────

        data_label_list = data_conf['data_label_list']
        compete_label_list = data_conf['compete_label_list']
        num_compete_labels = len(compete_label_list)
        data_label_str = "/".join(data_label_list)
        compete_label_str = "/".join(compete_label_list)
        # formatted_labels = [f'support_{label.replace(" ", "_")}_reasoning' for label in data_label_list]
        # formatted_labels.extend(['overall_reasoning', 'pred_label', 'pred_confidence'])
        # formatted_labels = ['pred_label', 'pred_confidence']
        # formatted_labels.extend([f'support_{label.replace(" ", "_")}_reasoning' for label in data_label_list])
        # formatted_labels.extend(['overall_reasoning'])
        formatted_labels = [f'why_{label.split(":")[0].replace(" ", "_")}' for label in compete_label_list]
        
        # evaluation_labels =[f'Evaluation_why_{label.replace(" ", "_")}' for label in data_label_list] + [f'{label.replace(" ", "_")}_confidence' for label in data_label_list]

        # evaluation_fields = ", ".join(evaluation_labels)

        explain_fields = ", ".join(formatted_labels)
        # fields_labels = formatted_labels + ['pred_label'] + ['why']# for explain first
        # fields_labels = formatted_labels + ['why'] + ['pred_label'] # for explain first
        # fields_labels = ['why'] + ['pred_label'] + formatted_labels + evaluation_labels
        fields_labels = formatted_labels
        fields_str = ", ".join(fields_labels)

        prompt_template = self._read_prompt_template(template_dir)
        prompt = prompt_template.format(
            task_type = data_conf['task_type'],
            task_information = data_conf['task_information'],
            task_description = data_conf['task_description'],
            input_description = data_conf['input_description'],
            category_description = data_conf['category_description'],
            class_types = data_label_str,
            num_classifier = num_compete_labels,
            compete_types = compete_label_str,
            fields = fields_str,
            explain_fields = explain_fields,
            # evaluation_fields = evaluation_fields,
            question = test_data['text'],
        )

        # ─── Query And Get The Returned Label ─────────────────────────
        res = self.safe_request(prompt)
        result = {}
        for label in fields_labels:
            # Construct the regular expression to match the content within brackets
            pattern = re.compile(rf'"{label}":\s*(\[\[.*?\]\]|\[.*?\]|\{{.*?\}}|\".*?\"|\d+\.\d+|\d+)', re.DOTALL)
            
            # Search for matches
            match = pattern.search(res)
            if match:
                key = label
                value = match.group(1).strip()
                result[key] = value
                if key == 'pred_label':
                    result['pred_label'] = value.replace('"', '')
                elif key.endswith("_confidence"):
                    result[key] = float(value)
        return result
    
    def uni_classify(self, template_dir: str, data_conf: dict, test_data: dict):

        # ─── Construct The Prompt From The Template ───────────────────

        data_label_list = data_conf['data_label_list']
        data_label_str = "/".join(data_label_list)
        formatted_labels = []
        formatted_labels.extend(['why', 'pred_label'])
        fields_str = ", ".join(formatted_labels)

        prompt_template = self._read_prompt_template(template_dir)
        prompt = prompt_template.format(
            task_type = data_conf['task_type'],
            task_description = data_conf['task_description'],
            class_types = data_label_str,
            fields = fields_str,
            question = test_data['text'],
        )

        # ─── Query And Get The Returned Label ─────────────────────────
        res = self.safe_request(prompt)
        result = {}
        for label in formatted_labels:
            # Construct the regular expression to match the content within brackets
            pattern = re.compile(rf'"{label}":\s*(\[\[.*?\]\]|\[.*?\]|\{{.*?\}}|\".*?\"|\d+)', re.DOTALL)
            
            # Search for matches
            match = pattern.search(res)
            if match:
                key = label
                value = match.group(1).strip()
                result[key] = value
                if key == 'pred_label':
                    result['pred_label'] = value.replace('"', '')
        return result
    
    def evaluating_score(self, template_dir: str, data_conf: dict, test_data: dict):

        # ─── Construct The Prompt From The Template ───────────────────

        formatted_labels = ['score_clarity', 'score_relevance' , 'score_completeness', 'score_consistency', 'score_credibility', 'justification_clarity', 'justification_relevance', 'justification_completeness', 'justification_consistency', 'justification_credibility']
        fields_str = ", ".join(formatted_labels)

        prompt_template = self._read_prompt_template(template_dir)
        prompt = prompt_template.format(
            fields = fields_str,
            original_text = test_data['text'],
            pred_label = test_data['pred_label'],
            explanation = test_data['explanation']
        )

        # ─── Query And Get The Returned Label ─────────────────────────
        res = self.safe_request(prompt)
        result = {}
        for label in formatted_labels:
            # Construct the regular expression to match the content within brackets
            pattern = re.compile(rf'"{label}":\s*(\[\[.*?\]\]|\[.*?\]|\{{.*?\}}|\".*?\"|\d+)', re.DOTALL)
            
            # Search for matches
            match = pattern.search(res)
            if match:
                key = label
                value = match.group(1).strip()
                result[key] = value
        return result    
    
    def generate_grimoire(self, template_dir: str, data_conf: dict, data: List[dict]=None, example_str: str=None):
        if example_str is None and data is not None:
            if len(data) == 0:
                examples_str = ''
            else:
                examples_str = 'Samples:\n\n' + '\n\n'.join([
                    '\nText: ' + data_point['text'] + '\nLabel: ' + data_point['ans_text']
                    for data_point in data
                ])
        elif example_str is not None and data is None:
            examples_str = example_str
        elif example_str is not None and data is not None:
            examples_str = example_str + '\n\n' + '\n\n'.join([
                '\nText: ' + data_point['text'] + '\nLabel: ' + data_point['ans_text']
                for data_point in data
            ])
        else:
            examples_str = ''


        profound_prompt_template = self._read_prompt_template(template_dir)

        profound_prompt = profound_prompt_template.format(
            task_description=data_conf['task_description'],
            examples=examples_str,
        )
        profound_grimoire = self.safe_request(profound_prompt)

        grimoire = profound_grimoire

        return grimoire
