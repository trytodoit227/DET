import json

import requests
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.llm.base import BaseLLM

class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0.01, max_new_tokens=4000, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(5), reraise=True)
    def _request(self, query: str) -> str:
        client = OpenAI(api_key=self.conf['api_key'], base_url=self.conf['base_url'])
        res = client.chat.completions.create(
            model=self.params['model_name'],
            messages=[{"role": "user", "content": query}],
            temperature=self.params['temperature'],
            max_tokens=self.params['max_new_tokens'],
            top_p=self.params['top_p'],
        )
        real_res = res.choices[0].message.content

        token_consumed = res.usage.total_tokens
        logger.info(
            f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res
