import torch
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from LMs.model import BertClassifier, BertClaInfModel
from data_utils.dataset import Dataset
from data_utils.load import load_data
from utils import init_path, time_logger, get_cur_time, fix
from transformers import AutoTokenizer, AutoModel

from transformers import TrainerCallback


class EpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # 打印state对象，查看它包含哪些字段
        print(f"Trainer state: {state}")

        # 获取并打印epoch值
        epoch = state.epoch
        print(f"Epoch {epoch} has finished.")
        model.set_epoch(epoch)



def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed
        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        self.output_dir = f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'

        # Preprocess data
        train_data = load_data(dataset=self.dataset_name, use_gpt=cfg.lm.train.use_gpt, mod='train') # import data
        test_data = load_data(dataset=self.dataset_name, use_gpt=cfg.lm.train.use_gpt, mod='test') # import data
        val_data = load_data(dataset=self.dataset_name, use_gpt=cfg.lm.train.use_gpt, mod='val')

        train_text = [d['text'] for d in train_data]
        val_text = [d['text'] for d in val_data]
        test_text = [d['text'] for d in test_data]
        train_label = [d['label'] for d in train_data]
        val_label = [d['label'] for d in val_data]
        test_label = [d['label'] for d in test_data]
        self.n_labels = len(set(test_label))
        self.train_num_nodes = len(train_label)
        self.test_num_nodes = len(test_label)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_token = tokenizer(train_text, padding=True, truncation=True, max_length=512) # 512 or 256
        val_token = tokenizer(val_text, padding=True, truncation=True, max_length=512)
        test_token = tokenizer(test_text, padding=True, truncation=True, max_length=512)

        self.train_dataset = Dataset(train_token, train_label) # formalize x and y
        self.val_dataset = Dataset(val_token, val_label)
        self.test_dataset = Dataset(test_token, test_label)

    def initialize_model(self):
        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

    @time_logger
    def train(self):
        self.initialize_model()
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

        # Define training parameters
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.train_num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=False,
            dataloader_drop_last=True,
            seed=fix
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            callbacks=[EpochCallback()]
        )

        self.trainer.model_init
        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        def create_memmap(file_name, shape):
            return np.memmap(init_path(f"{self.ckpt_dir}.{file_name}"),
                             dtype=np.float16,
                             mode='w+',
                             shape=shape)

        def evaluate_and_save(dataset, dataset_name):
            emb_shape = (len(dataset), self.feat_shrink if self.feat_shrink else 768)
            pred_shape = (len(dataset), self.n_labels) # 2 dimensions
            emb = create_memmap(f"{dataset_name}.emb", emb_shape)
            pred = create_memmap(f"{dataset_name}.pred", pred_shape)

            inf_model = BertClaInfModel(self.model, emb, pred, feat_shrink=self.feat_shrink)
            inf_model.eval()

            inference_args = TrainingArguments(
                output_dir=self.output_dir,
                do_train=False,
                do_predict=True,
                per_device_eval_batch_size=self.batch_size*8,
                dataloader_drop_last=False,
                dataloader_num_workers=1,
                fp16_full_eval=True,
            )

            trainer = Trainer(model=inf_model, args=inference_args,callbacks=[EpochCallback()])
            predictions = trainer.predict(dataset).predictions

            from utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

            def evaluator(preds, labels):
                return _evaluator.eval({
                    "y_true": torch.tensor(labels).view(-1, 1),
                    "y_pred": torch.tensor(preds).view(-1, 1),
                })

            metrics = evaluator(np.argmax(pred, -1), dataset.labels)
            accuracy = metrics['acc']
            f1_score = metrics['f1_score']

            print(f"{dataset_name} accuracy: {accuracy:.4f}", f"{dataset_name} f1_score: {f1_score:.4f}")
            return accuracy, f1_score, predictions
        
        def process_data(input_path, predictions, text_label_list, output_path, accuracy):
            """
            Processes the input data by updating the pred_result with predictions and calculates accuracy.

            Args:
                input_path (str): Path to the input JSON file.
                predictions (list): List of predictions.
                text_label_list (list): List of text labels corresponding to predictions.
                output_path (str): Path to save the processed JSON file.
                accuracy (float): Accuracy of the model on the data.

            Returns:
                None
            """
            with open(input_path, 'r') as f:
                data = json.load(f)

            results = []
            for i, pred in enumerate(predictions):
                result = {}
                pred_label = np.argmax(pred, -1)
                text_label = text_label_list[pred_label]
                if text_label == data["results"][i]["original_data"]["ans_text"]:
                    result["correct"] = True
                else:
                    result["correct"] = False
                result["pred_result"] = {}
                why_key = f"why_{text_label.replace(' ', '_')}"
                result["pred_result"]["why"] = data["results"][i]["pred_result"].get(why_key, "")
                result["pred_result"]["LM_pred_label"] = text_label
                result["original_data"] = data["results"][i]["original_data"]
                results.append(result)

            overall = data.get("overall", {})
            overall["LM_acc"] = accuracy
            data['results'] = results
            data['overall'] = overall

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)    
        

        train_acc, train_f1_score, train_predictions= evaluate_and_save(self.train_dataset, "train")
        val_acc, val_f1_score, val_predictions = evaluate_and_save(self.val_dataset, "val")
        test_acc, test_f1_score, test_predictions = evaluate_and_save(self.test_dataset, "test")
        

        input_train_path = f"/home/swufe/DHM/DET/Explanation_Guided_Training/dataset/{self.dataset_name}/exp-train.json"
        input_val_path = f"/home/swufe/DHM/DET/Explanation_Guided_Training/dataset/{self.dataset_name}/exp-val.json"
        input_test_path = f"/home/swufe/DHM/DET/Explanation_Guided_Training/dataset/{self.dataset_name}/exp-test.json"
        datetime = get_cur_time()
        output_train_path = f"/home/swufe/DHM/DET/Explanation_Guided_Training/output/{self.dataset_name}/exp-train-{datetime}.json"
        output_val_path = f"/home/swufe/DHM/DET/Explanation_Guided_Training/output/{self.dataset_name}/exp-val-{datetime}.json"
        output_test_path = f"/home/swufe/DHM/DET/Explanation_Guided_Training/output/{self.dataset_name}/exp-test-{datetime}.json"
        if not os.path.exists(output_train_path):
            os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
        with open(input_train_path, 'r') as f:
            train_data = json.load(f)
        text_label_list = train_data['info']['dataset']['data_label_list']

        directory = f"/home/swufe/DHM/DET/Explanation_Guided_Training/dataset/{self.dataset_name}"
        os.makedirs(directory, exist_ok=True)

        process_data(input_train_path, train_predictions, text_label_list, output_train_path, train_acc)
        process_data(input_val_path, val_predictions, text_label_list, output_val_path, val_acc)
        process_data(input_test_path, test_predictions, text_label_list, output_test_path, test_acc)


        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}, {'TrainF1': train_f1_score, 'ValF1': val_f1_score, 'TestF1': test_f1_score}
