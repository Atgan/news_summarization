import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, pipeline 
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import logging
from .schema import NewsDocument, NewsDocumentSchema
from marshmallow import ValidationError
from datetime import date


class Summarizer:
    def __init__(self, model_checkpoint="t5-small", max_length=512, collection=None):
        self.model_checkpoint = model_checkpoint
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.rouge = evaluate.load("rouge")
        self.tokenized_billsum = None
        
        

    def load_dataset(self):
        billsum = load_dataset("billsum", split="ca_test")
        self.tokenized_billsum = billsum.train_test_split(test_size=0.2)

    def preprocess_function(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)

        labels = self.tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train_model(self):
        if self.tokenized_billsum is None:
            self.load_dataset()

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_checkpoint)

        training_args = Seq2SeqTrainingArguments(
            output_dir="my_awesome_billsum_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=False,
            push_to_hub=False,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_billsum["train"],
            eval_dataset=self.tokenized_billsum["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.save_model("my_awesome_billsum_model")

    def summarize_text(self, text):
        summarizer = pipeline("summarization", model="news_api/nlp_summarization/my_awesome_billsum_model")
        segment_length = 512
        segments = [text[i:i + segment_length] for i in range(0, len(text), segment_length)]

        last_segment_length = len(text) % segment_length
        if last_segment_length < segment_length / 2:
            segments[-1] = text[-last_segment_length:] 
        
        summaries = [summarizer(segment)[0]['summary_text'] for segment in segments]
        return " ".join(summaries)

    def summarize_and_update_db(self):
        today = date.today()
        today = str(today)

        try:
            documents = NewsDocument.objects.all()

            if not documents:
                print("No documents found.")
                return

            for document in documents:
                content = document.content
                summary = self.summarize_text(content)
                print(type(summary), flush=True)

                result = document.update(set__summary=summary, set__summary_date=today)

            return "Summarization and update process completed."
        except Exception as e:
            return f"Error during summarization and update process: {e}"