import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from transformers import pipeline
from marshmallow import ValidationError












billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)


checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


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

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.save_model("my_awesome_billsum_model")
model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_billsum_model")

text = news_articles.objects.all()
text_schema = news_articlesSchema(many=True)


test = "Actor Alec Baldwin has been indicted by a New Mexico grand jury on charges connected to the 2021 fatal shooting on the set of the movie “Rust,” new court documents show. Baldwin is charged with two counts of involuntary manslaughter. The first involuntary manslaughter charge against Baldwin is described in court documents as “negligent use of a firearm” and the second as involuntary manslaughter without due caution or circumspection, which is detailed as “an act committed with the total disregard or indifference to the safety of others.” Both are fourth degree felonies. “We look forward to our day in court,” Baldwin’s attorneys Luke Nikas and Alex Spiro told CNN in a statement on Friday. Baldwin’s attorney has previously insisted his client is not guilty. Involuntary manslaughter charges were dropped against Baldwin last year, with prosecutors saying in a statement at the time that they could not “proceed under the current time constraints and on the facts and evidence turned over by law enforcement in its existing form” due to “new facts” in the case. Cinematographer Halyna Hutchins was killed and director Joel Souza injured when a gun held by Baldwin fired a live round during a scene rehearsal on the set of the Western. Baldwin denied pulling the trigger on the weapon in a previous interview with CNN. The initial decision to drop the involuntary manslaughter charge against Baldwin in April of last year came after authorities learned the gun used in the shooting may have been modified, a law enforcement source told CNN. However, prosecutors said the case could be re-filed at a later date. In October, prosecutors said “additional facts have come to light that we believe show Mr. Baldwin has criminal culpability in the death of Halyna Hutchins” and signaled a grand jury would decide on recharging the actor. Armorer Hannah Gutierrez Reed also faces involuntary manslaughter charges in the case. She has pleaded not guilty and is slated for trial in February. The film’s assistant director, David Halls, was identified as the person who handed the firearm to Baldwin that fateful day. In 2023, he signed a plea agreement “for the charge of negligent use of a deadly weapon,” prosecutors said, noting that terms of the deal include six months of probation. Attorney Gloria Allred, who is representing Hutchins’ family in a civil lawsuit against Baldwin, responded to Friday’s development in a statement to CNN. “They continue to seek the truth in our civil lawsuit for them and they also would like there to be accountability in the criminal justice system,” Allred said. “We are looking forward to the criminal trial which will determine if he should be convicted for the untimely death of Halyna.” Matt Hutchins, cinematographer Halyna Hutchins’ widower, declined to comment when reached via phone by CNN. Two charges are levied against Baldwin in the indictment, but he could ultimately only be convicted of one. Should he be convicted, Baldwin could face up to 18 months in prison and a $5,000 fine."

def summarize_new(model, text):
    summarizer = pipeline("summarization", model="my_awesome_billsum_model")
    segments = [text[i:i+512] for i in range(0, len(text), 512)]
    summaries = [summarizer(segment)[0]['summary_text'] for segment in segments]


    print(" ".join(summaries))


#scrapy crawl -s MONGODB_URI="mongodb://localhost:27017/news_article" -s MONGODB_DATABASE="news_articles" scraping
