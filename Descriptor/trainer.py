import torch
import util_functions
import huggingface_loader

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

def train_model(TRAINING_ITEM,
                model_path,
                datasets_path,
                **kwargs):

    # Load dataset and add image_path column
    print("Loading dataset... ", end="")
    dataset = huggingface_loader.load_dataset(datasets_path, "trimmed", "./national_gallery_dataset")
    if kwargs["max_length"] == -1 :
        kwargs["max_length"] = max([len(dataset[TRAINING_ITEM][i]) for i in range(len(dataset))])
    print("Success!")
    print(" * Maximum length of " + TRAINING_ITEM + " :", kwargs["max_length"])

    # Load model, feature extractor and tokenizer
    print("Loading model, feature extractor and tokenizer... ", end="")
    model = huggingface_loader.load_model(model_path, **kwargs)
    feature_extractor = huggingface_loader.load_feature_extractor(model_path)
    tokenizer = huggingface_loader.load_tokenizer(model_path)
    print("Success!")

    # Move model to GPU if available
    print("Moving model to GPU... ", end="")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model.to(device)
    print("Success!")

    # Preprocess dataset
    print("Preprocessing dataset... ", end="")
    processed_dataset = dataset.map(
        function=util_functions.preprocess_fn,
        batched=True,
        fn_kwargs={"max_target_length": kwargs["max_length"], "label_subject": TRAINING_ITEM,
                   "tokenizer": tokenizer, "feature_extractor": feature_extractor},
        remove_columns=dataset.column_names
    ).train_test_split(test_size=0.3)
    print("Success!")
    print(" * Number of training data :", len(processed_dataset['train']))
    print(" * Number of test data :", len(processed_dataset['test']))

    # Set training arguments
    print("Setting training arguments... ", end="")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./" + TRAINING_ITEM + " model",
        logging_dir="./" + TRAINING_ITEM + " model/logs",
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        predict_with_generate=True,
        evaluation_strategy="steps",
        num_train_epochs=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_steps=10,
        save_steps=120,
        eval_steps=120,
        warmup_steps=1000,
        save_total_limit=100,
        fp16=True,
        overwrite_output_dir=True,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        use_cpu=True if device == "cpu" else False,
    )
    print("Success!")

    # Set trainer
    print("Setting trainer... ", end="")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=feature_extractor,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
        data_collator=default_data_collator,
    )
    print("Success!")

    # Train model
    print("Training model... ")
    trainer.train()
    print("Finish!")

    # Save model
    print("Saving model... ", end="")
    trainer.save_model("./" + TRAINING_ITEM + " model/best_model")
    print("Success!")

    return model, feature_extractor, tokenizer, kwargs["max_length"]