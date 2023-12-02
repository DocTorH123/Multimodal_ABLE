import sys

import torch
import util_functions
import huggingface_loader

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

GPT2_MAX_NUM_POSITIONS = 1024

def train_model(TRAINING_ITEM,
                model_path,
                dataset_path,
                **kwargs):

    # Load dataset and add image_path column to it
    print("Loading dataset... ", end="")
    dataset = huggingface_loader.load_dataset(dataset_path, "trimmed", "./Multimodal_ABLE/Descriptor/national_gallery_dataset")
    if kwargs["max_length"] == -1 :
        kwargs["max_length"] = max([len(dataset[TRAINING_ITEM][i]) for i in range(len(dataset))])
    print("Success!")
    print(" * Number of " + TRAINING_ITEM + " data :", len(dataset))
    print(" * Maximum length of " + TRAINING_ITEM + " :", kwargs["max_length"])
    if kwargs["max_length"] >= GPT2_MAX_NUM_POSITIONS :
        kwargs["max_length"] = GPT2_MAX_NUM_POSITIONS - 1
        dataset = dataset.filter(lambda example : len(example[TRAINING_ITEM]) <= kwargs["max_length"])
        print(" * Maximum length of some items in " + TRAINING_ITEM + " was too long! (>" + str(GPT2_MAX_NUM_POSITIONS) + "), So filtered them. (New size: " + str(len(dataset)) + ")")

    # Load model, feature extractor and tokenizer from pretrained model
    print("Loading model, feature extractor and tokenizer... ", end="")
    model = huggingface_loader.load_model(model_path, **kwargs)
    feature_extractor = huggingface_loader.load_feature_extractor(model_path)
    tokenizer = huggingface_loader.load_tokenizer(model_path)
    print("Success!")

    # Move model to device (cpu or gpu)
    device = kwargs["device"]
    print("Moving model to " + device + "... ", end="")
    if device in ["gpu", "cuda"]:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device[0:5] == "cuda:" :
        device = device if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Success!")
    print(" * Training device :", device)

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
        output_dir="./Multimodal_ABLE/Descriptor/" + TRAINING_ITEM + " model",
        logging_dir="./Multimodal_ABLE/Descriptor/" + TRAINING_ITEM + " model/logs",
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
        fp16=False if device == "cpu" else True,
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
    print("Saving best model... ", end="")
    trainer.save_model("./Multimodal_ABLE/Descriptor/" + TRAINING_ITEM + " model/best_model")
    print("Success!")

    return model, feature_extractor, tokenizer, kwargs["max_length"]

if __name__ == "__main__" :
    # Parse arguments as dictionary
    argv_dict = {}
    for arg_index in range(1, len(sys.argv)) :
        if sys.argv[arg_index][0] == '-' and not sys.argv[arg_index][1:].isdigit() :
            try :
                argv_dict[sys.argv[arg_index][1:]] = int(sys.argv[arg_index + 1])
            except ValueError :
                argv_dict[sys.argv[arg_index][1:]] = sys.argv[arg_index + 1]

    # Set hyper parameters
    TRAINING_ITEM = argv_dict.get("train", False)
    MAX_LENGTH = argv_dict.get("max_len", -1)  # -1 : auto calculate, else : set manually
    MIN_LENGTH = argv_dict.get("min_len", 0)
    NUM_BEAMS = argv_dict.get("num_beams", 4)
    DEVICE = argv_dict.get("device", "cpu")
    model_path = argv_dict.get("model_path", "nlpconnect/vit-gpt2-image-captioning")
    dataset_path = argv_dict.get("dataset_path", "Yumbang/uk-national-gallery-thumbnail-and-description")

    # Train model and save it
    if TRAINING_ITEM :
        print(" ---------- Training " + argv_dict['train'] + " inference model ---------- ")
        Model_info = train_model(TRAINING_ITEM, model_path, dataset_path,
                                 device=DEVICE, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, min_length=MIN_LENGTH)