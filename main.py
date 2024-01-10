import torch
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    #Let's place all the notebooks together
    speech_recognition_model = "openai/whisper-small"
    pipe = pipeline("automatic-speech-recognition", model=speech_recognition_model)
    #recognise the text
    recognised_text = pipe("direct_test_audio.mp3") 

    #Now we have to retrieve the intentionality
    path_file = "./intent_model/checkpoint-200/"
    tokenizer = AutoTokenizer.from_pretrained(path_file, local_files_only=True)
    inputs = tokenizer(recognised_text, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(path_file, local_files_only=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_intention = model.config.id2label[predicted_class_id]

    #If we are talking directly to a person, detect his name:
    if(predicted_intention == 'direct'):
        ner_model_name = "MMG/xlm-roberta-large-ner-spanish"
        ner_pipe = pipeline("token-classification", model=ner_model_name)
        entities = ner_pipe(recognised_text)
        print(entities)

    #Then we will just search in the (vector) database and answer accordingly