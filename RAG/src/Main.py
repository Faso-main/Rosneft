import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
import faiss
from transformers import (
    RagTokenizer, 
    RagRetriever, 
    RagSequenceForGeneration,
    Trainer, 
    TrainingArguments
)

# 1. Подготовка русскоязычных данных (пример)
# В реальном проекте замените на свои данные
russian_documents = [
    {"text": "Москва - столица России, крупнейший город страны."},
    {"text": "Пушкин - великий русский поэт, автор 'Евгения Онегина'."},
    {"text": "Вторая мировая война закончилась в 1945 году."},
    # Добавьте свои документы
]

russian_qa = [
    {"question": "Какая столица России?", "answer": "Москва"},
    {"question": "Кто написал 'Евгения Онегина'?", "answer": "Пушкин"},
    {"question": "Когда закончилась Вторая мировая война?", "answer": "в 1945 году"},
    # Добавьте свои QA пары
]

# Создаем Dataset объекты
doc_dataset = Dataset.from_list(russian_documents)
qa_dataset = Dataset.from_list(russian_qa)

# 2. Создание индекса для поиска
encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Кодируем документы
doc_texts = [doc['text'] for doc in russian_documents]
doc_embeddings = encoder.encode(doc_texts, show_progress_bar=True)

# Создаем FAISS индекс
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

# Сохраняем индекс
faiss.write_index(index, "russian_docs.index")

# 3. Инициализация RAG модели
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", use_fast=False)

# Создаем кастомный ретривер
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    indexed_dataset=doc_texts,
    embeddings=encoder,
    index=index
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq", 
    retriever=retriever
)
model.config.index_name = "custom"

# 4. Подготовка данных для обучения
def process_data(examples):
    questions = [q.strip() for q in examples["question"]]
    answers = [a.strip() for a in examples["answer"]]
    
    inputs = tokenizer(
        questions, 
        max_length=128, 
        truncation=True, 
        padding="max_length",
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            answers,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids
    
    inputs["labels"] = labels
    return inputs

train_dataset = qa_dataset.map(process_data, batched=True)

# 5. Обучение модели
training_args = TrainingArguments(
    output_dir="./russian_rag",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# 6. Функция для вопросов
def ask_question(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Пример использования
print(ask_question("Какая столица России?"))  # Должно вывести: Москва
print(ask_question("Кто написал 'Евгения Онегина'?"))  # Должно вывести: Пушкин