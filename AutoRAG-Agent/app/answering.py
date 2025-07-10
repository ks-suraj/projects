
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def load_llm(model_name="google/flan-t5-small", device="cpu"):
    """Load the lightweight LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def generate_answer(query, retrieved_chunks, tokenizer, model, device="cpu"):
    """Generate answer from retrieved chunks and query using LLM."""
    # Combine chunks and query into prompt
    context = "\n".join(retrieved_chunks)
    prompt = f"""You are an expert assistant. Use the following context to answer the question as clearly and concisely as possible. Focus only on information in the context.

Context:
{context}

Question: {query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=3,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
