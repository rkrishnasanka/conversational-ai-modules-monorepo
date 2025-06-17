from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

local_model_dir = "./.model_cache/gpt2"

tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(local_model_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
hf_pipeline = HuggingFacePipeline(pipeline=pipe)


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf_pipeline

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))
