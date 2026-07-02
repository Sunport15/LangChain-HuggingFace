# # from transformers import pipeline

# # model = pipeline("summarization", model="facebook/bart-large-cnn")

# # response = model("The quick brown fox jumps over the lazy dog. The dog barked at the fox, but the fox was too quick and escaped into the forest.", max_length=50, min_length=25, do_sample=False)
# # print(response)
# # print(response[0]['summary_text'])

# from transformers import pipeline

# import torch

# #check if GPU is available 

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn" , device=0)

# ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
# A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
# Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
# In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
# Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
# 2010 marriage license application, according to court documents.
# Prosecutors said the marriages were part of an immigration scam.
# On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
# After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
# Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
# All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
# Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
# Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
# The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
# Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
# Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
# If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
# """
# print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))


from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0, max_length=256, truncation=True)

llm = HuggingFacePipeline(pipeline=model)

##create the prompt template 

template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

response = chain.invoke({"topic": topic, "age": age})
print(response)
