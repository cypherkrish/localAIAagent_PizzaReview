from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template =  """
You are a expert in answerting question about the Peter restarent.

Here are the reviews: {reviews}

Here is a qustion to ask: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n --------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == 'q':
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({ "reviews":reviews, "question": "What is the best pizza place in the town?"})

    print(result)


