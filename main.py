# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

def chat_template(question, context):
    """
    Creates a chat template for the Llama model.

    Args:
        question: The question to be answered.
        context: The context information to be used for generating the answer.

    Returns:
        A string containing the chat template.
    """

    template = f"""\
    <|im_start|>user
    Given the context, generate an SQL query for the following question
    context:{context}
    question:{question}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template

# Create a FastAPI instance
app = FastAPI()

# Define the request body using Pydantic BaseModel
class InputModel(BaseModel):
    question: str
    context: str

# Define the response model (optional)
class OutputModel(BaseModel):
    result: str

@app.post("/process", response_model=OutputModel)
async def process_input(input_data: InputModel):
    # Extract inputs
    question = input_data.question
    context = input_data.context

    # Process the inputs (example: sum)
    prompt = chat_template(question,context)
    tokenizer = AutoTokenizer.from_pretrained("hussnainahmedsaqib/tinyllama_text2sql")
    model = AutoModelForCausalLM.from_pretrained("hussnainahmedsaqib/tinyllama_text2sql")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    except:
        inputs = tokenizer(prompt, return_tensors="pt")
        continue

    output = model.generate(**inputs, max_new_tokens=512)

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    result = text

    return {"result": result}

# Run the application (use this if running the script directly)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




# Press the green button in the gutter to run the script.
# if __name__ == '__main__':

    # tokenizer = AutoTokenizer.from_pretrained("hussnainahmedsaqib/tinyllama_text2sql")
    # model = AutoModelForCausalLM.from_pretrained("hussnainahmedsaqib/tinyllama_text2sql")
    # # Prepare the Prompt.
    # question = ""
    # context = ""
    # prompt = chat_template(question, context)
    #
    # # Encode the prompt.
    # inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    #
    # # Generate the output.
    # output = model.generate(**inputs, max_new_tokens=512)
    #
    # # Decode the output.
    # text = tokenizer.decode(output[0], skip_special_tokens=True)
    #
    # # Print the generated SQL query.
    # print(text)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
