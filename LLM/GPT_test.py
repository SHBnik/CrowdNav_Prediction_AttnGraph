from openai import OpenAI
import LLM.My_Key as My_Key

client = OpenAI(api_key=My_Key.key)

response = client.chat.completions.create(
    model="text-davinci-002",  # Specify LLMA model here
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020?"},
    ],
)
print(response)
print(response.choices[0].message.content)
