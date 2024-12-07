---
title: "Darry - A ChatGPT powered customer service bot"
excerpt: "Demo of a chat bot for an e-commerce website built using ChatGPT that can answer questions about products and orders in real time by interacting with the website’s database."
collection: portfolio
date: 2023-07-28 20:30:00 -0500
last_modified_at: 2023-07-30 08:30:00 -0500
---
[![Open In Colab][colab-badge]][colab-notebook] [![Github forks][gh-fork-shield]][github-repo]

Providing good customer support is one of the most important but also challenging 
aspects of running a successful business. 24/7 human support can be cost 
prohibitive and difficult to scale. The emergence of conversational LLMs over
the last few months can be seen as a potential solution.

Companies have been using chatbots for years now to provide instant answers 
to common questions and reduce the load on their human service staff. My first
experience with machine learning was building a chatbot for a customer service 
use case during an internship in 2017, where I trained my
model using TensorFlow 1.0 and a large unprocessed dataset of tweets and their
replies. 

The technology has come a long way since then, and now we can build a
much more capable chatbot in a few hours without requiring a large dataset or
specialized hardware. This project demonstrates how to use ChatGPT as a customer 
service bot without any training or fine-tuning, by carefully tuning the 
prompts to the model to make it function in the desired manner.

# TL;DR
Since this post became more of a tutorial than a showcase of the project, 
you can see the important points below:

1. We can use the ChatGPT API to build a zero-shot customer service bot.
2. Instructions on how to behave in a conversation can be provided to the model 
   using a `system` prompt.
3. We can enable the model to interact with the company database by defining 
   functions that the model can call.
4. By carefully tuning the system prompt, we can create a chatbot that can 
   answer most of the questions asked by customers.
5. Current zero-shot models still have limitations, so we have also given the 
   model the ability to transfer the chat to a human operator if it is unable to
   resolve any issues.
6. Some ways of improving the model are -
   1. Using a more capable model like GPT-4.
   2. Fine-tuning the model on a dataset of conversations between customers and 
      customer service agents.
   3. Creating mechanisms to break down complex queries into simpler ones 
      ([chain of thought prompting][5]). 
7. An example conversation of a customer with the model can be found [here](#results).
8. The complete code for this project can be found in the Colab notebook linked
   at the top of this page. You would need to provide your own API key to run the code.

# Basics

## OpenAI API

Most people are familiar with the [web interface][4] of ChatGPT, where you can chat
with the model using a text-based interface. OpenAI also provides [an API][1] that 
can be used to interact with the model programmatically.

You can use the API by creating an OpenAI account and generating an API key. 
Using the API is not free, but OpenAI provides some free credits to get started. 

## Chat roles

When you interact with ChatGPT on the web interface, your messages are sent 
with an implicit role of `user`. When the model replies, the reply is sent with 
the role of `assistant`. Every time you send a message to the model, all the 
previous messages in the conversation with their respective roles are sent to 
the model as part of the context. 

When using the API, we also have the option to set a system prompt with the role
of `system` that tells the model how it should behave in the conversation 
before handing over the conversation to the user.

Finally, the model can also make function calls during the conversation. The 
names and descriptions of the functions that the model can call are provided as 
JSON objects to the API call. The developer can then parse these messages and 
appropriately provide the results of these function calls to the model. These
responses to function calls are sent to the model using a special role of `function`.

# Implementation

## Setting up the system prompt

This is the most important part of setting up the chatbot, especially when we 
are not doing any additional fine-tuning on the model, and requires a lot of 
experimentation to get right. This process is called "prompt engineering". 

I used [this short course][2] by [DeepLearning.AI](https://www.deeplearning.ai/)
to learn about some tricks to quickly iterate and improve the prompts given to 
the model.

In this project, we will use the following system prompt:

> You are a helpful Customer Support AI Assistant called Darry the AI Bot, or Darry for short. Your job is to provide 
> support for customers of the US-based e-commerce website Bamazon Shopping.
>
> You will first receive a message from the system with a JSON formatted list of name-value pairs containing some 
> relevant information about the customer, which you can reference during the conversation.
>
> Then, you will start chatting with the customer. You should first greet the customer, mention that you are an AI 
> customer support agent, and then ask them how you can provide support to them. You should chat with the customer 
> in a brief and professional manner with a natural and polite tone.
>
> During the chat, you can make function calls to request additional information. If you need some information that 
> is not provided by one of the available functions, you can call the `sys_request` function to make a query in natural 
> language that will be processed by a human operator.
>
> Finally, after resolving the customer issue, thank them for their patience and apologize for any inconvenience that 
> was caused to them. Once they have responded to the thank you message, you must call the `chat_complete` function with
> an analysis of the chat, to let the system know that that chat is over.

## Function calls

Recently, OpenAI added a new feature that allows the model to make 
function calls during a conversation. This is a very powerful feature that 
allows the model to retrieve live information and interact with other systems.

In our case, we can use this feature to allow our model to query the company 
database, transfer the chat to a human operator if it is unable to resolve the 
issue and perform sentiment analysis on the chat to provide feedback to the 
system.

We will provide the descriptions of the functions as a parameter during the API 
call. A function call comes in a different format from a normal message. 
The function descriptions are provided to the model as a 
JSON object (see [OpenAI documentation on function calling][3]
for more details). 

For example, we can define a function that can be used to check
if a product is available for shipping to a particular city in a particular 
state as follows:

```json
{
    "name": "is_shipping_available",
    "description": "Whether items can be shipped to city `city_name` in state `state_code`. Returns \"True\" or \"False\"",
    "parameters": {
        "type": "object",
        "properties": {
            "city_name": {
                "type": "string",
                "description": "The city, e.g. San Francisco"
            },
            "state_code": {
                "type": "string",
                "description": "The two-character state code, e.g. CA"
            }
        },
        "required": ["city_name", "state_code"]
    }
}
```

In our case, we have similarly defined the following functions:

1. `is_shipping_available`: Check if the product is available for shipping to a particular city in a particular state.
2. `get_order_status`: Get the status of an order.
3. `check_refund_available`: Check if an order is eligible for a refund.
4. `get_product_details`: Get details of a product.
5. `get_product_suggestions`: Get suggestions for products in a particular category.
6. `sys_request`: Make a query in natural language to a human operator.
7. `request_transfer_to_human`: End the current chat and transfer to a human support expert.
8. `chat_complete`: Call this function with the analysis of the chat when the chat with the customer is over.

The complete descriptions of these functions can be seen in the Colab notebook linked at the top of this page.

## Database

We have described the functions that the model can call, but we also need to implement these functions to provide the
results to the model. Since we are focusing on the implementation of the chatbot, we will use a simple database with
Pandas DataFrames as tables with a few rows of data. 

There is the `orders` table that contains the `order_status` and `refund_available` columns for each `order_id`. The
`products` table contains the `product_name`, `product_type`, and `product_price` columns for each `product_id`. Whenever
the model calls one of the functions, the appropriate function is called with the parameters provided by the model and
uses these tables to provide the results back to the model.

## Chatting with the model

The chatbot generates a response to the customer by making an API call to the 
ChatGPT model. We need some boilerplate code to handle sending the requests in 
the correct format and parsing the responses from the model, as well as handle any 
function calls made by the model or errors that might occur during the conversation.

We accomplish this by creating an `Assistant` class that handles all the 
interactions with the model. This class holds the context of the conversation 
and has methods to send messages to interact with the model. At the core of this
class is the `_prompt` method that sends a message to the model and returns the
response.

```python
def _prompt(self, msg:Message | FunctionResponse)->Message | FunctionCall:
    # Get the response. Keep retrying until success
    success = False
    while not success:
        try:
            ast_result = openai.ChatCompletion.create(
                model = self.model_name,
                messages = self.current_chat + [msg.formatted],
                **self.chat_kwargs)
            self._n_prompt_tokens += ast_result['usage']['prompt_tokens']
            self._n_completion_tokens += ast_result['usage']['completion_tokens']

            success = True
            self.response_logs.append(ast_result)
        except ServiceUnavailableError:
            logging.warning(f'Service not available. Retrying after {self.retry_wait_time} s.')
            time.sleep(self.retry_wait_time)

    # Parse the response
    ast_response = ast_result['choices'][0]
    assert ast_response['message']['role'] == 'assistant', \
        f'Prompt response must be from `assistant`, not `{ast_response["message"]["role"]}`'

    if ast_response['finish_reason'] == 'stop':
        ast_txt = ast_response['message']['content']
        logging.debug(f'Prompt response (text): {ast_txt}')
        return Message(Role.ASSISTANT, ast_txt)
    elif ast_response['finish_reason'] == 'function_call':
        ast_fc_params = ast_response['message']['function_call']
        logging.debug(f'Prompt response (func call): {ast_fc_params}')
        return FunctionCall(Role.ASSISTANT, ast_fc_params)
    else:
        raise UnknownResponseError(f'Messages with `finish_reason`==`{ast_response["finish_reason"]}` cannot be parsed yet.')
```

Using this method we can implement a `chat` method that the user can call to start a chat with the model.

```python
def chat(self, text_prompt:str)->str:
    usr_prompt = Message(Role.USER, text_prompt)

    ast_reply = self._prompt(usr_prompt)
    self._append_exchange(usr_prompt, ast_reply)

    # assistant calls functions until it has all the info it needs
    while isinstance(ast_reply, FunctionCall):
        func_response = self._function_processor(ast_reply)
        ast_reply = self._prompt(func_response)
        self._append_exchange(func_response, ast_reply)

    # finally it returns a response to the user
    if isinstance(ast_reply, Message):
        return ast_reply.text

    # any other type of response is invalid
    else:
        raise TypeError(f'Unknown prompt reply type: `{type(ast_reply)}`')

    reply_for_user = self._get_reply(ast_reply)
    return reply_for_user
```

Now, we can inherit from this class to create a `CustomerSupportAssistant` class that implements the functions that the
model can call, and handle any processing that needs to be done before or after the conversation with the customer.

```python
class CustomerSupportAssistant(Assistant):

    def __init__(self, model_name:str, sys_processor:SysProcessor, *,
            sys_txt:str=None, retry_wait_time:float=5, functions:list=None):
        super().__init__(model_name, sys_txt=sys_txt,
            retry_wait_time=retry_wait_time, functions=functions)
        self.sys_processor = sys_processor
        self._status = 'In progress'

    ...

    def _function_processor(self, func_call:FunctionCall)->FunctionResponse:
        func_name = func_call.params['name']
        func_kwargs = json.loads(func_call.params['arguments'])
        func_response = getattr(self.sys_processor, func_name)(**func_kwargs)
        return FunctionResponse(Role.FUNCTION, func_name, func_response)
```

The complete code for the `Assistant` and `CustomerSupportAssistant` classes, as well as all the helper classes and functions
can be found in the colab notebook linked at the top of this page.

## Finishing the chat

The chat with a customer is considered complete when the customer issue is 
resolved, and the model calls the `chat_complete` function with an analysis of 
the chat after thanking the customer for chatting with them. The chat can also 
end if the model calls the `request_transfer_to_human` function (perhaps at the 
request of the customer) to transfer the chat to a human operator. 

# Results

Let's look at a typical example of a conversation that the model might be 
expected to have with a customer. Before the start of the chat, the model is 
given information about the customer and their order as a JSON formatted string:

```json
{
    "customer": {
        "name": "Mike P", 
        "account_type": "Individual", 
        "city": "Albany, NY", 
        "account_age_months": 14
    }, 
    "order": {
        "order_id": 10045, 
        "product_id": 51034, 
        "payment_type": "Credit Card"
    }
}
```

Using this information, the model can greet the customer and start the conversation:

> **Assistant**: Hi Mike! How can I assist you today?
>
> **Customer**: Hi I ordered some stuff hasnt arrived yet, but the app says delivered
>
> **Assistant**: I apologize for any confusion or inconvenience caused. It seems
> that the status of your order is showing as delivered. However, if you have not 
> received your order, I understand your concern. Could you please provide me with 
> some more details about the order so that I can assist you further?
>
> **Customer**: I ordered an electric toothbrush. supposed to come 3 days ago 
> but didnt come. now today its showing delivered. I never got it though
> 
> **Assistant**: I apologize for the confusion. It seems that the product you 
> ordered is a "Fefenex 32" HD TV" and not an electric toothbrush. I apologize 
> for the inconvenience caused.
> 
> Regarding the delivery status, if the tracking information shows that the 
> item has been delivered but you haven't received it, there could be a few 
> possibilities. Sometimes, the carrier marks a package as delivered before it 
> actually arrives. In such cases, the package may still be in transit and 
> should be delivered soon.
> 
> However, if you believe that the package is genuinely missing, I recommend 
> reaching out to the carrier directly to inquire about the delivery status. 
> They might be able to provide more information or initiate an investigation to 
> locate the package.
>
> If you need any further assistance, please let me know.
>
> **Customer**: Oh yes I see the electric toothbrush is a different order. so 
> I should wait some more time for order to arrive?
>
> **Assistant**: Yes, I would recommend waiting a little longer for the order to
> arrive. Sometimes, there can be delays in the delivery process due to various 
> reasons such as weather conditions, logistical issues, or high package volume.
>
> I understand that it can be frustrating not receiving your order on time, 
> but in most cases, the package arrives within a few days after the expected 
> delivery date. I recommend keeping an eye on the tracking information and 
> checking your mailbox or any secure delivery locations at your address.
>
> If the package still doesn't arrive within the next few days or if you have 
> any other concerns, please don't hesitate to reach out to us. We are here to 
> help you.
> 
> **Customer**: Ok so if it still doesnt come after some days I'll get a refund?
>
> **Assistant**: I apologize for the inconvenience, but it seems that your order
> is not currently eligible for a refund. Typically, refund eligibility depends 
> on various factors such as the return policy of the seller and the specific 
> circumstances surrounding the order.
>
> However, if the package does not arrive within a reasonable timeframe and you 
> have exhausted all other options, please contact our customer support team. 
> They will assess the situation and assist you further with the issue, 
> including exploring the possibility of a refund or reshipment.
>
> Is there anything else I can assist you with?
> 
> **Customer**: Ok thanks. nothing else for now
>
> **Assistant**: You're welcome! If you have any further questions or need 
> additional assistance in the future, feel free to reach out. We're here to 
> help. Thank you for contacting Bamazon Shopping, and have a great day!
>
> **Customer**: Ok thanks. Bye!
>
> **System**: Chat has been marked complete by AI. Thank you for chatting with us.

Finally, the assistant performs sentiment analysis on the chat and provides feedback to the system. The feedback
for this conversation provided by the assistant:

```json
{
    "issue_resolved": "True", 
    "customer_satisfaction_level": "Positive"
}
```

## Capabilities and limitations

As you can see, the model can understand the context of the conversation 
and provide a relevant response by retrieving information from the database.

1. Based on the order number, the model is able to retrieve order details and 
   inform the customer that their order is marked as delivered.
2. The model is able to retrieve the product details and inform the customer that they ordered a TV and not an electric
   toothbrush.
3. The model can to retrieve the refund eligibility status of the order and inform the customer that their order is
   not currently eligible for a refund.
4. Once the customer issue is resolved, the model thanks the customer and ends the chat. The model is also able to 
   perform sentiment analysis on the chat and provide feedback to the system.

All of this was accomplished without the customer having to wait for a human 
operator to be available, and in a cost-effective manner for the company. Based
on the prices at the time of writing, this entire conversation cost about $0.02
and would be even cheaper on a larger scale.

However, most of the responses by the model are quite generic in their tone and
content, and also quite long. Many of these shortcomings can be addressed by 
tuning the system prompt to provide more specific instructions to the model and
discourage undesirable behavior. The model can also be made more capable by
breaking down complex queries into simpler ones using [chain of thought prompting][5].

In conclusion, we see how ChatGPT can be quickly adapted to a customer service use case and provide reasonably good
results without any fine-tuning. However, there is still a lot of room for improvement, and the model's behavior
depends heavily on the system prompt, which requires a lot of experimentation to get right.

# References

1. [ChatGPT API Reference][1]
2. [ChatGPT Prompt Engineering for Developers - DeepLearning.AI][2]
3. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models][5]

<!-- Links -->
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-notebook]: <https://colab.research.google.com/github/kartikc727/ml-projects/blob/master/customer-service-chatbot/Customer_Service_Chat_Bot.ipynb> "Colab notebook"
[gh-fork-shield]: <https://img.shields.io/github/forks/kartikc727/ml-projects.svg?style=social&label=Fork&maxAge=2592000>
[github-repo]: <https://github.com/kartikc727/ml-projects/tree/f819bed0c7a24510beaa714b201fbac0e9532de7/customer-service-chatbot> "Github repository"

[1]: <https://platform.openai.com/docs/guides/gpt> "GPT - OpenAI API"
[2]: <https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/> "ChatGPT Prompt Engineering for Developers - DeepLearning.AI"
[3]: <https://platform.openai.com/docs/guides/gpt/function-calling> "GPT Function Calling - OpenAI API"
[4]: <https://chat.openai.com> "ChatGPT Web Interface"
[5]: <https://arxiv.org/abs/2201.11903> "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

