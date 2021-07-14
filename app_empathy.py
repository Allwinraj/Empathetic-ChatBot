from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import requests
from flask import Flask, request,jsonify
import argparse


tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('model')


app = Flask(__name__)
def arg_():
    parser = argparse.ArgumentParser(
            description='starts server to serve an agent')
    parser.add_argument(
            '-p', '--port',
            type=int,
            default=5000,
            help="port to run the server at")
    return parser

@app.route('/empathy_convese',methods=['POST'])
def respond_tBot():
    response = request.json['Text']
    #chatid = request.json['chat_id']
    print(response)
    step = 1

    
    new_user_input_ids = tokenizer.encode(response + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step < 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=200,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature = 0.8
    )
    
    empathy_result = (tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    return jsonify({'Prediction':empathy_result})

if __name__ == '__main__':
    arg_parser = arg_()
    app.run(host='0.0.0.0', port=8009, threaded=True)


