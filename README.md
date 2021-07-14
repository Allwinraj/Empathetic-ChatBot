# Empathetic-ChatBot
Fine tuning GPT2 on the empathetic dataset to create an open-domain conversation model. REST API and Telegram bot are deployed separately from finely tuned model.
* The transformers library, which contains the latest NLP models (such as BERT, XLNet, GPT-2) will help us in our task. Microsoft’s DialoGPT was added to the Transformers model collection. This model is used.
* Hugginface configuration was used to train the model.
* REST API prototype was built using Python and the Flask web framework.
* python-telegram-bot package helped to create the telegram bot.

### USAGE:
1. Clone the Repository
2. **pip install - r requirements.txt**  to install package dependency
3. **train.py** to train and save the empathy model.
4. **interface_empathy.py** to test the model in console.
5. **app_empathy.py** to deploy the model in REST API.
6. **tele_empathy.py**  to crate the telegram bot.

### DATA:
Model training on publicly-available empathetic dialogue generation and EMPATHETICDIALOGUES from Allen School of Computer Science & Engineering, University of Washington and Facebook AI Research.
Dataset is cleaned and convert into conversation data.

### OUTPUT:
![GitHub Logo](/images/logo.png)

### REFERENCE:
a. https://research.fb.com/publications/towards-empathetic-open-domain-conversation-models-a-new-benchmark-and-dataset/

b. https://huggingface.co/


