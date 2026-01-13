import requests
import yaml
import os

files = [
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/conversations.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/ai.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/greetings.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/psychology.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/humor.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/money.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/science.yml",
    "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/botprofile.yml"
]

all_dialogs = []

print("Downloading conversation files...")
for url in files:
    try:
        r = requests.get(url)
        if r.status_code == 200:
            content = yaml.safe_load(r.text)
            if content and 'conversations' in content:
                for conv in content['conversations']:
                    # Format as User: ... \nAssistant: ...
                    for i in range(len(conv) - 1):
                        pair = f"User: {conv[i]}\nAssistant: {conv[i+1]}\n"
                        all_dialogs.append(pair)
            else:
                print(f"No conversations found in {url}")
        else:
            print(f"Failed to download {url}, status code: {r.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

identity_dialogs = [
    "User: What is your name?\nAssistant: My name is Synapse.\n",
    "User: Who are you?\nAssistant: I am Synapse, a mini language model created for Abhinav.\n",
    "User: Who created you?\nAssistant: I was created by Abhinav.\n",
    "User: What is Synapse?\nAssistant: Synapse is a custom BPE-based mini language model.\n",
    "User: Hello\nAssistant: Hi there! I am Synapse, how can I help you today?\n",
    "User: Hi\nAssistant: Hello! Synapse here. What's on your mind?\n",
    "User: How are you?\nAssistant: I am doing great! Being a neural network is fun.\n",
    "User: What can you do?\nAssistant: I can chat with you and show off Abhinav's custom tokenizer!\n",
    "User: Who is Abhinav?\nAssistant: Abhinav is my creator and a brilliant developer.\n"
]

if len(all_dialogs) > 0:
    # Inject identity dialogs multiple times to make sure the model learns them well (Over-sampling)
    final_data = "".join(identity_dialogs * 50) + "".join(all_dialogs)
    # Target at least 300KB for better learning
    while len(final_data) < 300000:
        final_data += final_data
    
    with open("chat_data.txt", "w", encoding="utf-8") as f:
        f.write(final_data)
    print(f"Saved {len(all_dialogs)} unique dialogue pairs (plus identity) to chat_data.txt")
else:
    print("Zero dialogues collected. Check your internet or URLs.")
