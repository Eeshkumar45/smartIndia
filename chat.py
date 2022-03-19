import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intentFile = 'intents/intents3.json'
FILE = "data.pth"
bot_name = "chatbot"

# with open(intentFile, 'r') as json_data:
#     #intents = json.load(json_data)
#     pass

intents = json.load(open(intentFile, encoding='utf-8'))



data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")

    returnedAMsg = False

    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)


    _, predicted = torch.max(output,dim=1)

    mostMatchedTag = tags[predicted.item()]
    print("most matched tag = " ,mostMatchedTag)



    probs = torch.softmax(output,dim=1)

##################################################################################

    # list of output messages
    chatbotMessages = []
    nextMostMatch = [-1,-1] #index,percentage
    # add most matched in for loop

    for x in range(len(tags)):
        prob = probs[0][x]
        matchPercentage = prob.item()

        print("\tmatched with tag",tags[x],"{:.10f}".format(matchPercentage))

        #finding second most matched tag
        if x!= predicted.item() and nextMostMatch[1]<matchPercentage:
            nextMostMatch[0] = x
            nextMostMatch[1] = matchPercentage
        # if x==predicted.item():
        #     chatbotMessages.insert(0,random.choice(intents['intents'][x]['responses']))
        if prob.item() > 0.75:
            returnedAMsg  = True
            tag = tags[x]
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    chatbotMessage = random.choice(intent['responses'])
                    chatbotMessages.append(chatbotMessage)
                    print(f"{bot_name}: {chatbotMessage}")

    if not returnedAMsg:
        #check genuinenss and add to unanswered questions
        pass
        #print(f"{bot_name}: I do not understand...")




    print("next most matched tag =",tags[nextMostMatch[0]],". percentage =",nextMostMatch[1])
    if(nextMostMatch[1]>0.005):
            tag = tags[nextMostMatch[0]]
            print("netMostMatch is greater than 0.005 -> ",tag)

            cIntent = None
            for intent in intents["intents"]:
                if(intent["tag"]==tag):
                    cIntent = intent

            chatbotMessage = random.choice(cIntent['responses'])
            chatbotMessages.append(chatbotMessage)
            print(f"{bot_name}: {chatbotMessage}")

    print(chatbotMessages)
##################################################################################################
