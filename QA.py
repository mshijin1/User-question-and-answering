# importing nessesary libraries
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch

# loading the pre-trained model
model_name='distilbert-base-uncased-distilled-squad'
model=DistilBertForQuestionAnswering.from_pretrained(model_name)
tokenizer=DistilBertTokenizer.from_pretrained(model_name)

def answer_question(question, context, max_len=512):
    # Tokenize question
    question_tokens=tokenizer.tokenize(tokenizer.cls_token + question + tokenizer.sep_token)
    max_context_len=max_len-len(question_tokens)-1


    # splitting context into chunks
    context_tokens=tokenizer.tokenize(context)
    chunks=[context_tokens[i:i + max_context_len] for i in range(0, len(context_tokens),max_context_len)]

    best_answer=""
    best_score=float('-inf')

    for chunk in chunks:
        # prepare inputs
        input_tokens = question_tokens + chunk + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids = torch.tensor([input_ids])

        with torch.no_grad():
            outputs = model(input_ids)

        # Extract the start and end scores for the answer
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # find the tokens with the highest 'start' and 'end' scores
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Convert tokens to words 
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
        score = answer_start_scores[0, answer_start].item() + answer_end_scores[0, answer_end - 1].item()

        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer

# Define a function to interact with the chatbot
def chat_with_bot():
    print("Welcome to the Mahabharata chatbot ! Type 'exit' to end conversation.")
    while True:
        user_input=input("You: ")
        if user_input.lower()=="exit":
            print("Chatbot: Goodbye!")
            break
        else:
            #Provide a context
            context="""The Mahabharata tells the story of two sets of cousins, the Pandavas and the Kauravas, whose rivalry leads to a massive conflict. It begins with King Shantanu's marriage to Ganga, who gives birth to Bhishma. Bhishma's vow of celibacy sets the stage for the unfolding drama. The central conflict arises between the Pandavas, led by Yudhishthira, and the Kauravas, led by Duryodhana. The Pandavas face trials including exile and betrayal, while the Kauravas conspire against them.

The epic culminates in the great war of Kurukshetra, where the Pandavas, with Lord Krishna as their advisor, confront the Kauravas. This war represents a clash of righteousness and unrighteousness, with moral and spiritual dilemmas at its core. The Bhagavad Gita, delivered by Lord Krishna to Arjuna during the war, offers profound insights into duty and the nature of existence. Arjuna gains clarity and embraces his role as a warrior.

The Mahabharata concludes with the triumph of the Pandavas, establishing dharma and justice under Yudhishthira's rule. Despite the victory, it comes at a great cost, symbolizing the complexities of human nature and the consequences of conflict. The epic marks the beginning of a new era in Bharatavarsha's history, leaving behind a legacy of wisdom and moral teachings that continue to resonate through generations."""
            answer=answer_question(user_input, context)
            print("Chatbot:", answer)
# Start chatting with the bot
chat_with_bot()