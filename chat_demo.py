"""LangChain + LangGraph demo: a conversation with memory. Follow-up questions ("What about their pizza?") are understood.

    python chat_demo.py
"""
from transformers.utils import logging as transformers_logging

from reviewsense.chat.conversation import Conversation
from reviewsense.chat.evaluate import score_followups
from reviewsense.data import load_reviews
from reviewsense.safety.safe_assistant import SafeAssistant

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable
transformers_logging.set_verbosity_error()          # and generation-settings notices

conversation = Conversation(SafeAssistant(load_reviews()))


def say(question, session="demo"):
    result = conversation.ask(session, question)
    print(f"You: {question}")
    if result["standalone"] != question or result["business"]:
        print(f"     (understood as: {result['standalone']!r}, searching {result['business'] or 'all restaurants'})")
    print(f"Bot: {result['answer']}\n")


# 1. A conversation: each follow-up refers back to the restaurant or person discussed before.
for question in ["Is Bella Napoli expensive?", "What about their pizza?", "Is parking easy there?",
                 "Where can I get spicy soup?", "Who is the chef at Golden Dragon?", "What did he recommend?"]:
    say(question)

# 2. LangGraph's checkpointer stored every turn of the conversation (LangChain message objects).
print(f"Stored: {len(conversation.history('demo'))} messages in conversation 'demo'\n")

# 3. Measured on hand-made conversations (data/followup_gold.yaml).
print("Follow-up resolution:", score_followups(conversation))

# 4. Your turn: a new session (press Enter on an empty line to stop).
while (question := input("\nYou (Enter to quit): ").strip()):
    say(question, session="you")
