"""Multi-turn conversations with LangChain and LangGraph: chat memory, and follow-up questions rewritten so the safe
assistant (Part 10) understands them.

  User: Is Bella Napoli expensive?        -> answered as before
  User: What about their pizza?           -> "their" = Bella Napoli: search only Bella Napoli's reviews for pizza

Each message runs through a small LangGraph graph:

    guard --(blocked)--> END
      |
      +--(allowed)--> resolve --> answer --> END

  guard   - Part 10's input guard on the user's own words. Checked BEFORE rewriting: an LLM rewrite of an attack
            could hide it, and the rewrite step would itself read the attack.
  resolve - only follow-ups ("their", "it", "there", "and the staff?") are rewritten, by a LangChain chain
            (ChatPromptTemplate with the chat history -> chat model -> text): "What did he recommend?" -> "What did
            Li Wei recommend?". The rewrite is checked before use (still one short question?), else the original is
            kept. The restaurant is the one named in the question or, for a follow-up, the one the conversation was
            about: the small LLM leaves some follow-ups unchanged, but the right reviews are still searched.
            Measured on data/followup_gold.yaml (10 conversations, real answers in the history): 80% -> 100%.
  answer  - the safe assistant answers the standalone question.
The graph's checkpointer stores each conversation (thread) after every step. InMemorySaver keeps them in this
process; in production a database checkpointer (Postgres, Redis) lets any server continue any conversation.
LangChain wraps the same LLM object as Haystack (one copy in memory). (LangChain's older RunnableWithMessageHistory
is deprecated in favour of this.)
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from ..linguistics.core import mentioned_business
from ..safety.input_guard import MESSAGES
from ..safety.safe_assistant import SafeAssistant

# The standard LangChain layout (earlier turns as chat messages) rewrote 6 of 8 follow-ups correctly; the same
# conversation pasted in as one block of text rewrote 4. "/no_think" switches off Qwen3's reasoning output.
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the conversation, rewrite the user's latest question so it can be understood on its own, "
               "naming the restaurant it is about. If it already stands alone, return it unchanged. "
               "Reply with only the question. /no_think"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
# Words that point back to something earlier in the conversation.
FOLLOW_UP = re.compile(r"\b(it|its|they|their|them|there|he|she|his|her|this place|that place)\b"
                       r"|^(and|what about|how about)\b", re.IGNORECASE)
THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def remove_thinking(text: str) -> str:
    """Qwen3 still writes an empty <think></think> block when reasoning is switched off."""
    return THINKING.sub("", text).strip()


def usable_rewrite(rewrite: str, question: str) -> bool:
    """Check the LLM's output before using it: a rewrite must still be one short question. With long, citation-filled
    answers in the history, the small model sometimes answered instead ("Review [2] says that parking ... is
    difficult.") or invented facts ("a parking lot on the east side")."""
    return rewrite.endswith("?") and "[" not in rewrite and len(rewrite) <= len(question) + 60


def conversation_business(history: list[BaseMessage]) -> str | None:
    """The restaurant the user asked about most recently."""
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            business = mentioned_business(message.content)
            if business:
                return business
    return None


class ChatState(MessagesState):
    """What the graph keeps per conversation: the messages (from MessagesState) plus details of the last turn."""
    blocked: str | None
    standalone: str
    business: str | None
    sources: list[dict]


class Conversation:
    def __init__(self, safe: SafeAssistant):
        self.safe = safe
        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=safe.assistant.llm.pipeline))   # same model object
        llm = llm.bind(skip_prompt=True, pipeline_kwargs={"max_new_tokens": 60, "do_sample": False})
        self.rewrite = REWRITE_PROMPT | llm | StrOutputParser() | RunnableLambda(remove_thinking)

        graph = StateGraph(ChatState)
        graph.add_node("guard", self.guard)
        graph.add_node("resolve", self.resolve_step)
        graph.add_node("answer", self.answer)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", self.after_guard, {"blocked": END, "allowed": "resolve"})
        graph.add_edge("resolve", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile(checkpointer=InMemorySaver())

    # ---- the graph's steps: each reads the state and returns the fields it changes
    def guard(self, state: ChatState) -> dict:
        question = state["messages"][-1].content
        blocked = self.safe.guard.check(question)
        if blocked:
            return {"blocked": blocked, "standalone": question, "business": None, "sources": [],
                    "messages": [AIMessage(MESSAGES[blocked])]}
        return {"blocked": None}

    def after_guard(self, state: ChatState) -> str:
        return "blocked" if state["blocked"] else "allowed"

    def resolve_step(self, state: ChatState) -> dict:
        standalone, business = self.resolve(state["messages"][-1].content, state["messages"][:-1])
        return {"standalone": standalone, "business": business}

    def answer(self, state: ChatState) -> dict:
        result = self.safe.ask(state["standalone"], state["business"])
        sources = [{"business": d.meta["business"], "review_id": d.meta["review_id"], "text": d.content.strip()}
                   for d in result["sources"]]
        return {"blocked": result["blocked"], "sources": sources, "messages": [AIMessage(result["answer"])]}

    # ---- follow-up resolution (also used directly by the evaluation)
    def resolve(self, question: str, history: list[BaseMessage]) -> tuple[str, str | None]:
        """-> (a question that stands on its own, the restaurant it is about or None)."""
        if not history or not FOLLOW_UP.search(question):
            return question, mentioned_business(question)     # stands alone: no rewrite, no earlier restaurant
        history = trim_messages(history, max_tokens=6, token_counter=len, strategy="last", start_on="human")
        rewrite = self.rewrite.invoke({"history": history, "question": question})
        standalone = rewrite if usable_rewrite(rewrite, question) else question
        business = mentioned_business(standalone) or conversation_business(history)
        return standalone, business

    # ---- public API
    def ask(self, session_id: str, question: str) -> dict:
        """Add a message to the conversation `session_id` (a new id starts a new conversation)."""
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.invoke({"messages": [HumanMessage(question)]}, config=config)
        return {"answer": state["messages"][-1].content, "blocked": state["blocked"],
                "standalone": state["standalone"], "business": state["business"], "sources": state["sources"]}

    def history(self, session_id: str) -> list[BaseMessage]:
        state = self.graph.get_state({"configurable": {"thread_id": session_id}})
        return state.values.get("messages", [])
