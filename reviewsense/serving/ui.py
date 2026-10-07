"""A web page for people (Gradio), served by the API at /ui. It calls the same loaded models as the API endpoints.

Three tabs: Chat (the Part 12 chatbot; each browser gets its own conversation), Star rating, Search.
Gradio is the quick way to put models in front of colleagues and testers; a customer-facing product would have its
own front end calling the /chat, /stars and /search endpoints.
"""
from __future__ import annotations

import gradio
from fastapi import FastAPI


def build_ui(app: FastAPI) -> gradio.Blocks:
    def components():
        return app.state.components                       # loaded by the API at startup

    def chat(message: str, history: list, request: gradio.Request) -> str:
        chatbot = components().chatbot
        if chatbot is None:
            return "Chat is switched off on this server."
        return chatbot.ask(f"ui-{request.session_hash}", message)["answer"]   # one conversation per browser tab

    def stars(text: str) -> str:
        if not text.strip():
            return ""
        value = components().stars.stars([text])[0]
        return f"{value:.1f} / 5  " + "★" * round(value)

    def search(query: str) -> list[list[str]]:
        if not query.strip():
            return []
        result = components().search.search(query, k=5)
        return [[hit["business"], hit["text"]] for hit in result["results"]]

    with gradio.Blocks(title="ReviewSense") as ui:
        gradio.Markdown("# ReviewSense\nAsk about the restaurants, rate a review, or search what customers say.")
        with gradio.Tab("Chat"):
            gradio.ChatInterface(chat, examples=["Is Bella Napoli expensive?", "Summarize Sunrise Cafe",
                                                 "Book a table at Golden Dragon for 2 tomorrow at 7 pm"])
        with gradio.Tab("Star rating"):
            review = gradio.Textbox(label="Review", lines=4, placeholder="The tacos were great but the wait was long.")
            rating = gradio.Textbox(label="Predicted rating")
            review.submit(stars, review, rating)
            gradio.Button("Rate").click(stars, review, rating)
        with gradio.Tab("Search"):
            query = gradio.Textbox(label="Search the reviews", placeholder="slow service")
            results = gradio.Dataframe(headers=["Restaurant", "Passage"], wrap=True)
            query.submit(search, query, results)
    return ui
