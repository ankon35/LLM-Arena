import os
import random
import requests
from typing import List

from dotenv import load_dotenv
load_dotenv()

from ddgs import DDGS

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# =====================================================
# ⚡ TOKEN OPTIMIZED SETTINGS
# =====================================================
MAX_MEMORY = 3
MAX_WEB_RESULTS = 3
MAX_SNIPPET_LEN = 120


# =====================================================
# 🌐 WEB SEARCH (OPTIMIZED)
# =====================================================
def google_search(query, max_results=3):
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return ""

    try:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": api_key, "num": max_results}
        data = requests.get(url, params=params).json()

        results = []
        for r in data.get("organic_results", []):
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")[:MAX_SNIPPET_LEN]
            results.append(f"{title} | {snippet} | {link}")

        return "\n".join(results)

    except:
        return ""


def ddg_search(query, max_results=3):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")[:MAX_SNIPPET_LEN]
                link = r.get("href", "")
                results.append(f"{title} | {body} | {link}")

        return "\n".join(results)
    except:
        return ""


def web_search(query):
    google = google_search(query)
    ddg = ddg_search(query)

    return (google + "\n" + ddg).strip()[:1500]  # 🔥 hard token cap


# =====================================================
# 🧠 SMART SEARCH TRIGGER
# =====================================================
def should_search(query: str) -> bool:
    keywords = [
        "latest", "news", "today", "2026", "current",
        "update", "price", "who is", "what is",
        "vs", "best", "comparison"
    ]
    return any(k in query.lower() for k in keywords)


# =====================================================
# 🧠 MEMORY (TOKEN OPTIMIZED)
# =====================================================
class AgentMemory:
    def __init__(self):
        self.history: List[str] = []

    def add(self, text: str):
        self.history.append(text[-300:])  # trim stored memory

    def get(self):
        return "\n".join(self.history[-MAX_MEMORY:])


# =====================================================
# 🚀 MAIN SYSTEM
# =====================================================
class ChatBotSystem:

    def __init__(self, choice):

        self.choice = choice

        self.agents = {
            "openai": ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            ),

            "gemini": ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            ),

            "claude": ChatAnthropic(
                model="claude-sonnet-4-6",
                temperature=0.7,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
        }

        self.memory = {
            "openai": AgentMemory(),
            "gemini": AgentMemory(),
            "claude": AgentMemory(),
        }

    # =================================================
    # ENTRY
    # =================================================
    def get_response(self, question):

        if self.choice != "4":
            return self.single_model(question)

        return self.debate_system(question)

    # =================================================
    # SINGLE MODE (OPTIMIZED)
    # =================================================
    def single_model(self, question):

        mapping = {
            "1": "openai",
            "2": "gemini",
            "3": "claude"
        }

        agent = mapping.get(self.choice, "openai")

        context = self.build_context(question)

        prompt = f"""
Be concise (max 150 words).

{context}

Question:
{question}
"""

        return self.agents[agent].invoke(
            [HumanMessage(content=prompt)]
        ).content

    # =================================================
    # 🌐 CONTEXT BUILDER (1 SEARCH ONLY)
    # =================================================
    def build_context(self, question):

        if should_search(question):

            # query rewrite (cheap + effective)
            refined = self.agents["gemini"].invoke([
                HumanMessage(content=f"Rewrite as search query: {question}")
            ]).content

            results = web_search(refined)

            return f"""
SEARCH:
{refined}

RESULTS:
{results}
"""

        return question

    # =================================================
    # 🧠 MULTI-AGENT DEBATE SYSTEM
    # =================================================
    def debate_system(self, question):

        agents = list(self.agents.keys())
        context = self.build_context(question)

        responses = {}

        # =========================
        # ROUND 1 (SHORT ANSWERS)
        # =========================
        for a in agents:

            prompt = f"""
You are {a.upper()}.

Be concise (max 120 words).

Context:
{context}

Memory:
{self.memory[a].get()}

Answer:
"""

            resp = self.agents[a].invoke(
                [HumanMessage(content=prompt)]
            ).content

            responses[a] = resp
            self.memory[a].add(resp)

        # =========================
        # ROUND 2 (CRITIQUE)
        # =========================
        for a in agents:

            others = [x for x in agents if x != a]

            prompt = f"""
You are {a.upper()}.

Be short.

Your answer:
{responses[a][:300]}

Others:
{chr(10).join([f"{o}: {responses[o][:200]}" for o in others])}

Improve & fix errors.
"""

            responses[a] = self.agents[a].invoke(
                [HumanMessage(content=prompt)]
            ).content

            self.memory[a].add(responses[a])

        # =========================
        # ROUND 3 (FINAL REFINEMENT)
        # =========================
        for _ in range(1):  # reduced loop for token saving

            updated = {}

            for a in agents:

                prompt = f"""
Refine answer (max 120 words).

Memory:
{self.memory[a].get()}

Others:
{chr(10).join([f"{o}: {responses[o][:150]}" for o in agents if o != a])}
"""

                updated[a] = self.agents[a].invoke(
                    [HumanMessage(content=prompt)]
                ).content

            responses = updated

        # =========================
        # 📊 SCORING (SHORT)
        # =========================
        scores = self.score(responses)

        # =========================
        # 🧑‍⚖️ FINAL JUDGE
        # =========================
        return self.judge(question, responses, scores)

    # =================================================
    # 📊 SCORING (TOKEN OPTIMIZED)
    # =================================================
    def score(self, responses):

        prompt = f"""
Score 0-10:

OpenAI: {responses['openai'][:200]}
Gemini: {responses['gemini'][:200]}
Claude: {responses['claude'][:200]}

Return JSON only.
"""

        return self.agents["gemini"].invoke(
            [HumanMessage(content=prompt)]
        ).content

    # =================================================
    # 🧑‍⚖️ INDEPENDENT JUDGE
    # =================================================
    def judge(self, question, responses, scores):

        judge = random.choice([
            self.agents["gemini"],
            self.agents["claude"]
        ])

        prompt = f"""
Be extremely concise.

Q: {question}

A1: {responses['openai'][:200]}
A2: {responses['gemini'][:200]}
A3: {responses['claude'][:200]}

Scores:
{scores}

Output:
Winner + Reason + Final Answer
"""

        return judge.invoke(
            [HumanMessage(content=prompt)]
        ).content