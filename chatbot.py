import os
import random
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

from ddgs import DDGS

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# =====================================================
# 🌐 WEB SEARCH TOOL
# =====================================================
def web_search(query, max_results=5):
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"{r['title']} - {r['href']}")
    except Exception as e:
        return f"Search error: {str(e)}"

    return "\n".join(results)


# =====================================================
# 🧠 SMART SEARCH TRIGGER
# =====================================================
def should_search(query: str) -> bool:
    keywords = [
        "latest", "news", "today", "2026", "current",
        "update", "price", "who is", "what is", "real time"
    ]
    return any(k in query.lower() for k in keywords)


# =====================================================
# 🧠 MEMORY SYSTEM
# =====================================================
class AgentMemory:
    def __init__(self):
        self.history: List[str] = []

    def add(self, text: str):
        self.history.append(text)

    def get(self, limit=6):
        return "\n".join(self.history[-limit:])


# =====================================================
# 🚀 MAIN SYSTEM
# =====================================================
class ChatBotSystem:

    def __init__(self, choice):

        self.choice = choice

        # ---------------- LLMs ----------------
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

        # ---------------- MEMORY ----------------
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
    # SINGLE MODEL MODE
    # =================================================
    def single_model(self, question):

        mapping = {
            "1": "openai",
            "2": "gemini",
            "3": "claude"
        }

        agent = mapping.get(self.choice, "openai")

        context = self.build_context(question)

        return self.agents[agent].invoke(
            [HumanMessage(content=context)]
        ).content

    # =================================================
    # 🌐 CONTEXT BUILDER (WEB ENABLED)
    # =================================================
    def build_context(self, question):

        if should_search(question):
            results = web_search(question)
            return f"""
REAL-TIME WEB RESULTS:
{results}

QUESTION:
{question}
"""
        return question

    # =================================================
    # 🧠 MULTI-AGENT DEBATE SYSTEM
    # =================================================
    def debate_system(self, question):

        agents = list(self.agents.keys())

        context = self.build_context(question)

        # =========================
        # ROUND 1: INITIAL ANSWERS
        # =========================
        responses = {}

        for a in agents:

            prompt = f"""
You are {a.upper()}.

Context:
{context}

Memory:
{self.memory[a].get()}

Give best answer.
"""

            resp = self.agents[a].invoke(
                [HumanMessage(content=prompt)]
            ).content

            responses[a] = resp
            self.memory[a].add(f"Initial: {resp}")

        # =========================
        # ROUND 2: CRITIQUE PHASE
        # =========================
        for a in agents:

            others = [x for x in agents if x != a]

            prompt = f"""
You are {a.upper()}.

Your answer:
{responses[a]}

Other agents:
{chr(10).join([f"{o}: {responses[o]}" for o in others])}

TASK:
- Critique others
- Improve your answer
- Correct mistakes

Return improved answer only.
"""

            improved = self.agents[a].invoke(
                [HumanMessage(content=prompt)]
            ).content

            responses[a] = improved
            self.memory[a].add(f"Critique: {improved}")

        # =========================
        # ROUND 3: MESSAGE PASSING LOOP
        # =========================
        for _ in range(2):

            updated = {}

            for a in agents:

                prompt = f"""
You are {a.upper()} in debate.

Memory:
{self.memory[a].get()}

Other agents:
{chr(10).join([f"{o}: {responses[o]}" for o in agents if o != a])}

Refine your reasoning.
"""

                resp = self.agents[a].invoke(
                    [HumanMessage(content=prompt)]
                ).content

                updated[a] = resp
                self.memory[a].add(f"Refined: {resp}")

            responses = updated

        # =========================
        # 📊 SCORING
        # =========================
        scores = self.score(responses)

        # =========================
        # 🧑‍⚖️ FINAL JUDGE
        # =========================
        return self.judge(question, responses, scores)

    # =================================================
    # 📊 SCORING ENGINE
    # =================================================
    def score(self, responses):

        judge = self.agents["gemini"]

        prompt = f"""
Score (0-10):

OpenAI:
{responses['openai']}

Gemini:
{responses['gemini']}

Claude:
{responses['claude']}

Return JSON only:
{{
  "openai": {{"accuracy":0,"reasoning":0,"depth":0}},
  "gemini": {{"accuracy":0,"reasoning":0,"depth":0}},
  "claude": {{"accuracy":0,"reasoning":0,"depth":0}}
}}
"""

        return judge.invoke([HumanMessage(content=prompt)]).content

    # =================================================
    # 🧑‍⚖️ INDEPENDENT JUDGE
    # =================================================
    def judge(self, question, responses, scores):

        judge_model = random.choice([
            self.agents["gemini"],
            self.agents["claude"]
        ])

        prompt = f"""
You are a neutral judge.

QUESTION:
{question}

ANSWERS:
OpenAI:
{responses['openai']}

Gemini:
{responses['gemini']}

Claude:
{responses['claude']}

SCORES:
{scores}

TASK:
1. Compare reasoning
2. Pick winner
3. Give final improved answer

FORMAT:
Winner:
Reason:
Final Answer:
"""

        return judge_model.invoke(
            [HumanMessage(content=prompt)]
        ).content