import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai


ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

# =========================================================
# 1. Choose company and experiment setting here
# =========================================================

# Change these two lines when switching company.
COMPANY = "NVDA"
FACTS_FILE = "NVDA_facts_package.json"

# Choose one experiment setting each time:
# "gpt_only"
# "claude_only"
# "gemini_only"
# "mixed_claude_gpt_claude"
# "mixed_gemini_gpt_claude"
EXPERIMENT = "gemini_only"

# Model names: replace with models available in your accounts if needed.
OPENAI_MODEL = "gpt-4.1"
CLAUDE_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.5-pro"


# =========================================================
# 2. Map experiment to provider roles
# =========================================================

EXPERIMENT_MAP = {
    "gpt_only": {
        "analyst": "openai",
        "critic": "openai",
        "reviser": "openai",
        "output_suffix": "config3_gpt"
    },
    "claude_only": {
        "analyst": "claude",
        "critic": "claude",
        "reviser": "claude",
        "output_suffix": "config3_claude"
    },
    "gemini_only": {
        "analyst": "gemini",
        "critic": "gemini",
        "reviser": "gemini",
        "output_suffix": "config3_gemini"
    },
    "mixed_claude_gpt_claude": {
        "analyst": "claude",
        "critic": "openai",
        "reviser": "claude",
        "output_suffix": "config3_mixed_claude_gpt_claude"
    },
    "mixed_gemini_gpt_claude": {
        "analyst": "gemini",
        "critic": "openai",
        "reviser": "claude",
        "output_suffix": "config3_mixed_gemini_gpt_claude"
    },
}

setting = EXPERIMENT_MAP[EXPERIMENT]


# =========================================================
# 3. Read prompt/template/data files
# =========================================================

analyst_prompt = (ROOT / "prompts" / "skills" / "earnings_analyst.md").read_text(encoding="utf-8")
critic_prompt = (ROOT / "prompts" / "skills" / "earnings_critic.md").read_text(encoding="utf-8")
reviser_prompt = (ROOT / "prompts" / "skills" / "earnings_reviser.md").read_text(encoding="utf-8")
template = (ROOT / "prompts" / "templates" / "earnings_update_template.md").read_text(encoding="utf-8")
facts = json.loads((ROOT / "data" / FACTS_FILE).read_text(encoding="utf-8"))




# =========================================================
# 4. Initialize clients
# =========================================================
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_GENAI_API_KEY")
)

openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Anthropic(api_key=claude_api_key)
gemini_client = genai.Client(api_key=gemini_api_key)

print("ROOT =", ROOT)
print(".env exists:", (ROOT / ".env").exists())
print("OpenAI key loaded:", openai_api_key is not None)
print("Claude key loaded:", claude_api_key is not None)
print("Gemini key loaded:", gemini_api_key is not None)
# ----------------------------------------

# =========================================================
# 5. Unified generation function
# =========================================================

def generate_text(provider: str, system_prompt: str, user_input: str, temperature: float = 0.2) -> str:
    """
    provider: openai, claude, or gemini
    """

    if provider == "openai":
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    if provider == "claude":
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=3500,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_input},
            ],
        )
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    if provider == "gemini":
        # Google Gen AI SDK uses generate_content for text generation.
        # We combine system prompt and user input into one content string for a simple text workflow.
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_prompt}\n\n{user_input}",
        )
        return response.text

    raise ValueError(f"Unknown provider: {provider}")


# =========================================================
# 6. Output folder
# =========================================================

reports_dir = ROOT / "reports" / "config3_model_experiments"
reports_dir.mkdir(parents=True, exist_ok=True)

output_file = f"{COMPANY}_{setting['output_suffix']}.md"
report_path = reports_dir / output_file


# =========================================================
# 7. Analyst
# =========================================================

analyst_user_input = f"""
Facts package:
{json.dumps(facts, indent=2, ensure_ascii=False)}

Required template:
{template}
"""

analyst_output = generate_text(
    provider=setting["analyst"],
    system_prompt=analyst_prompt,
    user_input=analyst_user_input,
    temperature=0.2,
)

print("=== ANALYST OUTPUT ===")
print(analyst_output)


# =========================================================
# 8. Critic
# =========================================================

critic_user_input = f"""
Facts package:
{json.dumps(facts, indent=2, ensure_ascii=False)}

Analyst draft:
{analyst_output}
"""

critic_output = generate_text(
    provider=setting["critic"],
    system_prompt=critic_prompt,
    user_input=critic_user_input,
    temperature=0.1,
)

print("\n=== CRITIC OUTPUT ===")
print(critic_output)


# =========================================================
# 9. Reviser
# =========================================================

reviser_user_input = f"""
Facts package:
{json.dumps(facts, indent=2, ensure_ascii=False)}

Analyst draft:
{analyst_output}

Critique report:
{critic_output}

Required template:
{template}
"""

final_output = generate_text(
    provider=setting["reviser"],
    system_prompt=reviser_prompt,
    user_input=reviser_user_input,
    temperature=0.1,
)

print("\n=== FINAL REVISED OUTPUT ===")
print(final_output)


# =========================================================
# 10. Save final report
# =========================================================

metadata = f"""# Experiment Metadata

- Company: {COMPANY}
- Experiment: {EXPERIMENT}
- Analyst Provider: {setting['analyst']}
- Critic Provider: {setting['critic']}
- Reviser Provider: {setting['reviser']}
- OpenAI Model: {OPENAI_MODEL}
- Claude Model: {CLAUDE_MODEL}
- Gemini Model: {GEMINI_MODEL}

---

"""

report_path.write_text(metadata + final_output, encoding="utf-8")

print(f"\nSaved model experiment report to: {report_path}")