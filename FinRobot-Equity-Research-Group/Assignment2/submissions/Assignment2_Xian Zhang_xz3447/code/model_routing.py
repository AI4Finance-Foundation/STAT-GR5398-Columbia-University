"""Model-routing profiles for the experiment matrix.

A profile maps each report-section role to a specific model, plus the model
used by the critic (for the critique-revise enhancement). Profiles are picked
via the FINROBOT_PROFILE environment variable or the --profile CLI flag of the
experiment runner.

Profile design choices (Track A — model experimentation):

- gpt_baseline      : All-OpenAI (gpt-4.1 / gpt-4o-mini). Reproduces the
                      original FinRobot stack for a fair baseline.
- claude_all        : All-Anthropic (Opus 4.7 for heavy analysis, Haiku 4.5
                      for light copy). Tests Claude's analytical depth.
- gemini_all        : All-Google (Gemini 2.5 Pro / Flash). Tests Gemini.
- mixed_critic      : Claude Opus authors heavy sections, GPT-4.1 critiques,
                      Gemini Flash handles short copy. Demonstrates
                      role-specific assignment + cross-model critique.

The "heavy" sections are investment_overview / risks / valuation_overview /
competitor_analysis / company_overview. "Light" sections are
tagline / major_takeaways / news_summary.
"""

from typing import Dict, List, Optional


HEAVY_SECTIONS = {
    "company_overview",
    "investment_overview",
    "valuation_overview",
    "risks",
    "competitor_analysis",
}
LIGHT_SECTIONS = {"tagline", "major_takeaways", "news_summary"}

# Sections subjected to the Claude-inspired critique-revise stage.
# Limited to the three highest-stakes analytical sections to control cost.
CRITIQUE_SECTIONS = ("investment_overview", "risks", "valuation_overview")


def _build_routing(heavy_model: str, light_model: str) -> Dict[str, str]:
    routing = {sec: heavy_model for sec in HEAVY_SECTIONS}
    routing.update({sec: light_model for sec in LIGHT_SECTIONS})
    return routing


PROFILES: Dict[str, Dict] = {
    # ---- Baseline: original FinRobot stack ----
    "gpt_baseline": {
        "label": "GPT Baseline (all OpenAI)",
        "routing": _build_routing(heavy_model="gpt-4.1", light_model="gpt-4o-mini"),
        "critic_model": "gpt-4.1",
        "critique_enabled": False,  # baseline = no critique stage
    },
    # ---- All-Claude ----
    "claude_all": {
        "label": "All-Claude (Opus 4.7 / Haiku 4.5)",
        "routing": _build_routing(
            heavy_model="claude-opus-4-5",
            light_model="claude-haiku-4-5",
        ),
        "critic_model": "claude-opus-4-5",
        "critique_enabled": True,
    },
    # ---- All-Gemini ----
    "gemini_all": {
        "label": "All-Gemini (2.5 Pro / Flash)",
        "routing": _build_routing(
            heavy_model="gemini-2.5-pro",
            light_model="gemini-2.5-flash",
        ),
        "critic_model": "gemini-2.5-pro",
        "critique_enabled": True,
    },
    # ---- Mixed best-of-breed ----
    "mixed_critic": {
        "label": "Mixed: Claude analyst + GPT critic + Gemini light",
        "routing": {
            # Heavy analytical work: Claude Opus 4.7
            "company_overview": "claude-opus-4-5",
            "investment_overview": "claude-opus-4-5",
            "valuation_overview": "claude-opus-4-5",
            "risks": "claude-opus-4-5",
            "competitor_analysis": "claude-opus-4-5",
            # Light / copywriting: Gemini Flash
            "tagline": "gemini-2.5-flash",
            "major_takeaways": "gemini-2.5-flash",
            "news_summary": "gemini-2.5-flash",
        },
        # Cross-model critique: a different family reviews Claude's work
        "critic_model": "gpt-4.1",
        "critique_enabled": True,
    },
}


def get_profile(name: str) -> Dict:
    if name not in PROFILES:
        raise KeyError(
            f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}"
        )
    return PROFILES[name]


def list_profiles() -> List[str]:
    return list(PROFILES.keys())


def resolve_section_model(profile_name: str, section: str, override: Optional[str] = None) -> str:
    if override:
        return override
    profile = get_profile(profile_name)
    routing = profile["routing"]
    if section not in routing:
        raise KeyError(f"Section '{section}' not in profile '{profile_name}' routing")
    return routing[section]
