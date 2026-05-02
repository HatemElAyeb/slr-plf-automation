"""
The 6 research questions defined by the supervisor for the PLF SLR.
Each question gets its own pipeline run + Qdrant collection + report.
"""

QUESTIONS = [
    {
        "id": "q1_technical",
        "category": "Technical",
        "text": "What sensor and AI combinations are most prevalent in cattle livestock farming?",
    },
    {
        "id": "q2_technical",
        "category": "Technical",
        "text": (
            "Which intermediate variables, or proxies, are most frequently identified "
            "as robust predictors of health and well-being of cattle?"
        ),
    },
    {
        "id": "q3_outcomes",
        "category": "Outcomes",
        "text": (
            "How do current PLF interventions quantify improvements in cattle welfare "
            "and sustainability?"
        ),
    },
    {
        "id": "q4_outcomes",
        "category": "Outcomes",
        "text": (
            "What are the target applications of sensors and AI combinations in "
            "livestock farming systems?"
        ),
    },
    {
        "id": "q5_gaps",
        "category": "Gaps",
        "text": (
            "What are the identified technical, economic, or socio-cultural barriers "
            "to adoption of precision livestock farming technologies?"
        ),
    },
    {
        "id": "q6_future",
        "category": "Future",
        "text": (
            "What are the current research trajectories for improving AI model "
            "robustness and explainability in Precision Livestock Farming (XAI)?"
        ),
    },
]
