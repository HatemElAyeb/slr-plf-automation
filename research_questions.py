"""
The 6 research questions defined by the supervisor for the PLF SLR.

Each question has:
- id, category, text (research question)
- custom_fields: dict {field_name: description} — question-specific data
  the extractor will pull from each included paper, in addition to the
  standard fields (animal_species, sensor_types, ml_methods, ...).
"""

QUESTIONS = [
    {
        "id": "q1_technical",
        "category": "Technical",
        "text": "What sensor and AI combinations are most prevalent in cattle livestock farming?",
        "custom_fields": {
            "application_domains": (
                "Specific application domains addressed by the paper "
                "(e.g. lameness detection, behavior classification, weight estimation, "
                "estrus detection, body condition scoring, disease detection)."
            ),
            "sensor_ai_combinations": (
                "The actual sensor + AI/ML pairings used in this paper, as compact "
                "strings like 'camera + CNN', 'accelerometer + LSTM', 'RFID + Random Forest'."
            ),
        },
    },
    {
        "id": "q2_technical",
        "category": "Technical",
        "text": (
            "Which intermediate variables, or proxies, are most frequently identified "
            "as robust predictors of health and well-being of cattle?"
        ),
        "custom_fields": {
            "predictor_variables": (
                "Intermediate / proxy variables this paper uses as predictors "
                "(e.g. rumination time, lying time, body temperature, milk yield, "
                "activity level, feeding duration, gait score)."
            ),
            "predicted_outcomes": (
                "What the predictors are used to predict (e.g. mastitis, lameness, "
                "ketosis, estrus, calving, heat stress, feed efficiency)."
            ),
        },
    },
    {
        "id": "q3_outcomes",
        "category": "Outcomes",
        "text": (
            "How do current PLF interventions quantify improvements in cattle welfare "
            "and sustainability?"
        ),
        "custom_fields": {
            "welfare_indicators": (
                "Measurable welfare indicators reported (e.g. lying behaviour, "
                "lameness score, body condition score, stress-related cortisol)."
            ),
            "sustainability_metrics": (
                "Sustainability / environmental metrics reported (e.g. methane "
                "emissions, feed conversion ratio, antibiotic use reduction, "
                "carbon footprint, water use)."
            ),
            "intervention_type": (
                "The type of PLF intervention evaluated (e.g. wearable monitoring, "
                "automated milking, real-time alerts, decision-support system)."
            ),
        },
    },
    {
        "id": "q4_outcomes",
        "category": "Outcomes",
        "text": (
            "What are the target applications of sensors and AI combinations in "
            "livestock farming systems?"
        ),
        "custom_fields": {
            "application_categories": (
                "High-level application categories (choose all that apply): "
                "health monitoring, behavior monitoring, productivity optimization, "
                "reproduction management, breeding/genetics, supply-chain traceability, "
                "facility/environment management, identification, welfare assessment."
            ),
            "use_case": (
                "Concrete use case described in the paper (1-2 short phrases)."
            ),
        },
    },
    {
        "id": "q5_gaps",
        "category": "Gaps",
        "text": (
            "What are the identified technical, economic, or socio-cultural barriers "
            "to adoption of precision livestock farming technologies?"
        ),
        "custom_fields": {
            "barriers": (
                "Specific barriers to PLF adoption identified or discussed in the "
                "paper (free text, e.g. 'high upfront cost', 'lack of broadband', "
                "'low digital literacy', 'data privacy concerns')."
            ),
            "barrier_categories": (
                "High-level barrier categories that apply (subset of): technical, "
                "economic, socio-cultural, regulatory, infrastructure, educational, "
                "data-related."
            ),
            "stakeholders": (
                "Stakeholders mentioned (subset of): farmers, veterinarians, "
                "researchers, policy-makers, technology suppliers, extension "
                "services, consumers."
            ),
            "proposed_solutions": (
                "Solutions or recommendations the paper proposes for the identified "
                "barriers (free text)."
            ),
        },
    },
    {
        "id": "q6_future",
        "category": "Future",
        "text": (
            "What are the current research trajectories for improving AI model "
            "robustness and explainability in Precision Livestock Farming (XAI)?"
        ),
        "custom_fields": {
            "xai_methods": (
                "Explainability / interpretability methods used (e.g. SHAP, LIME, "
                "Grad-CAM, attention visualization, counterfactual explanations, "
                "feature importance, saliency maps, surrogate models)."
            ),
            "robustness_techniques": (
                "Robustness, uncertainty, or trust techniques used (e.g. adversarial "
                "training, uncertainty quantification, calibration, domain adaptation, "
                "ensembling for robustness)."
            ),
            "model_types": (
                "Type of model being explained or made robust (e.g. CNN, transformer, "
                "ensemble, LSTM)."
            ),
            "evaluation_approach": (
                "How interpretability or robustness was evaluated (e.g. user study, "
                "fidelity metric, accuracy under perturbation, qualitative inspection)."
            ),
        },
    },
]
