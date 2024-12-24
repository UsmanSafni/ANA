class QuestionCategorizer:
    """Classify the category of the question."""
    def __init__(self):
        self.categories = ["Exercise", "Diet", "General health", "Sleep","Mental Health","Nutrition","Drugs"]

    def classify(self, question: str) -> str:
        # Example: Rule-based categorization (replace with ML model if needed)
        if "exercise" in question.lower():
            return "Exercise"
        elif "diet" in question.lower() or "food" in question.lower():
            return "Diet"
        elif "sleep" in question.lower():
            return "Sleep"
        elif "mind" in question.lower() or "mental health" in question.lower():
            return "Mental Health"
        elif "nutrition" in question.lower():
            return "Nutrition"
        elif "medicine" in question.lower() or "drugs" in question.lower():
            return "Drugs"
        else:
            return "General health"
