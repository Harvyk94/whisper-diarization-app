from openai import OpenAI
import os
import json
import os
os.environ['DEEPSEEK_API_KEY'] = 'sk-2dd36cb849c6483781d8308f2bdc115c'


# Test text examples
test_restrictions = [
    "For treatment of type 2 diabetes mellitus in combination with metformin",
    "For severe chronic plaque psoriasis where patient has failed to achieve adequate response",
    "Patient must have chronic heart failure with reduced ejection fraction (LVEF less than or equal to 40%)"
]

class TestRestrictionProcessor:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def extract_keyword(self, text: str) -> dict:
        prompt = f"""
        Analyze the following medical restriction text and provide:
        1. The primary diagnosis as a keyword
        2. Three similar or related conditions that might be relevant
        
        Format your response as a JSON object with these exact keys:
        {{
            "primary_diagnosis": "condition name",
            "similar_conditions": ["condition1", "condition2", "condition3"]
        }}

        Text: {text}
        """

        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a medical text analyzer. Extract conditions in a structured format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        return json.loads(response.choices[0].message.content.strip())

# Create test processor
processor = TestRestrictionProcessor()

# Test each restriction
print("Testing keyword extraction:\n")
for text in test_restrictions:
    print(f"Text: {text}")
    result = processor.extract_keyword(text)
    print(f"Primary Diagnosis: {result['primary_diagnosis']}")
    print(f"Similar Conditions: {', '.join(result['similar_conditions'])}\n")