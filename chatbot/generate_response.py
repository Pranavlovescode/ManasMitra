from mistral_model import generate_response

test_prompt = input("Enter your message to the Therapist Bot: ")
print("\n🧠 Testing model response...\n")
response = generate_response(test_prompt)
print("User:", test_prompt)
print("Therapist Bot:", response)