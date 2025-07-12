import requests
import json
from datetime import datetime, timedelta

# --- Configuration ---
BASE_URL = "http://192.168.206.252:23333/api/v1"
USER_ID = "user_alex_123"

# --- Helper Functions ---
def print_response(response: requests.Response):
    """Prints the API response in a readable format."""
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print("Response Text:")
        print(response.text)
    print("-" * 30)

# --- Test Data ---

# Day 1: Exploring Frontend Development
day1_messages = [
    {
        "role": "user",
        "message_content": "I've been a React developer for a few years, but I'm curious about Vue.js. What are the main differences?",
        "session_id": "session_frontend_1",
        "conversation_title": "Exploring Frontend Frameworks",
        "date": '2024-07-11T12:43:45.331597',
    },
    {
        "role": "assistant",
        "message_content": "That's a great question! The key differences are in their approach to state management and component structure. React uses the virtual DOM and JSX, while Vue uses a more traditional template-based syntax. Vue's state management with Pinia is often seen as simpler than Redux for many use cases.",
        "session_id": "session_frontend_1",
        "conversation_title": "Exploring Frontend Frameworks",
        "date": '2024-07-11T12:43:50.331597',
    },
    {
        "role": "user",
        "message_content": "Interesting. I've heard Vue has a lower barrier to entry. I think I'll build a small project with it to see how it feels. Thanks!",
        "session_id": "session_frontend_1",
        "conversation_title": "Exploring Frontend Frameworks",
        "date": '2024-07-11T12:44:02.331597'
    },
]

# Day 2: Pivoting to Machine Learning
day2_messages = [
    {
        "role": "user",
        "message_content": "You know, after thinking about it, I feel a bit burnt out on frontend work. I want a new challenge. I'm thinking of pivoting my career to machine learning. Where should I start?",
        "session_id": "session_ml_pivot_2",
        "conversation_title": "Career Pivot to Machine Learning",
        "date": '2024-07-12T12:44:02.331597',
    },
    {
        "role": "assistant",
        "message_content": "That's an exciting and significant career move! For a software developer, a good path would be to solidify your Python skills, then dive into the math fundamentals (Linear Algebra, Calculus, Probability), and finally start with core ML libraries like Scikit-learn, Pandas, and NumPy before moving to deep learning with TensorFlow or PyTorch.",
        "session_id": "session_ml_pivot_2",
        "conversation_title": "Career Pivot to Machine Learning",
        "date": '2024-07-12T12:44:32.331597',
    },
    {
        "role": "user",
        "message_content": "That makes sense. It feels like a mountain to climb, but I'm excited. I'll start by sharpening my Python skills specifically for data science. This feels like the right direction for me. Thanks for the roadmap!",
        "session_id": "session_ml_pivot_2",
        "conversation_title": "Career Pivot to Machine Learning",
        "date": '2024-07-12T12:45:02.331597',
    },
]

# Day 3: A Year Later - New City, New Hobby
day365_messages = [
    {
        "role": "user",
        "message_content": "It's been a while! A lot has changed for me in the last year. I actually moved from San Francisco to Austin a few months ago. The vibe is so different here, much more laid back.",
        "session_id": "session_life_update_3",
        "conversation_title": "Life Updates and New Hobbies",
        "date": '2025-07-12T12:44:02.331597',
    },
    {
        "role": "assistant",
        "message_content": "Wow, Austin! That's a big and exciting move. It's a major hub for both tech and music. How are you finding the transition?",
        "session_id": "session_life_update_3",
        "conversation_title": "Life Updates and New Hobbies",
        "date": '2025-07-12T12:45:02.331597',
    },
    {
        "role": "user",
        "message_content": "I love it! And speaking of music, I've picked up the guitar and I'm really getting into blues. It's a fantastic way to unwind after a day of coding. I'm even thinking of building a small app to help me practice scales.",
        "session_id": "session_life_update_3",
        "conversation_title": "Life Updates and New Hobbies",
        "date": '2025-07-12T12:45:32.331597',
    },
]

# --- Test Execution ---
def run_test():
    """Runs the memory ingestion test."""
    headers = {"Content-Type": "application/json"}

    # --- Send Day 1 History ---
    print("--- Sending Day 1: Frontend Discussion ---")
    day1_payload = {
        "input": day1_messages,
        "metadata": {"user_id": USER_ID},
        "target_type": "personal_memory",
        "input_type": "chat_history",
        "process_strategy": {"force_reprocess": False},
    }
    try:
        response_day1 = requests.post(
            f"{BASE_URL}/save", headers=headers, data=json.dumps(day1_payload)
        )
        print_response(response_day1)
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Could not connect to the API at {BASE_URL}.")
        print("Please ensure the FastAPI server is running.")
        return

    return
    # --- Send Day 2 History ---
    print("--- Sending Day 2: Machine Learning Pivot ---")
    day2_payload = {
        "input": day2_messages,
        "metadata": {"user_id": USER_ID},
        "target_type": "personal_memory",
        "input_type": "chat_history",
        "process_strategy": {"force_reprocess": False},
    }
    response_day2 = requests.post(
        f"{BASE_URL}/save", headers=headers, data=json.dumps(day2_payload)
    )
    print_response(response_day2)

    # --- Send Day 3 History ---
    print("--- Sending Day 3: Life Changes (New City & Hobby) ---")
    day365_payload = {
        "input": day365_messages,
        "metadata": {"user_id": USER_ID},
        "target_type": "personal_memory",
        "input_type": "chat_history",
        "process_strategy": {"force_reprocess": False},
    }
    response_day3 = requests.post(
        f"{BASE_URL}/save", headers=headers, data=json.dumps(day365_payload)
    )
    print_response(response_day3)

    print("Test complete. Check the server logs and responses to verify.")
    print(f"You can now use the retrieve endpoint to query memories for user '{USER_ID}'.")


if __name__ == "__main__":
    run_test() 