import requests
import json
import time
from datetime import datetime, timedelta

# --- Configuration ---
BASE_URL = "http://192.168.206.252:23333/api/v1"
USER_ID = "user_sophia_456"

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
    print("-" * 50)

def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# --- Rich Test Data That Shows Memory Value ---

# Month 1: User exploring data science career
month1_data_science = [
    {
        "role": "user",
        "message_content": "I'm a marketing analyst but I'm really interested in transitioning to data science. I have a background in statistics from college but haven't used Python much. Where should I start?",
        "session_id": "session_career_start",
        "conversation_title": "Data Science Career Transition",
        "date": '2024-01-15T10:30:00.000000',
    },
    {
        "role": "assistant", 
        "message_content": "That's exciting! Your statistics background is a great foundation. I'd recommend starting with Python basics, then pandas and numpy for data manipulation, followed by scikit-learn for machine learning. Your marketing domain knowledge will be valuable.",
        "session_id": "session_career_start",
        "conversation_title": "Data Science Career Transition",
        "date": '2024-01-15T10:31:00.000000',
    },
    {
        "role": "user",
        "message_content": "That sounds good. I'm particularly interested in customer behavior analysis and predictive modeling for marketing campaigns. I think that's where I can leverage my marketing experience.",
        "session_id": "session_career_start", 
        "conversation_title": "Data Science Career Transition",
        "date": '2024-01-15T10:32:00.000000',
    }
]

# Month 2: User learning and making progress
month2_learning = [
    {
        "role": "user",
        "message_content": "I've been learning Python for a few weeks now. I completed a pandas tutorial and built my first data analysis project analyzing customer churn using a sample dataset. It was challenging but fun!",
        "session_id": "session_learning_progress",
        "conversation_title": "Learning Progress Update",
        "date": '2024-02-20T14:15:00.000000',
    },
    {
        "role": "assistant",
        "message_content": "That's fantastic progress! Customer churn analysis is a perfect project for someone with your marketing background. How did you find working with the data cleaning and feature engineering aspects?",
        "session_id": "session_learning_progress",
        "conversation_title": "Learning Progress Update", 
        "date": '2024-02-20T14:16:00.000000',
    },
    {
        "role": "user",
        "message_content": "Data cleaning was more time-consuming than I expected, but I can see why it's so important. I used logistic regression for the churn prediction and got decent results. Now I want to try more advanced algorithms like random forests.",
        "session_id": "session_learning_progress",
        "conversation_title": "Learning Progress Update",
        "date": '2024-02-20T14:17:00.000000',
    }
]

# Month 3: User facing challenges and getting specific help
month3_challenges = [
    {
        "role": "user",
        "message_content": "I'm working on a personal project analyzing social media engagement data for different marketing campaigns. I'm struggling with feature selection - there are so many variables and I'm not sure which ones are most important.",
        "session_id": "session_feature_selection",
        "conversation_title": "Feature Selection Challenge",
        "date": '2024-03-10T16:45:00.000000',
    },
    {
        "role": "assistant",
        "message_content": "Feature selection is definitely tricky! For marketing engagement data, I'd suggest starting with correlation analysis and feature importance from tree-based models. Also consider domain knowledge - engagement metrics like click-through rates, time spent, and demographic segments are usually powerful predictors.",
        "session_id": "session_feature_selection",
        "conversation_title": "Feature Selection Challenge",
        "date": '2024-03-10T16:46:00.000000',
    },
    {
        "role": "user",
        "message_content": "That's really helpful! I hadn't thought about using my marketing intuition to guide the feature selection. I'll try the correlation analysis and see which features make the most business sense too.",
        "session_id": "session_feature_selection",
        "conversation_title": "Feature Selection Challenge",
        "date": '2024-03-10T16:47:00.000000',
    }
]

# Month 4: User sharing success and planning next steps
month4_success = [
    {
        "role": "user",
        "message_content": "Great news! I just got an interview for a Junior Data Scientist position at a marketing tech company. They were impressed by my portfolio projects, especially the customer segmentation analysis I did. I'm nervous but excited!",
        "session_id": "session_interview_success",
        "conversation_title": "Job Interview Success",
        "date": '2024-04-05T11:20:00.000000',
    },
    {
        "role": "assistant",
        "message_content": "That's wonderful! Your unique combination of marketing domain expertise and growing data science skills is exactly what many companies are looking for. For the interview, be ready to discuss your projects and how you approached real business problems.",
        "session_id": "session_interview_success",
        "conversation_title": "Job Interview Success",
        "date": '2024-04-05T11:21:00.000000',
    },
    {
        "role": "user",
        "message_content": "Thank you! I'm planning to continue learning about A/B testing and causal inference since that seems really relevant for marketing applications. Any recommendations for resources?",
        "session_id": "session_interview_success",
        "conversation_title": "Job Interview Success",
        "date": '2024-04-05T11:22:00.000000',
    }
]

# --- Test Memory Queries That Show Value ---
memory_test_queries = [
    {
        "name": "User Background and Goals",
        "query": "What is the user's professional background and career goals?",
        "expected_value": "Should find marketing analyst background, statistics education, and data science career transition goals"
    },
    {
        "name": "Learning Progress and Skills",
        "query": "What programming skills has the user developed and what projects have they worked on?",
        "expected_value": "Should find Python learning, pandas/numpy skills, customer churn analysis project"
    },
    {
        "name": "Domain Expertise", 
        "query": "What specific domain knowledge and interests does the user have?",
        "expected_value": "Should identify marketing expertise, customer behavior analysis, predictive modeling for campaigns"
    },
    {
        "name": "Current Challenges",
        "query": "What challenges or learning obstacles has the user faced?",
        "expected_value": "Should find feature selection struggles, data cleaning time challenges"
    },
    {
        "name": "Recent Achievements",
        "query": "What recent successes or milestones has the user achieved?",
        "expected_value": "Should find job interview success, portfolio projects, customer segmentation analysis"
    },
    {
        "name": "Future Learning Plans",
        "query": "What does the user want to learn next?",
        "expected_value": "Should find A/B testing, causal inference, random forests interest"
    }
]

# --- Main Test Function ---
def run_memory_value_test():
    """Run the comprehensive memory value test."""
    headers = {"Content-Type": "application/json"}
    
    print_section("MEMORY VALUE DEMONSTRATION TEST")
    print("This test shows how memory enables personalized, context-aware assistance")
    print("by tracking user's journey, preferences, and learning progress over time.")
    
    # Step 1: Ingest conversation history
    print_section("STEP 1: INGESTING CONVERSATION HISTORY")
    """
    conversation_batches = [
        ("Month 1: Career Transition Start", month1_data_science),
        ("Month 2: Learning Progress", month2_learning), 
        ("Month 3: Technical Challenges", month3_challenges),
        ("Month 4: Success and Next Steps", month4_success)
    ]
    
    for batch_name, messages in conversation_batches:
        print(f"\n--- {batch_name} ---")
        payload = {
            "input": messages,
            "metadata": {"user_id": USER_ID},
            "target_type": "personal_memory",
            "input_type": "chat_history",
            "process_strategy": {"force_reprocess": False},
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/save", 
                headers=headers, 
                data=json.dumps(payload)
            )
            print_response(response)
            
            if response.status_code == 200:
                print(f"✅ Successfully ingested {batch_name}")
            else:
                print(f"❌ Failed to ingest {batch_name}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection Error: Could not connect to {BASE_URL}")
            print("Please ensure the FastAPI server is running.")
            return
        except Exception as e:
            print(f"❌ Error ingesting {batch_name}: {str(e)}")
            
        # Small delay between batches
        time.sleep(1)
    """

    
    # Step 2: Query memory to answer questions
    print_section("STEP 2: QUERYING MEMORY TO ANSWER QUESTIONS")
    print("Now we'll query the memory system to answer questions about the user.")
    print("This demonstrates how memory enables personalized, context-aware responses.")
    
    for i, test_query in enumerate(memory_test_queries, 1):
        print(f"\n--- Query {i}: {test_query['name']} ---")
        print(f"🔍 Question: {test_query['query']}")
        print(f"📝 Expected Value: {test_query['expected_value']}")
        print()
        
        # Query the memory
        query_payload = {
            "query": test_query['query'],
            "user_id": USER_ID,
            "memory_types": ["conversation", "insights"],
            "top_k": 5
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/memory/retrieve",
                headers=headers,
                data=json.dumps(query_payload)
            )
            
            if response.status_code == 200:
                data = response.json()
                total_found = data.get('data', {}).get('total_found', 0)
                results = data.get('data', {}).get('results', {})
                
                print(f"📊 Memory Search Results: Found {total_found} relevant memories")
                
                # Show conversation results
                conversations = results.get('conversations', [])
                if conversations:
                    print(f"💬 Found {len(conversations)} relevant conversations:")
                    for conv in conversations[:2]:  # Show first 2
                        print(f"   - {conv.get('name', 'Unknown')}")
                        content_preview = conv.get('content', '')[:100] + "..."
                        print(f"     Preview: {content_preview}")
                
                # Show insights results
                insights = results.get('insights', [])
                if insights:
                    print(f"🧠 Found {len(insights)} relevant insights:")
                    for insight in insights[:2]:  # Show first 2
                        print(f"   - {insight.get('name', 'Unknown')}")
                        desc_preview = insight.get('description', '')[:100] + "..."
                        print(f"     Description: {desc_preview}")
                
                if total_found > 0:
                    print("✅ Memory system successfully retrieved relevant information!")
                else:
                    print("⚠️  No relevant memories found for this query.")
                    
            else:
                print(f"❌ Query failed with status {response.status_code}")
                print_response(response)
                
        except Exception as e:
            print(f"❌ Error querying memory: {str(e)}")
        
        print("-" * 50)
    
    # Step 3: Demonstrate memory value
    print_section("STEP 3: MEMORY VALUE DEMONSTRATION")
    print("🎯 MEMORY SYSTEM VALUE DEMONSTRATED:")
    print()
    print("1. 📚 CONTEXT PRESERVATION: The system remembers the user's journey")
    print("   from marketing analyst to aspiring data scientist over 4 months.")
    print()
    print("2. 🧠 PERSONALIZED UNDERSTANDING: It knows their specific interests")
    print("   (customer behavior, marketing campaigns, predictive modeling).")
    print()
    print("3. 🎯 TAILORED ASSISTANCE: Can provide relevant help based on their")
    print("   background, current skill level, and learning goals.")
    print()
    print("4. 🔄 PROGRESS TRACKING: Remembers what they've learned and what")
    print("   challenges they've faced to avoid repetition.")
    print()
    print("5. 🚀 FUTURE GUIDANCE: Can suggest next steps based on their")
    print("   interests and career trajectory.")
    print()
    print("💡 WITHOUT MEMORY: Each conversation would start from scratch.")
    print("🎉 WITH MEMORY: The AI becomes a personalized learning companion!")
    
    print_section("TEST COMPLETED")
    print(f"User ID: {USER_ID}")
    print("You can now use this data to test more advanced memory queries!")

if __name__ == "__main__":
    run_memory_value_test() 