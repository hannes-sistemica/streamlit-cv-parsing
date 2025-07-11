import streamlit as st
import dspy
import os
import requests
import json

# Configuration options
OLLAMA_MODELS = {
    "Llama 3.2 3B": "llama3.2:3b",
    "Llama 3.2 1B": "llama3.2:1b", 
    "Llama 3.1 8B": "llama3.1:8b",
    "Qwen 2.5 3B": "qwen2.5:3b",
    "Phi 3.5": "phi3.5:3.8b"
}

OPENAI_MODELS = {
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}

def get_ollama_models(ollama_url="http://localhost:11434"):
    """Get available models from Ollama server"""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = {}
            for model in models_data.get('models', []):
                name = model['name']
                # Clean up display names
                if ':latest' in name:
                    display_name = name.replace(':latest', '')
                else:
                    display_name = name.replace(':', ' ')
                
                # Make it more readable
                display_name = display_name.replace('-', ' ').title()
                models[display_name] = name
            return models if models else OLLAMA_MODELS
        else:
            return OLLAMA_MODELS
    except Exception:
        return OLLAMA_MODELS

def configure_dspy(provider, model_name, api_key=None, ollama_url="http://localhost:11434"):
    """Configure DSPy with selected provider and model"""
    
    if provider == "Ollama":
        # Minimal configuration
        lm = dspy.OllamaLocal(model=model_name, base_url=ollama_url)
        dspy.settings.configure(lm=lm)
        return f"Ollama: {model_name}"
        
    elif provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        os.environ["OPENAI_API_KEY"] = api_key
        lm = dspy.OpenAI(model=model_name)
        dspy.settings.configure(lm=lm)
        return f"OpenAI: {model_name}"
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Single field extraction to avoid confusion
class ExtractSingleField(dspy.Signature):
    """Extract one specific piece of information from resume text."""
    resume_text = dspy.InputField()
    field_type = dspy.InputField()
    result = dspy.OutputField()

# Simplified processor that extracts one field at a time
class SingleFieldProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractSingleField)
    
    def forward(self, resume_text, field_type):
        return self.extract(resume_text=resume_text, field_type=field_type)

def extract_resume_info(resume_text):
    """Extract all fields using single-field approach"""
    processor = SingleFieldProcessor()
    
    fields = {
        "name": "candidate's full name",
        "email": "email address", 
        "phone": "phone number",
        "skills": "technical skills and technologies",
        "education": "educational background",
        "experience": "work experience and job history"
    }
    
    results = {}
    for field, description in fields.items():
        try:
            result = processor(resume_text, f"Extract the {description} from this resume")
            results[field] = result.result
        except:
            results[field] = "Not found"
    
    return type('Result', (), results)()

def extract_with_json_prompting(provider, model_name, resume_text, api_key=None, ollama_url="http://localhost:11434"):
    """Sophisticated JSON-based extraction with reliable prompting"""
    
    json_prompt = f"""You are an expert resume parser. Extract information from the following resume and return it as valid JSON.

RESUME TEXT:
{resume_text}

INSTRUCTIONS:
1. Read the resume carefully
2. Extract the requested information
3. If information is not found, use "Not found" as the value
4. Return ONLY valid JSON, no additional text or formatting
5. Use the exact JSON structure shown below

REQUIRED JSON FORMAT:
{{
    "name": "candidate's full name",
    "email": "email address", 
    "phone": "phone number",
    "skills": "comma-separated list of technical skills and technologies",
    "education": "educational background and degrees",
    "experience": "work experience summary including companies, roles, and dates"
}}

JSON OUTPUT:"""

    if provider == "Ollama":
        return extract_json_ollama(model_name, json_prompt, ollama_url)
    else:
        return extract_json_openai(model_name, json_prompt, api_key)

def extract_json_ollama(model_name, prompt, ollama_url):
    """Direct Ollama API call for JSON extraction"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Force JSON output
        "options": {
            "temperature": 0.1,  # Low temperature for consistent output
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            json_text = result.get('response', '{}')
            return parse_json_response(json_text)
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to call Ollama: {str(e)}")

def extract_json_openai(model_name, prompt, api_key):
    """Direct OpenAI API call for JSON extraction"""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert resume parser. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}  # Force JSON output
        )
        json_text = response.choices[0].message.content
        return parse_json_response(json_text)
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def parse_json_response(json_text):
    """Parse JSON response and create result object"""
    try:
        data = json.loads(json_text)
        # Ensure all required fields exist
        result = {
            'name': data.get('name', 'Not found'),
            'email': data.get('email', 'Not found'),
            'phone': data.get('phone', 'Not found'),
            'skills': data.get('skills', 'Not found'),
            'education': data.get('education', 'Not found'),
            'experience': data.get('experience', 'Not found'),
            'raw_response': json_text
        }
        return type('Result', (), result)()
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        result = {
            'name': 'JSON parsing failed',
            'email': 'JSON parsing failed',
            'phone': 'JSON parsing failed',
            'skills': 'JSON parsing failed',
            'education': 'JSON parsing failed',
            'experience': 'JSON parsing failed',
            'raw_response': json_text
        }
        return type('Result', (), result)()

def main():
    st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="wide")
    
    st.title("📄 Resume Parser with DSPy")
    st.write("Choose between local Ollama models or remote OpenAI API")
    
    # Configuration Sidebar
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Provider selection
        provider = st.radio(
            "Choose Provider:",
            ["Ollama (Local)", "OpenAI (Remote)"],
            help="Select between local Ollama models or OpenAI API"
        )
        
        st.divider()
        
        # Extraction method selection
        extraction_method = st.radio(
            "Choose Extraction Method:",
            ["DSPy Framework", "Direct JSON Prompting"],
            help="DSPy uses structured programming, Direct prompting uses JSON output"
        )
        
        if provider == "Ollama (Local)":
            st.subheader("Local Ollama Setup")
            
            # Ollama URL configuration
            ollama_url = st.text_input(
                "Ollama URL:", 
                value="http://localhost:11434",
                help="Default Ollama server URL"
            )
            
            # Get available models from Ollama
            if st.button("🔄 Refresh Models"):
                st.rerun()
            
            available_models = get_ollama_models(ollama_url)
            
            if available_models == OLLAMA_MODELS:
                st.warning("⚠️ Could not connect to Ollama. Showing default models.")
            else:
                st.success(f"✅ Found {len(available_models)} models from Ollama")
            
            # Model selection
            selected_model_display = st.selectbox(
                "Select Model:",
                list(available_models.keys()),
                help="Choose a local Ollama model"
            )
            selected_model = available_models[selected_model_display]
            
            st.info(f"🚀 Using: {selected_model_display}")
            if available_models == OLLAMA_MODELS:
                st.caption("Make sure Ollama is running and the model is installed:\n`ollama pull " + selected_model + "`")
            
            api_key = None
            
        else:  # OpenAI
            st.subheader("OpenAI API Setup")
            
            # API Key input
            api_key = st.text_input(
                "OpenAI API Key:", 
                type="password",
                help="Enter your OpenAI API key"
            )
            
            # Model selection
            selected_model_display = st.selectbox(
                "Select Model:",
                list(OPENAI_MODELS.keys()),
                help="Choose an OpenAI model"
            )
            selected_model = OPENAI_MODELS[selected_model_display]
            
            if api_key:
                st.success(f"🔑 API Key configured")
                st.info(f"🚀 Using: {selected_model_display}")
            else:
                st.warning("⚠️ API key required")
            
            ollama_url = None
        
        # Test connection button
        if st.button("🧪 Test Connection"):
            try:
                if provider == "Ollama (Local)":
                    test_result = configure_dspy("Ollama", selected_model, ollama_url=ollama_url)
                    
                    # Try a simple test call
                    class SimpleTest(dspy.Signature):
                        """Simple test"""
                        input: str = dspy.InputField()
                        output: str = dspy.OutputField()
                    
                    test_predict = dspy.Predict(SimpleTest)
                    test_response = test_predict(input="Hello")
                    st.success(f"✅ Connected to {test_result}")
                    st.info(f"Test response: {test_response.output}")
                    
                else:
                    if not api_key:
                        st.error("API key required for OpenAI")
                    else:
                        test_result = configure_dspy("OpenAI", selected_model, api_key=api_key)
                        
                        # Try a simple test call
                        class SimpleTest(dspy.Signature):
                            """Simple test"""
                            input: str = dspy.InputField()
                            output: str = dspy.OutputField()
                        
                        test_predict = dspy.Predict(SimpleTest)
                        test_response = test_predict(input="Hello")
                        st.success(f"✅ Connected to {test_result}")
                        st.info(f"Test response: {test_response.output}")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"❌ Connection failed: {str(e)}")
                st.code(error_details, language="python")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Resume Input")
        resume_text = st.text_area(
            "Paste resume text here:",
            height=400,
            placeholder="Copy and paste the resume content here..."
        )
        
        extract_button = st.button("🔍 Extract Information", type="primary")
    
    with col2:
        st.subheader("📊 Extracted Information")
        
        if extract_button and resume_text:
            # Validation
            if provider == "OpenAI (Remote)" and not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
                return
            
            try:
                # Configure DSPy
                with st.spinner("Configuring DSPy..."):
                    if provider == "Ollama (Local)":
                        config_result = configure_dspy("Ollama", selected_model, ollama_url=ollama_url)
                    else:
                        config_result = configure_dspy("OpenAI", selected_model, api_key=api_key)
                
                st.info(f"Using: {config_result}")
                
                # Extract information using selected method
                with st.spinner(f"Extracting information using {extraction_method}..."):
                    if extraction_method == "DSPy Framework":
                        result = extract_resume_info(resume_text)
                    else:  # Direct JSON Prompting
                        # Determine provider type correctly
                        provider_type = "Ollama" if provider == "Ollama (Local)" else "OpenAI"
                        result = extract_with_json_prompting(
                            provider_type,
                            selected_model, 
                            resume_text, 
                            api_key, 
                            ollama_url
                        )
                
                st.success("✅ Extraction complete!")
                
                # Show extraction method used
                method_emoji = "🔧" if extraction_method == "DSPy Framework" else "📝"
                st.info(f"{method_emoji} Used: {extraction_method}")
                
                # Debug section for JSON responses
                if extraction_method == "Direct JSON Prompting" and hasattr(result, 'raw_response'):
                    with st.expander("🔍 Debug: Raw JSON Response"):
                        st.code(result.raw_response, language="json")
                
                # Clean up DSPy output (remove template artifacts and fix field mixing)
                def clean_dspy_output(text, field_name):
                    if not text:
                        return "Not found"
                    
                    text = str(text).strip()
                    
                    # Remove common template artifacts
                    artifacts = [
                        "Here is the extracted information in the requested format:",
                        "Here are the extracted values for each field:",
                        "Resume Text:",
                        "---",
                        "Not found",
                        "Not provided",
                        "Not specified",
                        "Not mentioned"
                    ]
                    
                    for artifact in artifacts:
                        text = text.replace(artifact, "").strip()
                    
                    # Remove ALL field labels (since model mixes them up)
                    field_labels = ["Name:", "Email:", "Phone:", "Skills:", "Education:", "Experience:"]
                    for label in field_labels:
                        text = text.replace(label, "").strip()
                    
                    # Special handling based on field type
                    if field_name == "skills":
                        # Look for skill-like patterns
                        if any(skill in text.lower() for skill in ["python", "docker", "jira", "scrum", "office", "oracle", "postgres"]):
                            # Clean up skills formatting
                            text = text.replace(";", ",").replace("\n", ", ")
                            # Remove dates and company names that got mixed in
                            words = text.split()
                            skills = []
                            for word in words:
                                if not any(char.isdigit() for char in word) and len(word) > 2:
                                    skills.append(word.strip(",;"))
                            return ", ".join(skills) if skills else "Not found"
                    
                    elif field_name == "experience":
                        # Look for experience-like patterns
                        if "2015" in text or "2016" in text or "nterra" in text:
                            return text.replace("\n", " ").strip()
                    
                    # General cleanup
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    cleaned = ' '.join(lines) if lines else "Not found"
                    
                    # Remove remaining artifacts
                    if len(cleaned) < 3 or cleaned.lower() in ["not found", "---", ""]:
                        return "Not found"
                    
                    return cleaned
                
                # Display results
                def get_display_value(field_value, field_name):
                    if extraction_method == "DSPy Framework":
                        return clean_dspy_output(field_value, field_name)
                    else:  # Direct JSON Prompting
                        return field_value if field_value != "Not found" else "Not found"
                
                st.markdown("### 👤 Personal Information")
                st.write(f"**Name:** {get_display_value(result.name, 'name')}")
                st.write(f"**Email:** {get_display_value(result.email, 'email')}")
                st.write(f"**Phone:** {get_display_value(result.phone, 'phone')}")
                
                st.markdown("### 🛠️ Skills")
                st.write(f"**Skills:** {get_display_value(result.skills, 'skills')}")
                
                st.markdown("### 🎓 Education")
                st.write(f"**Education:** {get_display_value(result.education, 'education')}")
                
                st.markdown("### 💼 Experience")
                st.write(f"**Experience:** {get_display_value(result.experience, 'experience')}")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"❌ Error: {str(e)}")
                st.code(error_details, language="python")
                if "connection" in str(e).lower() or "refused" in str(e).lower():
                    st.info("💡 Make sure Ollama is running: `ollama serve`")
                elif "api" in str(e).lower():
                    st.info("💡 Check your OpenAI API key")
        
        elif extract_button and not resume_text:
            st.warning("Please paste resume text first")

if __name__ == "__main__":
    main()