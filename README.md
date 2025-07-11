# Resume Parser with DSPy

A Streamlit application for parsing resumes/CVs from raw text using DSPy framework. Supports both local Ollama models and remote OpenAI API.

## Features

- 📄 **Resume parsing** - Extract structured information (name, email, phone, skills, education, experience)
- 🏠 **Local models** - Use Ollama for privacy and cost savings
- ☁️ **Remote models** - Use OpenAI API for better accuracy
- 🐳 **Docker support** - Easy deployment and consistent environment
- 🔧 **Clean DSPy implementation** - Declarative programming approach

## Quick Start

### Option 1: Docker (Recommended)

1. **Build and run:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   Open your browser and go to `http://localhost:8501`

### Option 2: Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Configuration Options

### 🏠 Local Ollama Setup

1. **Install Ollama:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model:**
   ```bash
   ollama pull llama3.2:3b
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Configure in app:**
   - Select "Ollama (Local)" in the sidebar
   - Choose your model
   - Test the connection

### ☁️ OpenAI API Setup

1. **Get an API key** from [OpenAI](https://platform.openai.com/api-keys)

2. **Configure in app:**
   - Select "OpenAI (Remote)" in the sidebar
   - Enter your API key
   - Choose your model
   - Test the connection

## Supported Models

### Local Ollama Models
- **Llama 3.2 3B** - Good balance of speed and accuracy
- **Llama 3.2 1B** - Fastest, basic accuracy
- **Llama 3.1 8B** - Best accuracy, slower
- **Qwen 2.5 3B** - Alternative option
- **Phi 3.5** - Microsoft's model

### OpenAI Models
- **GPT-4o Mini** - Recommended, fast and affordable
- **GPT-4o** - Best accuracy, more expensive
- **GPT-3.5 Turbo** - Budget option

## Usage

1. **Configure your model** in the sidebar
2. **Test the connection** to ensure everything works
3. **Paste resume text** in the input area
4. **Click "Extract Information"** to parse the resume
5. **View results** in the structured output

## Docker Network Configuration

If using Ollama with Docker, you may need to adjust network settings:

### For Ollama on host machine:
Uncomment this line in `docker-compose.yml`:
```yaml
network_mode: "host"
```

### For Ollama in another container:
Update the Ollama URL in the app to point to the container name.

## Troubleshooting

### Ollama Issues
- **Connection refused**: Make sure Ollama is running (`ollama serve`)
- **Model not found**: Pull the model first (`ollama pull model-name`)
- **Network issues**: Check if Docker can access host network

### OpenAI Issues
- **API key invalid**: Verify your API key is correct
- **Rate limits**: Check your OpenAI usage limits
- **Network issues**: Ensure internet connectivity

## Examples

Try the app with this sample resume text:

```
John Smith
Software Engineer
Email: john.smith@email.com
Phone: (555) 123-4567

Skills: Python, JavaScript, React, Docker, AWS

Education:
Bachelor of Computer Science, MIT, 2020

Experience:
Senior Developer at TechCorp (2021-2024)
- Built scalable web applications
- Led team of 5 engineers
```

## Technical Details

The app uses DSPy framework with:
- **Simple signatures** for clear input/output definitions
- **Chain-of-Thought** prompting for better reasoning
- **Modular design** for easy extension and maintenance

This follows the declarative programming approach shown in the DSPy documentation and tutorials.