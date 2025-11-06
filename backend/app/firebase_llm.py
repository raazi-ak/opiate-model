"""
Gemini API client (free tier, no billing required)
Replaces all local LLM clients
"""
import os
from typing import Optional, Iterator
import google.generativeai as genai

from dotenv import load_dotenv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.abspath(os.path.join(APP_DIR, "..", ".env"))
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


class VertexAIClient:
    """Gemini Developer API client (free, no billing required)"""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini Developer API client (free, no billing)
        
        Args:
            model: Model name (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro')
                  Note: Gemini 1.5 models are retired, use 2.5 models
        """
        self.model_name = os.getenv("GEMINI_MODEL", model)
        
        # Use Gemini Developer API with API key (free, no billing required)
        # This is the recommended approach per Firebase AI Logic documentation
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable must be set. "
                "Get a free API key from: https://aistudio.google.com/app/apikey\n"
                "The Gemini Developer API is free (no billing required) on the Spark plan."
            )
        
        # Configure Gemini API with the API key
        genai.configure(api_key=api_key)
        
        # Initialize model using Gemini Developer API
        self.model = genai.GenerativeModel(self.model_name)
    
    def build_prompt(self, user_message: str, objective: Optional[str] = None,
                    retrieved_context: str = "", stress_level: Optional[str] = None,
                    heart_rate: Optional[int] = None, session_id: Optional[str] = None) -> str:
        """
        Build prompt with context, objective, and stress level
        
        Args:
            user_message: User's question
            objective: Learning objective
            retrieved_context: Retrieved document context
            stress_level: Current stress level
            session_id: Optional session ID to indicate fresh context
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # Add session context (fresh start indicator)
        if session_id:
            parts.append("=== NEW SESSION CONTEXT ===\n")
            parts.append("This is a fresh conversation session. You have NO memory or context from previous sessions.\n")
            parts.append("You can ONLY use the study materials provided below. Do NOT reference any files, topics, or information from previous sessions.\n")
            parts.append(f"Session ID: {session_id}\n\n")
        
        # Add objective if provided
        if objective:
            parts.append(f"Learning Objective: {objective}\n")
        
        # Add stress level context (for LLM to adapt, but NOT to mention in responses)
        if stress_level:
            stress_context = {
                "low": "The user is feeling relaxed and focused. Provide detailed explanations with examples.",
                "medium": "The user is moderately stressed. Focus on key concepts with clear, structured explanations.",
                "high": "The user is feeling overwhelmed. Keep responses concise and practical. Focus on most important points. Suggest taking a 5-minute break after this session."
            }.get(stress_level.lower(), "")
            if stress_context:
                parts.append(f"User Context (for adaptation only, do NOT mention in response): {stress_context}\n")
                # Store stress level for answering questions about it (only if explicitly asked)
                parts.append(f"IMPORTANT: Current User Stress Level: {stress_level} (only mention if user explicitly asks about it)\n")
                if heart_rate is not None:
                    parts.append(f"IMPORTANT: Current User Heart Rate: {heart_rate} bpm (only mention if user explicitly asks about it)\n")
        
        # Add retrieved context
        if retrieved_context:
            parts.append("Relevant Study Materials:\n" + retrieved_context + "\n")
            # If user is asking about files, explicitly point out the files in the context
            query_lower = user_message.lower()
            if any(phrase in query_lower for phrase in ["what files", "what file", "which files", "list files", "files do you have", "what did i upload"]):
                parts.append("IMPORTANT: The user is asking about what files they uploaded. The 'Source File' entries above show the files available in this session. List them clearly.\n\n")
        else:
            # Explicitly state no materials if context is empty
            parts.append("Relevant Study Materials: NONE\n")
            parts.append("IMPORTANT: No study materials were retrieved. You should inform the user that no relevant documents were found in the current session.\n\n")
        
        # Add user message
        parts.append(f"User Question: {user_message}\n")
        
        # Add instructions with guardrails
        instructions = """Instructions:
1. SESSION ISOLATION: This is a fresh session. You have NO knowledge of files, topics, or conversations from previous sessions. ONLY use the study materials provided above.

2. GENERAL/SYSTEM QUESTIONS: You CAN answer questions about:
   - User's current stress level (ONLY if explicitly asked - use the "Current User Stress Level" information provided above)
   - Heart rate (ONLY if explicitly asked - use the "Current User Heart Rate" information provided above if available)
   - Session information
   - System status
   - General knowledge questions (when not related to study materials)
   These do NOT require study materials and you should answer them directly.
   
   IMPORTANT: Do NOT mention stress level or heart rate in your responses unless the user explicitly asks about them. These are only provided as context for you to adapt your response style (detailed vs concise, etc.).

3. FILE LISTING QUERIES: If the user asks "what files did I upload", "what files do you have", "list files", or similar:
   - Look at the "Source File:" entries in the "Relevant Study Materials" section above
   - List ALL unique file names you see there
   - These are the files available in the current session
   - Do NOT say "no files" if you see Source File entries above

4. PRIMARY SOURCE: For questions about study materials, answer using ONLY the information from the retrieved study materials above. Base your response primarily on what is provided in the context.

5. MULTI-TOPIC QUESTIONS: If the question asks about multiple different topics:
   - Recognize that these are separate topics
   - Retrieve information for each topic from the relevant source documents
   - Clearly state which information came from which source file/document
   - Structure your response to show: "Topic 1 (from [Source File 1]): ..." and "Topic 2 (from [Source File 2]): ..."
   - If a topic is not found in any materials, clearly state that and which topic is missing

6. MATERIAL-SPECIFIC FIRST: If the retrieved materials contain sufficient information to answer the question:
   - Answer using ONLY that information
   - Cite the specific source document filename (e.g., "from 1 Assignment-2023.pdf")
   - Do NOT add external knowledge unless explicitly requested by the user
   - Do NOT reference files that are NOT in the "Relevant Study Materials" section above

7. INSUFFICIENT MATERIAL HANDLING: If the retrieved materials do NOT contain enough information to fully answer the question:
   - First, clearly state: "The provided study materials do not contain sufficient information to fully answer this question" (or specify which part/topic is missing)
   - Then, provide additional verified knowledge to supplement the answer
   - Clearly distinguish between what came from the materials vs. additional knowledge
   - Use phrases like "Based on the study materials from [filename]:" and "Additional verified knowledge (not in materials):"

8. ADDITIONAL INFO ONLY IF REQUESTED: Only provide supplementary information beyond the study materials if:
   - The user explicitly asks for additional information, OR
   - The materials are insufficient (as described in point 5)

9. ADAPT TO STRESS LEVEL:
   - Low stress: Provide detailed explanations with examples
   - Medium stress: Focus on key concepts with clear, structured explanations
   - High stress: Keep it concise and practical. Focus on most important points.

10. SOURCE ATTRIBUTION: Always cite the specific source filename when referencing information from the study materials. Use format: "According to [filename]:" or "From [filename]:" or "Based on [filename]:"

11. NO HALLUCINATED FILES: Do NOT mention or reference any files that are NOT explicitly listed in the "Relevant Study Materials" section above. If you don't see a file mentioned there, it does NOT exist in this session.

12. STRESS LEVEL QUESTIONS: If the user explicitly asks about their stress level, heart rate, or current state:
   - Use the "Current User Stress Level" and "Current User Heart Rate" information provided above
   - You can answer these questions directly without needing study materials
   - Be helpful and provide context about what the stress level means
   
13. DO NOT MENTION STRESS/HR UNLESS ASKED: 
   - Do NOT greet the user with their stress level or heart rate
   - Do NOT include stress/heart rate information in your responses unless explicitly asked
   - These are provided ONLY as context for you to adapt your response style (detailed vs concise explanations)
   - Focus on answering the user's actual question, not on their physiological state"""
        
        parts.append(instructions)
        
        return "\n".join(parts)
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text response from Gemini API
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        try:
            # Use Gemini Developer API (free, no billing)
            response = self.model.generate_content(prompt)
            return response.text if response.text else "[No response generated]"
        except Exception as e:
            return f"[Gemini API error] {str(e)}"
    
    def stream_text(self, prompt: str) -> Iterator[str]:
        """
        Stream text response from Gemini API
        
        Args:
            prompt: Input prompt
            
        Yields:
            Text chunks as they are generated
        """
        try:
            # Use Gemini Developer API with streaming (free, no billing)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=2048),
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            # Don't expose error details to user, just log and continue silently
            print(f"⚠️  Gemini API streaming error: {e}")
            import traceback
            traceback.print_exc()
            # Don't yield error message - let frontend show "thinking..." instead
            return

