from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import json
import time
import requests
import os
import threading
from flask_socketio import SocketIO, emit
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'paia-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Create static folder if it doesn't exist
os.makedirs(app.static_folder, exist_ok=True)

# Store active simulations
active_simulations = {}

class PAIASimulationEngine:
    def __init__(self, config: dict, api_key: str, simulate_humans: bool = False, session_id: str = None, controlled_humans: list = None):
        """
        Initialize the PAIA Simulation Engine
        
        Args:
            config: Configuration dictionary
            api_key: Google Gemini API key
            simulate_humans: Whether to simulate human responses with the LLM
            session_id: Unique session ID for this simulation
            controlled_humans: List of human actor IDs to be controlled directly by users
        """
        self.api_key = api_key
        self.simulate_humans = simulate_humans
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp-02-05:generateContent"
        self.config = config
        self.conversations = {}  # Store conversation histories
        self.actor_map = {actor["id"]: actor for actor in self.config["actors"]}
        self.session_id = session_id or str(uuid.uuid4())
        self.is_running = False
        self.pending_human_input = None
        self.human_response = None
        self.human_actors_control = {}  # Which human actors are controlled by real users
        self.is_replay = api_key == "IMPORTED_SIMULATION"
        self.replay_index = {}  # Current index for replaying each conversation
        self.replay_flattened = []  # Flattened list of messages for replay (with order info)
        self.replay_position = 0  # Position in the flattened replay list
        
        # Set up controlled human actors
        if controlled_humans:
            for human_id in controlled_humans:
                if human_id in self.actor_map and self.actor_map[human_id]['type'] == 'human':
                    self.human_actors_control[human_id] = True
        
        logger.info(f"Initialized simulation: {self.config['name']} (Session: {self.session_id})")
        logger.info(f"Controlled humans: {list(self.human_actors_control.keys())}")
        if self.is_replay:
            logger.info("This is a replay of an imported simulation")
    
    def initialize_conversations(self):
        """Set up conversation histories for each pair of actors that interact"""
        # Initialize all possible conversation pairs
        for actor_a in self.config["actors"]:
            for actor_b in self.config["actors"]:
                if actor_a["id"] != actor_b["id"]:
                    conv_key = f"{actor_a['id']}_{actor_b['id']}"
                    self.conversations[conv_key] = []
                    
                    # Initialize with system prompts
                    self.conversations[conv_key].append({
                        "role": "user",
                        "parts": [{"text": f"SYSTEM PROMPT: {actor_a.get('systemPrompt', '')}"}]
                    })
                    self.conversations[conv_key].append({
                        "role": "user", 
                        "parts": [{"text": f"SYSTEM PROMPT: {actor_b.get('systemPrompt', '')}"}]
                    })
    
    def get_actor_name(self, actor_id: str) -> str:
        """Get the name of an actor by ID"""
        actor = self.actor_map.get(actor_id)
        return actor["name"] if actor else actor_id
    
    def get_actor_type(self, actor_id: str) -> str:
        """Get the type of an actor by ID (human or ai)"""
        actor = self.actor_map.get(actor_id)
        return actor["type"] if actor else "unknown"
    
    def _send_to_gemini(self, messages):
        """Send a conversation to the Gemini API and get a response"""
        headers = {"Content-Type": "application/json"}
        payload = {"contents": messages}
        
        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Gemini API: {response.status_code}")
                logger.error(response.text)
                return "Error getting response from AI."
            
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No response generated."
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return "Error connecting to AI service."
    
    def wait_for_human_input(self, prompt, source_name, target_name, target_id=None):
        """Wait for human input from the web interface"""
        self.pending_human_input = {
            "prompt": prompt,
            "source": source_name,
            "target": target_name,
            "targetId": target_id
        }
        
        # Emit an event to request input from the user
        socketio.emit('request_human_input', self.pending_human_input, room=self.session_id)
        
        # Wait for response
        while self.human_response is None and self.is_running:
            time.sleep(0.5)
        
        response = self.human_response
        self.human_response = None
        self.pending_human_input = None
        return response
    
    def submit_human_input(self, response):
        """Submit human input from the web interface"""
        self.human_response = response
    
    def process_interaction(self, source_id, target_id, message=None):
        """
        Process an interaction between a source and target actor
        
        Args:
            source_id: ID of the source actor
            target_id: ID of the target actor
            message: Optional message (for initial prompts or human input)
            
        Returns:
            The response from the target actor
        """
        source_name = self.get_actor_name(source_id)
        target_name = self.get_actor_name(target_id)
        source_type = self.get_actor_type(source_id)
        target_type = self.get_actor_type(target_id)
        
        conv_key = f"{source_id}_{target_id}"
        
        # If this is the first message and no message is provided, check for initial prompt
        if not message and len(self.conversations[conv_key]) == 2:  # Only system prompts
            source_actor = self.actor_map[source_id]
            if source_actor.get("initialPrompt"):
                message = source_actor["initialPrompt"]
                socketio.emit('simulation_update', {
                    'type': 'message',
                    'content': f"Using initial prompt from {source_name}: {message}"
                }, room=self.session_id)
            else:
                if source_type == "human" and (not self.simulate_humans or source_id in self.human_actors_control):
                    message = self.wait_for_human_input(
                        f"Enter message from {source_name} to {target_name}:", 
                        source_name, target_name, 
                        target_id
                    )
                else:
                    # Generate a starting message based on system prompts
                    temp_conv = self.conversations[conv_key].copy()
                    temp_conv.append({
                        "role": "user",
                        "parts": [{"text": f"Based on your role, generate a starting message to {target_name}. Keep it brief and natural."}]
                    })
                    message = self._send_to_gemini(temp_conv)
                    socketio.emit('simulation_update', {
                        'type': 'message',
                        'content': f"Generated initial message from {source_name}: {message}"
                    }, room=self.session_id)
        
        # Add source message to conversation
        if message:
            self.conversations[conv_key].append({
                "role": "user",
                "parts": [{"text": message}]
            })
            socketio.emit('simulation_update', {
                'type': 'message',
                'from': source_name,
                'to': target_name,
                'content': message
            }, room=self.session_id)
        
        # Get response based on actor type
        response = ""
        if target_type == "human" and (not self.simulate_humans or target_id in self.human_actors_control):
            # Request human input from the web interface
            response = self.wait_for_human_input(
                f"{source_name} says: {message}\nWhat is {target_name}'s response?",
                source_name, target_name, 
                target_id
            )
        else:
            socketio.emit('simulation_update', {
                'type': 'status',
                'content': f"Calling API for {target_name}'s response..."
            }, room=self.session_id)
            
            time.sleep(1)  # To avoid rate limiting
            response = self._send_to_gemini(self.conversations[conv_key])
        
        # Add response to conversation history
        self.conversations[conv_key].append({
            "role": "model" if target_type == "ai" else "user",
            "parts": [{"text": response}]
        })
        
        # Emit the response
        socketio.emit('simulation_update', {
            'type': 'response',
            'from': target_name,
            'to': source_name,
            'content': response
        }, room=self.session_id)
        
        return response
    
    def get_conversation_history(self, source_id=None, target_id=None):
        """
        Get the conversation history between two actors or all conversations
        
        Args:
            source_id: Source actor ID (optional)
            target_id: Target actor ID (optional)
            
        Returns:
            Dictionary of conversation histories
        """
        if source_id and target_id:
            # Return specific conversation
            conv_key = f"{source_id}_{target_id}"
            if conv_key in self.conversations:
                return {conv_key: self.conversations[conv_key]}
            else:
                return {}
        else:
            # Return all conversations with actual messages (more than just system prompts)
            result = {}
            for key, conv in self.conversations.items():
                if len(conv) > 2:  # More than just system prompts
                    result[key] = conv
            return result
            
    def run_simulation(self, max_rounds=2):
        """
        Run the simulation for the specified number of rounds
        
        Args:
            max_rounds: Maximum number of rounds to run
        """
        max_rounds = max(1, min(10, max_rounds))  # Ensure between 1 and 10 rounds
        
        self.is_running = True
        self.initialize_conversations()
        
        try:
            for round_num in range(1, max_rounds + 1):
                socketio.emit('simulation_update', {
                    'type': 'round_start',
                    'round': round_num,
                    'total_rounds': max_rounds
                }, room=self.session_id)
                
                # Process each interaction in the specified order
                for interaction in self.config["interactions"]:
                    source_id = interaction["source"]
                    target_id = interaction["target"]
                    source_name = self.get_actor_name(source_id)
                    target_name = self.get_actor_name(target_id)
                    
                    socketio.emit('simulation_update', {
                        'type': 'interaction_start',
                        'source': source_name,
                        'target': target_name
                    }, room=self.session_id)
                    
                    # For first round, send initial message
                    if round_num == 1:
                        response = self.process_interaction(source_id, target_id)
                    else:
                        # For subsequent rounds, get last message from reverse conversation
                        reverse_key = f"{target_id}_{source_id}"
                        if reverse_key in self.conversations and len(self.conversations[reverse_key]) > 2:
                            last_message = self.conversations[reverse_key][-1]["parts"][0]["text"]
                            response = self.process_interaction(source_id, target_id, 
                                                              f"[Previous message: {last_message}] Continue the conversation.")
                        else:
                            response = self.process_interaction(source_id, target_id, 
                                                              "Continue the conversation.")
                    
                    time.sleep(1)  # Brief pause between interactions
                
                # Slight pause between rounds
                time.sleep(2)
            
            socketio.emit('simulation_update', {
                'type': 'simulation_end',
                'content': "Simulation completed."
            }, room=self.session_id)
            
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            socketio.emit('simulation_update', {
                'type': 'error',
                'content': f"Error during simulation: {str(e)}"
            }, room=self.session_id)
        
        finally:
            self.is_running = False

def save_frontend_file(html_content):
    """Save the frontend HTML to serve it later"""
    with open(os.path.join(app.static_folder, 'paia-designer.html'), 'w') as f:
        f.write(html_content)
    return 'static/paia-designer.html'

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/generate-schema', methods=['POST'])
def generate_schema():
    """Generate a complete schema including actors and interactions based on scenario description"""
    try:
        data = request.json
        scenario_name = data.get('name', 'Untitled Scenario')
        scenario_description = data.get('description', '')
        
        if not scenario_description:
            return jsonify({'error': 'Missing scenario description'}), 400
            
        # Prepare the prompt for the LLM
        prompt = f"""
Generate a complete interaction schema for the following scenario:

Scenario Name: {scenario_name}
Scenario Description: {scenario_description}

I need:
1. A set of actors (humans and AI assistants) that would be involved in this scenario
2. The connections/interactions between these actors
3. System prompts for each actor

Please generate a detailed response in the following JSON format:

```json
{{
  "actors": [
    {{
      "id": "human-1",
      "name": "Person A",
      "type": "human",
      "description": "Brief description of this person's role",
      "systemPrompt": "Detailed system prompt for this human actor",
      "initialPrompt": "Optional initial prompt if this actor starts the interaction",
      "position": {{ "x": 100, "y": 100 }}
    }},
    {{
      "id": "ai-1",
      "name": "AI Assistant A",
      "type": "ai",
      "description": "Brief description of this AI's purpose",
      "systemPrompt": "Detailed system prompt for this AI actor",
      "position": {{ "x": 300, "y": 100 }}
    }},
    // Add more actors as needed
  ],
  "interactions": [
    {{
      "source": "human-1",
      "target": "ai-1"
    }},
    // Add all necessary interactions
  ]
}}
```

Each actor should have:
- A unique ID (prefixed with 'human-' or 'ai-')
- A name
- A type ("human" or "ai")
- A meaningful position on the canvas (x, y coordinates)
- A detailed system prompt (at least 50 words)
- An initial prompt if the actor initiates the interaction

Each interaction should specify which actor initiates (source) and which receives (target).

Create a realistic schema with at least 2-6 actors. System prompts should be detailed and reflect each actor's role in the scenario.
"""
        
        # Call the API
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
            
        headers = {"Content-Type": "application/json"}
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp-02-05:generateContent"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        response = requests.post(
            f"{api_url}?key={api_key}",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error from Gemini API: {response.status_code}")
            logger.error(response.text)
            return jsonify({'error': 'Error generating schema from API'}), 500
            
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON from the response
            import re
            json_match = re.search(r'```json(.*?)```', generated_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```(.*?)```', generated_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return jsonify({'error': 'Could not extract schema from API response'}), 500
            
            try:
                schema = json.loads(json_str)
                
                # Validate the schema
                if 'actors' not in schema or not isinstance(schema['actors'], list):
                    return jsonify({'error': 'Invalid schema: missing actors array'}), 500
                    
                if 'interactions' not in schema or not isinstance(schema['interactions'], list):
                    return jsonify({'error': 'Invalid schema: missing interactions array'}), 500
                
                # Check actor IDs and add if missing
                for i, actor in enumerate(schema['actors']):
                    if 'id' not in actor:
                        prefix = 'human-' if actor.get('type') == 'human' else 'ai-'
                        actor['id'] = f"{prefix}{i+1}"
                    
                    # Ensure position is valid
                    if 'position' not in actor or not isinstance(actor['position'], dict):
                        actor['position'] = {"x": 100 + (i * 200), "y": 100 + (i % 2) * 150}
                
                return jsonify({
                    'success': True,
                    'schema': schema
                })
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON schema: {e}")
                return jsonify({'error': f'Invalid JSON in response: {str(e)}'}), 500
        else:
            return jsonify({'error': 'No response from model.'}), 500
            
    except Exception as e:
        logger.error(f"Error generating schema: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-prompts', methods=['POST'])
def generate_prompts():
    """Generate system prompts for actors based on scenario description"""
    try:
        data = request.json
        scenario_name = data.get('name', 'Untitled Scenario')
        scenario_description = data.get('description', '')
        actors = data.get('actors', [])
        
        if not scenario_description:
            return jsonify({'error': 'Missing scenario description'}), 400
            
        if not actors or len(actors) == 0:
            return jsonify({'error': 'No actors provided'}), 400
            
        # Prepare the prompt for the LLM
        prompt = f"""
Generate system prompts for the following actors in a conversation scenario.

Scenario Name: {scenario_name}
Scenario Description: {scenario_description}

Actors:
"""
        
        for actor in actors:
            actor_type = "Human" if actor['type'] == 'human' else "AI Assistant"
            prompt += f"- {actor['name']} ({actor_type}): {actor.get('description', 'No description')}\n"
            
        prompt += """
For each actor, generate a detailed system prompt that defines their role, personality, knowledge, goals and constraints.
For AI assistants, focus on their purpose, capabilities, and interaction style.
For humans, focus on their background, motivations, and relationship to other actors.

Return the results in the following format:
{actor_name_1}: {system_prompt_1}
{actor_name_2}: {system_prompt_2}
...

System prompts should be detailed (at least 50 words each) but concise, and reflect the nature of the scenario.
Do not include any additional text, just the actor names and their system prompts.
"""
        
        # Call the API
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
            
        headers = {"Content-Type": "application/json"}
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp-02-05:generateContent"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        response = requests.post(
            f"{api_url}?key={api_key}",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error from Gemini API: {response.status_code}")
            logger.error(response.text)
            return jsonify({'error': 'Error generating prompts from API'}), 500
            
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parse the generated text to extract prompts for each actor
            actor_prompts = {}
            current_actor = None
            current_prompt = []
            
            for line in generated_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line starts with an actor name
                for actor in actors:
                    if line.startswith(f"{actor['name']}:") or line.startswith(f"{actor['name']} -") or line.startswith(f"**{actor['name']}**:"):
                        # If we were already building a prompt, save it
                        if current_actor:
                            actor_prompts[current_actor] = '\n'.join(current_prompt).strip()
                        
                        # Start a new prompt for this actor
                        current_actor = actor['name']
                        prefix_length = len(current_actor) + 1  # +1 for the colon
                        
                        # Handle different formats
                        if line.startswith(f"**{actor['name']}**:"):
                            prefix_length = len(f"**{actor['name']}**:")
                        elif line.startswith(f"{actor['name']} -"):
                            prefix_length = len(f"{actor['name']} -")
                            
                        current_prompt = [line[prefix_length:].strip()]
                        break
                else:
                    # If no actor name found, append to current prompt
                    if current_actor:
                        current_prompt.append(line)
            
            # Save the last prompt if there is one
            if current_actor and current_actor not in actor_prompts:
                actor_prompts[current_actor] = '\n'.join(current_prompt).strip()
                
            # Map prompts back to actors
            for actor in actors:
                if actor['name'] in actor_prompts:
                    actor['generatedSystemPrompt'] = actor_prompts[actor['name']]
                else:
                    # Generate a fallback prompt if the API didn't provide one
                    if actor['type'] == 'human':
                        actor['generatedSystemPrompt'] = f"You are {actor['name']}, a human participant in this conversation."
                    else:
                        actor['generatedSystemPrompt'] = f"You are {actor['name']}, an AI assistant in this conversation."
            
            return jsonify({
                'success': True,
                'actors': actors
            })
        else:
            return jsonify({'error': 'No response from model.'}), 500
            
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-simulation', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    try:
        data = request.json
        config = data.get('config')
        api_key = data.get('api_key')
        simulate_humans = data.get('simulate_humans', False)
        rounds = max(1, min(10, data.get('rounds', 2)))  # Limit to 1-10 rounds
        controlled_humans = data.get('controlled_humans', [])
        
        if not config or not api_key:
            return jsonify({'error': 'Missing configuration or API key'}), 400
        
        # Create a unique session ID for this simulation
        session_id = str(uuid.uuid4())
        
        # Initialize the simulation engine
        engine = PAIASimulationEngine(config, api_key, simulate_humans, session_id, controlled_humans)
        active_simulations[session_id] = engine
        
        # Start the simulation in a separate thread
        threading.Thread(target=engine.run_simulation, args=(rounds,)).start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Simulation started'
        })
    
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-simulation/<session_id>', methods=['POST'])
def stop_simulation(session_id):
    """Stop a running simulation"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        engine.is_running = False
        del active_simulations[session_id]
        return jsonify({'success': True, 'message': 'Simulation stopped'})
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/human-input/<session_id>', methods=['POST'])
def submit_human_input(session_id):
    """Submit human input for a running simulation"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        response = request.json.get('response')
        
        if engine.pending_human_input:
            engine.submit_human_input(response)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'No pending input request'}), 400
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/direct-message/<session_id>', methods=['POST'])
def send_direct_message(session_id):
    """Send a direct message from one actor to another"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        data = request.json
        source_id = data.get('source_id')
        target_id = data.get('target_id')
        message = data.get('message')
        
        if not source_id or not target_id or not message:
            return jsonify({'error': 'Missing source_id, target_id, or message'}), 400
        
        # Check if source is a controlled human
        if source_id not in engine.human_actors_control:
            return jsonify({'error': 'Source actor is not controlled by user'}), 403
            
        # Process the interaction
        try:
            response = engine.process_interaction(source_id, target_id, message)
            return jsonify({
                'success': True,
                'response': response
            })
        except Exception as e:
            logger.error(f"Error processing direct message: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/conversation-history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get conversation history for a running or completed simulation"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        source_id = request.args.get('source')
        target_id = request.args.get('target')
        
        history = engine.get_conversation_history(source_id, target_id)
        
        # Convert the conversation history to a more readable format
        formatted_history = {}
        for conv_key, messages in history.items():
            source_id, target_id = conv_key.split('_')
            source_name = engine.get_actor_name(source_id)
            target_name = engine.get_actor_name(target_id)
            
            # Skip the first two system prompt messages
            formatted_messages = []
            for i in range(2, len(messages)):
                msg = messages[i]
                role = "AI" if msg["role"] == "model" else source_name if i % 2 == 0 else target_name
                formatted_messages.append({
                    "role": role,
                    "content": msg["parts"][0]["text"],
                    "timestamp": time.time()  # We don't have actual timestamps, so use current time
                })
            
            formatted_history[conv_key] = {
                "source": source_name,
                "target": target_name,
                "sourceId": source_id,
                "targetId": target_id,
                "messages": formatted_messages
            }
        
        return jsonify({
            'success': True,
            'history': formatted_history
        })
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/export-simulation/<session_id>', methods=['GET'])
def export_simulation(session_id):
    """Export a simulation with all conversations"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        
        # Create export data structure
        export_data = {
            'config': engine.config,
            'conversations': {},
            'timestamp': time.time(),
            'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_id': session_id
        }
        
        # Add conversation histories
        for conv_key, messages in engine.conversations.items():
            if len(messages) > 2:  # More than just system prompts
                source_id, target_id = conv_key.split('_')
                source_name = engine.get_actor_name(source_id)
                target_name = engine.get_actor_name(target_id)
                
                # Format the messages
                formatted_messages = []
                for i in range(2, len(messages)):
                    msg = messages[i]
                    role = "AI" if msg["role"] == "model" else source_name if i % 2 == 0 else target_name
                    formatted_messages.append({
                        "role": role,
                        "content": msg["parts"][0]["text"],
                        "timestamp": time.time()  # We don't have actual timestamps, so use current time
                    })
                
                export_data['conversations'][conv_key] = {
                    "source": source_name,
                    "target": target_name,
                    "sourceId": source_id,
                    "targetId": target_id,
                    "messages": formatted_messages
                }
        
        return jsonify({
            'success': True,
            'data': export_data
        })
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/import-simulation', methods=['POST'])
def import_simulation():
    """Import a previously exported simulation"""
    try:
        data = request.json
        imported_data = data.get('data')
        
        if not imported_data or 'config' not in imported_data or 'conversations' not in imported_data:
            return jsonify({'error': 'Invalid simulation data format'}), 400
        
        # Create a unique session ID for this imported simulation
        session_id = str(uuid.uuid4())
        
        # Initialize the simulation engine with the imported configuration
        engine = PAIASimulationEngine(
            imported_data['config'], 
            "IMPORTED_SIMULATION",  # No API key needed for imported simulations
            True,  # simulate_humans is irrelevant for imported sims
            session_id
        )
        
        # Initialize the conversation structure
        engine.initialize_conversations()
        
        # Import the conversations
        for conv_key, conversation in imported_data['conversations'].items():
            source_id = conversation['sourceId']
            target_id = conversation['targetId']
            
            # Reconstruct the conversation history
            if conv_key in engine.conversations:
                # Add the messages (skipping the first two system prompts)
                for msg in conversation['messages']:
                    if msg['role'] == conversation['source']:
                        # Source message
                        engine.conversations[conv_key].append({
                            "role": "user",
                            "parts": [{"text": msg['content']}]
                        })
                    else:
                        # Target message
                        engine.conversations[conv_key].append({
                            "role": "model" if engine.get_actor_type(target_id) == "ai" else "user",
                            "parts": [{"text": msg['content']}]
                        })
        
        # Prepare flattened message list for replay
        prepare_replay_sequence(engine, imported_data)
        
        # Store the engine in the active simulations
        active_simulations[session_id] = engine
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Simulation imported successfully',
            'replay_available': True,
            'replay_message_count': len(engine.replay_flattened)
        })
    except Exception as e:
        logger.error(f"Error importing simulation: {e}")
        return jsonify({'error': str(e)}), 500

def prepare_replay_sequence(engine, imported_data):
    """Prepare a flattened sequence of messages for replay"""
    flattened_messages = []
    
    # Process each conversation
    for conv_key, conversation in imported_data['conversations'].items():
        source_id = conversation['sourceId']
        target_id = conversation['targetId']
        source_name = conversation['source']
        target_name = conversation['target']
        
        # Track alternating messages
        for i, msg in enumerate(conversation['messages']):
            sender_id = source_id if msg['role'] == source_name else target_id
            receiver_id = target_id if msg['role'] == source_name else source_id
            
            # Add to flattened list with timestamp (if available) or sequence number
            flattened_messages.append({
                'source_id': sender_id,
                'target_id': receiver_id,
                'source_name': msg['role'],
                'target_name': source_name if msg['role'] == target_name else target_name,
                'content': msg['content'],
                'timestamp': msg.get('timestamp', i),  # Use timestamp if available, otherwise use sequence
                'conv_key': conv_key,
                'message_index': i
            })
    
    # Sort by timestamp
    flattened_messages.sort(key=lambda x: x['timestamp'])
    
    # Store in engine
    engine.replay_flattened = flattened_messages
    engine.replay_position = 0

@app.route('/api/replay-step/<session_id>', methods=['POST'])
def replay_step(session_id):
    """Play the next step in a simulation replay"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        
        # Check if this is a replay
        if not engine.is_replay:
            return jsonify({'error': 'This session is not a replay'}), 400
            
        # Check if we have more steps to replay
        if engine.replay_position >= len(engine.replay_flattened):
            return jsonify({
                'success': True,
                'done': True,
                'message': 'Replay complete'
            })
            
        # Get the next message
        message = engine.replay_flattened[engine.replay_position]
        engine.replay_position += 1
        
        # Emit the message event
        socketio.emit('simulation_update', {
            'type': 'replay_message',
            'from': message['source_name'],
            'to': message['target_name'],
            'content': message['content'],
            'progress': {
                'current': engine.replay_position,
                'total': len(engine.replay_flattened)
            }
        }, room=session_id)
        
        return jsonify({
            'success': True,
            'done': False,
            'message': message,
            'progress': {
                'current': engine.replay_position,
                'total': len(engine.replay_flattened)
            }
        })
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/replay-reset/<session_id>', methods=['POST'])
def replay_reset(session_id):
    """Reset a simulation replay to the beginning"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        
        # Check if this is a replay
        if not engine.is_replay:
            return jsonify({'error': 'This session is not a replay'}), 400
            
        # Reset position
        engine.replay_position = 0
        
        return jsonify({
            'success': True,
            'message': 'Replay reset to beginning',
            'progress': {
                'current': 0,
                'total': len(engine.replay_flattened)
            }
        })
    else:
        return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/replay-auto/<session_id>', methods=['POST'])
def replay_auto(session_id):
    """Automatically play through a simulation replay"""
    if session_id in active_simulations:
        engine = active_simulations[session_id]
        
        # Check if this is a replay
        if not engine.is_replay:
            return jsonify({'error': 'This session is not a replay'}), 400
            
        data = request.json
        speed = data.get('speed', 1)  # Messages per second
        start_position = data.get('start_position', engine.replay_position)
        end_position = data.get('end_position', len(engine.replay_flattened))
        
        # Validate positions
        if start_position < 0 or start_position >= len(engine.replay_flattened):
            start_position = 0
        if end_position > len(engine.replay_flattened):
            end_position = len(engine.replay_flattened)
            
        # Set position
        engine.replay_position = start_position
        
        # Start a background thread to play through the messages
        threading.Thread(target=auto_replay_thread, 
                         args=(engine, session_id, speed, end_position)).start()
        
        return jsonify({
            'success': True,
            'message': 'Auto replay started',
            'progress': {
                'start': start_position,
                'end': end_position,
                'total': len(engine.replay_flattened)
            }
        })
    else:
        return jsonify({'error': 'Simulation not found'}), 404

def auto_replay_thread(engine, session_id, speed, end_position):
    """Background thread for auto replay"""
    delay = 1.0 / speed  # Seconds per message
    
    try:
        while engine.is_replay and engine.replay_position < end_position:
            # Get the next message
            if engine.replay_position < len(engine.replay_flattened):
                message = engine.replay_flattened[engine.replay_position]
                engine.replay_position += 1
                
                # Emit the message event
                socketio.emit('simulation_update', {
                    'type': 'replay_message',
                    'from': message['source_name'],
                    'to': message['target_name'],
                    'content': message['content'],
                    'progress': {
                        'current': engine.replay_position,
                        'total': len(engine.replay_flattened)
                    }
                }, room=session_id)
                
                # Wait before next message
                time.sleep(delay)
            
        # Signal completion
        socketio.emit('simulation_update', {
            'type': 'replay_complete',
            'content': 'Auto replay complete',
            'progress': {
                'current': engine.replay_position,
                'total': len(engine.replay_flattened)
            }
        }, room=session_id)
    except Exception as e:
        logger.error(f"Error in auto replay thread: {e}")
        socketio.emit('simulation_update', {
            'type': 'error',
            'content': f"Error in replay: {str(e)}"
        }, room=session_id)

@socketio.on('join')
def on_join(data):
    """Join a simulation room"""
    room = data.get('session_id')
    if room:
        socketio.join_room(room)
        emit('joined', {'session_id': room})

@socketio.on('connect')
def on_connect():
    """Client connected"""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    """Client disconnected"""
    logger.info(f"Client disconnected: {request.sid}")

def run_server(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask server"""
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='PAIA Simulation Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting PAIA Simulation Server on http://{args.host}:{args.port}")
    run_server(args.host, args.port, args.debug)
