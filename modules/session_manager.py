from upstash_redis import Redis
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv


class SessionManager:
    """
    Upstash Redis-based Session Manager for conversational recipe assistant

    Manages:
    - Recipe session state (current step, progress)
    - Conversation history
    - Structured response tracking (ingredients/steps spoken)
    - User preferences and context
    """

    def __init__(self, session_ttl=3600):
        """
        Initialize Session Manager with Upstash Redis connection

        Args:
            session_ttl (int): Session time-to-live in seconds (default: 1 hour)
        """
        load_dotenv()

        # Get Upstash Redis configuration from environment
        redis_url = os.getenv('REDIS_URL')

        if not redis_url:
            print("Warning: REDIS_URL not found in environment variables")
            print("Session Manager will operate in degraded mode (no persistence)")
            self.redis_client = None
            self.session_ttl = session_ttl
            self.default_history_limit = 10
            return

        try:
            # Parse Upstash Redis URL
            # Format: rediss://default:token@host:port
            if redis_url.startswith('rediss://'):
                # Extract token and URL
                parts = redis_url.replace('rediss://', '').split('@')
                if len(parts) == 2:
                    auth_part = parts[0]
                    host_part = parts[1]

                    # Extract token (after 'default:')
                    token = auth_part.split(':')[1] if ':' in auth_part else auth_part

                    # Build URL
                    url = f"https://{host_part.split(':')[0]}"

                    # Initialize Upstash Redis client
                    self.redis_client = Redis(url=url, token=token)

                    # Test connection
                    self.redis_client.ping()
                    print(f"Session Manager connected to Upstash Redis at {url}")
                else:
                    raise ValueError("Invalid REDIS_URL format")
            else:
                raise ValueError("REDIS_URL must start with 'rediss://'")

        except Exception as e:
            print(f"Warning: Could not connect to Upstash Redis: {e}")
            print("Session Manager will operate in degraded mode (no persistence)")
            self.redis_client = None

        self.session_ttl = session_ttl
        self.default_history_limit = 10  # Keep last 10 conversation turns

    def create_session(self, session_id: str, recipe_id: str, recipe_title: str,
                      total_steps: int, recipe_data: Optional[Dict] = None) -> Dict:
        """
        Create a new session for a recipe

        Args:
            session_id (str): Unique session identifier (e.g., user_id)
            recipe_id (str): Recipe ID from database
            recipe_title (str): Recipe title
            total_steps (int): Total number of steps in recipe
            recipe_data (dict, optional): Full recipe data including ingredients and steps

        Returns:
            Dictionary containing the created session data
        """
        session_data = {
            "session_id": session_id,
            "recipe_id": recipe_id,
            "recipe_title": recipe_title,
            "step_index": 0,
            "total_steps": total_steps,
            "current_chunk_id": "",
            "last_intent": "",
            "clarification_pending": False,
            "paused": False,
            "current_state": "RECIPE_SELECTED",
            "current_section": "idle",  # idle, greeting, ingredients, steps, closing
            "context_buffer": "",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),

            # Structured response tracking
            "response_structure": recipe_data if recipe_data else {},
            "ingredients_spoken": [],  # List of ingredient indices that have been spoken
            "steps_spoken": [],  # List of step numbers that have been spoken
            "sections_spoken": [],  # List of sections spoken (greeting, ingredients, steps, closing)

            # Conversation history
            "conversation_history": [],  # List of {role, content, timestamp, intent}

            # User preferences and context
            "user_preferences": {},  # Store user choices (e.g., skip ingredient warnings)

            # Timer management
            "timer_active": False,
            "timer_end_time": "",
            "timer_duration": 0,

            # Audio playback tracking
            "current_audio_chunk": 0,
            "total_audio_chunks": 0,
            "playback_position": 0,  # For resuming interrupted playback
        }

        if self.redis_client:
            try:
                # Store session data as JSON string with TTL
                # Upstash Redis setex: setex(key, seconds, value)
                self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_ttl,
                    json.dumps(session_data)
                )
                print(f"Session created: {session_id} for recipe '{recipe_title}'")
            except Exception as e:
                print(f"Error creating session in Redis: {e}")

        return session_data

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve session data

        Args:
            session_id (str): Session identifier

        Returns:
            Dictionary containing session data or None if not found
        """
        if not self.redis_client:
            return None

        try:
            session_json = self.redis_client.get(f"session:{session_id}")
            if session_json:
                session_data = json.loads(session_json)
                # Update last activity timestamp
                session_data["last_activity"] = datetime.utcnow().isoformat()
                self._update_session(session_id, session_data)
                return session_data
            return None
        except Exception as e:
            print(f"Error retrieving session: {e}")
            return None

    def update_session(self, session_id: str, updates: Dict) -> bool:
        """
        Update specific fields in session

        Args:
            session_id (str): Session identifier
            updates (dict): Dictionary of fields to update

        Returns:
            Boolean indicating success
        """
        session_data = self.get_session(session_id)
        if not session_data:
            print(f"Session {session_id} not found")
            return False

        # Update fields
        session_data.update(updates)
        session_data["last_activity"] = datetime.utcnow().isoformat()

        return self._update_session(session_id, session_data)

    def _update_session(self, session_id: str, session_data: Dict) -> bool:
        """
        Internal method to update entire session data

        Args:
            session_id (str): Session identifier
            session_data (dict): Complete session data

        Returns:
            Boolean indicating success
        """
        if not self.redis_client:
            return False

        try:
            self.redis_client.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session_data)
            )
            return True
        except Exception as e:
            print(f"Error updating session: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id (str): Session identifier

        Returns:
            Boolean indicating success
        """
        if not self.redis_client:
            return False

        try:
            self.redis_client.delete(f"session:{session_id}")
            print(f"Session deleted: {session_id}")
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists

        Args:
            session_id (str): Session identifier

        Returns:
            Boolean indicating if session exists
        """
        if not self.redis_client:
            return False

        try:
            return self.redis_client.exists(f"session:{session_id}") > 0
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return False

    def add_conversation_turn(self, session_id: str, role: str, content: str,
                             intent: Optional[str] = None, entities: Optional[Dict] = None) -> bool:
        """
        Add a conversation turn to history

        Args:
            session_id (str): Session identifier
            role (str): 'user' or 'assistant'
            content (str): The message content
            intent (str, optional): Detected intent for user messages
            entities (dict, optional): Extracted entities

        Returns:
            Boolean indicating success
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        conversation_turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "intent": intent,
            "entities": entities if entities else {}
        }

        # Add to conversation history
        history = session_data.get("conversation_history", [])
        history.append(conversation_turn)

        # Limit history size
        if len(history) > self.default_history_limit * 2:  # *2 because user+assistant = 1 turn
            history = history[-self.default_history_limit * 2:]

        session_data["conversation_history"] = history

        # Update last intent for user messages
        if role == "user" and intent:
            session_data["last_intent"] = intent

        return self._update_session(session_id, session_data)

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history

        Args:
            session_id (str): Session identifier
            limit (int, optional): Number of recent turns to retrieve

        Returns:
            List of conversation turns
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return []

        history = session_data.get("conversation_history", [])

        if limit:
            return history[-limit:]
        return history

    def mark_ingredient_spoken(self, session_id: str, ingredient_index: int) -> bool:
        """
        Mark an ingredient as spoken

        Args:
            session_id (str): Session identifier
            ingredient_index (int): Index of ingredient in the list

        Returns:
            Boolean indicating success
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        ingredients_spoken = session_data.get("ingredients_spoken", [])
        if ingredient_index not in ingredients_spoken:
            ingredients_spoken.append(ingredient_index)
            session_data["ingredients_spoken"] = ingredients_spoken
            return self._update_session(session_id, session_data)

        return True

    def mark_step_spoken(self, session_id: str, step_number: int) -> bool:
        """
        Mark a step as spoken

        Args:
            session_id (str): Session identifier
            step_number (int): Step number

        Returns:
            Boolean indicating success
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        steps_spoken = session_data.get("steps_spoken", [])
        if step_number not in steps_spoken:
            steps_spoken.append(step_number)
            session_data["steps_spoken"] = steps_spoken
            session_data["step_index"] = step_number
            return self._update_session(session_id, session_data)

        return True

    def mark_section_spoken(self, session_id: str, section: str) -> bool:
        """
        Mark a section as spoken (greeting, ingredients, steps, closing)

        Args:
            session_id (str): Session identifier
            section (str): Section name

        Returns:
            Boolean indicating success
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        sections_spoken = session_data.get("sections_spoken", [])
        if section not in sections_spoken:
            sections_spoken.append(section)
            session_data["sections_spoken"] = sections_spoken
            session_data["current_section"] = section
            return self._update_session(session_id, session_data)

        return True

    def get_next_unspoken_ingredient(self, session_id: str) -> Optional[Dict]:
        """
        Get the next ingredient that hasn't been spoken

        Args:
            session_id (str): Session identifier

        Returns:
            Dictionary with ingredient data or None
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return None

        response_structure = session_data.get("response_structure", {})
        ingredients = response_structure.get("ingredients", [])
        ingredients_spoken = session_data.get("ingredients_spoken", [])

        for i, ingredient in enumerate(ingredients):
            if i not in ingredients_spoken:
                return {"index": i, "data": ingredient}

        return None

    def get_next_unspoken_step(self, session_id: str) -> Optional[Dict]:
        """
        Get the next step that hasn't been spoken

        Args:
            session_id (str): Session identifier

        Returns:
            Dictionary with step data or None
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return None

        response_structure = session_data.get("response_structure", {})
        steps = response_structure.get("steps", [])
        steps_spoken = session_data.get("steps_spoken", [])

        for step in steps:
            step_num = step.get("step_num", 0)
            if step_num not in steps_spoken:
                return {"step_num": step_num, "data": step}

        return None

    def navigate_to_step(self, session_id: str, step_number: int) -> bool:
        """
        Navigate to a specific step

        Args:
            session_id (str): Session identifier
            step_number (int): Target step number

        Returns:
            Boolean indicating success
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        total_steps = session_data.get("total_steps", 0)
        if 1 <= step_number <= total_steps:
            session_data["step_index"] = step_number
            session_data["current_section"] = "steps"
            return self._update_session(session_id, session_data)

        return False

    def navigate_next(self, session_id: str) -> Optional[int]:
        """
        Navigate to next step

        Args:
            session_id (str): Session identifier

        Returns:
            New step index or None if at end
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return None

        current_step = session_data.get("step_index", 0)
        total_steps = session_data.get("total_steps", 0)

        if current_step < total_steps:
            new_step = current_step + 1
            session_data["step_index"] = new_step
            self._update_session(session_id, session_data)
            return new_step

        return None

    def navigate_prev(self, session_id: str) -> Optional[int]:
        """
        Navigate to previous step

        Args:
            session_id (str): Session identifier

        Returns:
            New step index or None if at beginning
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return None

        current_step = session_data.get("step_index", 0)

        if current_step > 1:
            new_step = current_step - 1
            session_data["step_index"] = new_step
            self._update_session(session_id, session_data)
            return new_step

        return None

    def set_pause(self, session_id: str, paused: bool) -> bool:
        """
        Set pause state

        Args:
            session_id (str): Session identifier
            paused (bool): Pause state

        Returns:
            Boolean indicating success
        """
        return self.update_session(session_id, {
            "paused": paused,
            "current_state": "PAUSED" if paused else "RECIPE_ACTIVE"
        })

    def set_timer(self, session_id: str, duration_seconds: int) -> bool:
        """
        Set a cooking timer

        Args:
            session_id (str): Session identifier
            duration_seconds (int): Timer duration in seconds

        Returns:
            Boolean indicating success
        """
        end_time = datetime.utcnow() + timedelta(seconds=duration_seconds)

        return self.update_session(session_id, {
            "timer_active": True,
            "timer_end_time": end_time.isoformat(),
            "timer_duration": duration_seconds
        })

    def clear_timer(self, session_id: str) -> bool:
        """
        Clear active timer

        Args:
            session_id (str): Session identifier

        Returns:
            Boolean indicating success
        """
        return self.update_session(session_id, {
            "timer_active": False,
            "timer_end_time": "",
            "timer_duration": 0
        })

    def check_timer(self, session_id: str) -> Dict:
        """
        Check timer status

        Args:
            session_id (str): Session identifier

        Returns:
            Dictionary with timer info {active, remaining_seconds, expired}
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return {"active": False, "remaining_seconds": 0, "expired": False}

        timer_active = session_data.get("timer_active", False)
        if not timer_active:
            return {"active": False, "remaining_seconds": 0, "expired": False}

        end_time_str = session_data.get("timer_end_time", "")
        if not end_time_str:
            return {"active": False, "remaining_seconds": 0, "expired": False}

        try:
            end_time = datetime.fromisoformat(end_time_str)
            now = datetime.utcnow()
            remaining = (end_time - now).total_seconds()

            if remaining <= 0:
                # Timer expired
                self.clear_timer(session_id)
                return {"active": False, "remaining_seconds": 0, "expired": True}

            return {"active": True, "remaining_seconds": int(remaining), "expired": False}
        except Exception as e:
            print(f"Error checking timer: {e}")
            return {"active": False, "remaining_seconds": 0, "expired": False}

    def update_playback_position(self, session_id: str, chunk_index: int,
                                 total_chunks: int, position: int = 0) -> bool:
        """
        Update audio playback position for resumption

        Args:
            session_id (str): Session identifier
            chunk_index (int): Current audio chunk being played
            total_chunks (int): Total number of audio chunks
            position (int): Position within current chunk (optional)

        Returns:
            Boolean indicating success
        """
        return self.update_session(session_id, {
            "current_audio_chunk": chunk_index,
            "total_audio_chunks": total_chunks,
            "playback_position": position
        })

    def get_playback_position(self, session_id: str) -> Dict:
        """
        Get current playback position

        Args:
            session_id (str): Session identifier

        Returns:
            Dictionary with playback info
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return {"current_chunk": 0, "total_chunks": 0, "position": 0}

        return {
            "current_chunk": session_data.get("current_audio_chunk", 0),
            "total_chunks": session_data.get("total_audio_chunks", 0),
            "position": session_data.get("playback_position", 0)
        }

    def get_all_session_ids(self) -> List[str]:
        """
        Get all active session IDs

        Returns:
            List of session IDs
        """
        if not self.redis_client:
            return []

        try:
            keys = self.redis_client.keys("session:*")
            return [key.replace("session:", "") for key in keys]
        except Exception as e:
            print(f"Error getting session IDs: {e}")
            return []

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (Redis handles this automatically with TTL)
        This is a utility method for manual cleanup if needed

        Returns:
            Number of sessions cleaned up
        """
        if not self.redis_client:
            return 0

        count = 0
        session_ids = self.get_all_session_ids()

        for session_id in session_ids:
            session_data = self.get_session(session_id)
            if session_data:
                last_activity = session_data.get("last_activity", "")
                if last_activity:
                    try:
                        last_activity_time = datetime.fromisoformat(last_activity)
                        if (datetime.utcnow() - last_activity_time).total_seconds() > self.session_ttl:
                            self.delete_session(session_id)
                            count += 1
                    except Exception:
                        pass

        return count


# Test function
def test_session_manager():
    """Test the session manager functionality"""
    print("=" * 60)
    print("SESSION MANAGER TEST")
    print("=" * 60)

    # Initialize session manager
    sm = SessionManager()

    if not sm.redis_client:
        print("\n⚠️  Redis not available. Please start Redis server:")
        print("   sudo systemctl start redis")
        print("   or: redis-server")
        return

    # Test 1: Create session
    print("\n1. Creating session...")
    recipe_data = {
        "greeting": "Let's make delicious pasta!",
        "ingredients": [
            {"text": "500g pasta"},
            {"text": "2 cups tomato sauce"},
            {"text": "1 onion, chopped"}
        ],
        "steps": [
            {"step_num": 1, "text": "Boil water in a large pot"},
            {"step_num": 2, "text": "Add pasta and cook for 10 minutes"},
            {"step_num": 3, "text": "Drain and serve with sauce"}
        ],
        "closing": "Enjoy your pasta!"
    }

    session_data = sm.create_session(
        session_id="test_user_123",
        recipe_id="recipe_456",
        recipe_title="Simple Pasta",
        total_steps=3,
        recipe_data=recipe_data
    )
    print(f"✓ Session created: {session_data['session_id']}")

    # Test 2: Retrieve session
    print("\n2. Retrieving session...")
    retrieved = sm.get_session("test_user_123")
    print(f"✓ Session retrieved: {retrieved['recipe_title']}")

    # Test 3: Add conversation
    print("\n3. Adding conversation history...")
    sm.add_conversation_turn("test_user_123", "user", "How do I make pasta?", "search_recipe")
    sm.add_conversation_turn("test_user_123", "assistant", "Let's make Simple Pasta!")
    history = sm.get_conversation_history("test_user_123")
    print(f"✓ Conversation history: {len(history)} turns")

    # Test 4: Mark progress
    print("\n4. Tracking progress...")
    sm.mark_section_spoken("test_user_123", "greeting")
    sm.mark_ingredient_spoken("test_user_123", 0)
    sm.mark_ingredient_spoken("test_user_123", 1)
    sm.mark_step_spoken("test_user_123", 1)
    print("✓ Progress tracked")

    # Test 5: Navigation
    print("\n5. Testing navigation...")
    next_step = sm.navigate_next("test_user_123")
    print(f"✓ Navigate next: step {next_step}")
    prev_step = sm.navigate_prev("test_user_123")
    print(f"✓ Navigate prev: step {prev_step}")

    # Test 6: Get unspoken content
    print("\n6. Getting unspoken content...")
    next_ingredient = sm.get_next_unspoken_ingredient("test_user_123")
    print(f"✓ Next unspoken ingredient: {next_ingredient}")
    next_step_unspoken = sm.get_next_unspoken_step("test_user_123")
    print(f"✓ Next unspoken step: {next_step_unspoken}")

    # Test 7: Pause/Resume
    print("\n7. Testing pause/resume...")
    sm.set_pause("test_user_123", True)
    paused_session = sm.get_session("test_user_123")
    print(f"✓ Paused: {paused_session['paused']}")
    sm.set_pause("test_user_123", False)
    print("✓ Resumed")

    # Test 8: Timer
    print("\n8. Testing timer...")
    sm.set_timer("test_user_123", 5)  # 5 second timer
    timer_status = sm.check_timer("test_user_123")
    print(f"✓ Timer set: {timer_status}")

    # Test 9: Playback position
    print("\n9. Testing playback position...")
    sm.update_playback_position("test_user_123", 2, 5, 150)
    playback = sm.get_playback_position("test_user_123")
    print(f"✓ Playback position: chunk {playback['current_chunk']}/{playback['total_chunks']}")

    # Test 10: Cleanup
    print("\n10. Cleanup...")
    sm.delete_session("test_user_123")
    exists = sm.session_exists("test_user_123")
    print(f"✓ Session deleted: exists={exists}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_session_manager()

