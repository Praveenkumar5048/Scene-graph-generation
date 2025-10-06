import requests
import json
import base64
from typing import Dict, List, Tuple, Any
import os
from dotenv import load_dotenv
from config import VG_OFFICIAL_CLASSES, VG_OFFICIAL_PREDICATES

# Load environment variables
load_dotenv()

class GeminiSceneGraphGenerator:
    def __init__(self):
        """
        Initialize Scene Graph Generator
        """
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        # Load VG classes and predicates for structured output
        self.object_classes = VG_OFFICIAL_CLASSES[1:]  # Remove background
        self.predicates = VG_OFFICIAL_PREDICATES
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for API request
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_scene_graph_prompt(self) -> str:
        """
        Create a detailed prompt for scene graph generation
        
        Returns:
            Formatted prompt string
        """
        # Create formatted lists for the prompt
        classes_str = '", "'.join(self.object_classes)
        predicates_str = '", "'.join(self.predicates)
        
        prompt = f"""
        Analyze this image and generate a scene graph that describes the visual relationships between objects.
        
        STRICT REQUIREMENTS:
        1. You MUST ONLY use object classes from this EXACT list: ["{classes_str}"]
        2. You MUST ONLY use relationship predicates from this EXACT list: ["{predicates_str}"]
        3. DO NOT use any other object names or relationship words outside these lists
        4. If you see an object that doesn't match the list exactly, choose the closest match from the provided list
        5. If you see a relationship that doesn't match the list exactly, choose the closest match from the provided predicates
        
        Please provide your response in the following JSON format:
        {{
            "objects": [
                {{
                    "id": 0,
                    "class": "object_name_from_list_only"
                }}
            ],
            "relationships": [
                {{
                    "subject_id": 0,
                    "predicate": "relationship_from_list_only", 
                    "object_id": 1
                }}
            ]
        }}
        
        Guidelines:
        1. Identify all visible objects in the image
        2. Map each object to the closest match from the provided object class list
        3. Map each relationship to the closest match from the provided predicate list
        4. Focus on clear, visible relationships between objects
        5. Include both spatial relationships and semantic relationships
        6. Only include objects and relationships that are clearly visible in the image
        
        CRITICAL: Every "class" value must be exactly one of these: {self.object_classes}
        CRITICAL: Every "predicate" value must be exactly one of these: {self.predicates}
        
        Make sure the response is valid JSON format.
        """
        return prompt
    
    def validate_and_fix_scene_graph(self, scene_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix scene graph to ensure all classes and predicates are from allowed lists
        
        Args:
            scene_graph: Raw scene graph from Gemini
            
        Returns:
            Validated and corrected scene graph
        """
        fixed_objects = []
        fixed_relationships = []
        
        # Fix objects
        for obj in scene_graph.get("objects", []):
            obj_class = obj.get("class", "").lower()
            
            # Find exact match first
            if obj_class in self.object_classes:
                fixed_objects.append(obj)
            else:
                # Find closest match
                closest_match = self.find_closest_class(obj_class)
                if closest_match:
                    obj["class"] = closest_match
                    fixed_objects.append(obj)
                    print(f"Warning: Changed '{obj_class}' to '{closest_match}'")
        
        # Fix relationships
        for rel in scene_graph.get("relationships", []):
            predicate = rel.get("predicate", "").lower()
            
            # Find exact match first
            if predicate in self.predicates:
                fixed_relationships.append(rel)
            else:
                # Find closest match
                closest_match = self.find_closest_predicate(predicate)
                if closest_match:
                    rel["predicate"] = closest_match
                    fixed_relationships.append(rel)
                    print(f"Warning: Changed '{predicate}' to '{closest_match}'")
        
        return {
            "objects": fixed_objects,
            "relationships": fixed_relationships
        }
    
    def find_closest_class(self, target_class: str) -> str:
        """
        Find the closest matching class from the allowed list
        """
        target_class = target_class.lower()
        
        # Direct substring matches
        for cls in self.object_classes:
            if target_class in cls.lower() or cls.lower() in target_class:
                return cls
        
        # Common mappings
        class_mappings = {
            'frisbee': 'plate',  # frisbee -> plate
            'disc': 'plate',
            'flying disc': 'plate',
            'person': 'person',
            'human': 'person',
            'people': 'people',
            'canine': 'dog',
            'puppy': 'dog',
            'hound': 'dog'
        }
        
        if target_class in class_mappings:
            return class_mappings[target_class]
        
        # Default fallback
        return 'animal' if target_class else None
    
    def find_closest_predicate(self, target_predicate: str) -> str:
        """
        Find the closest matching predicate from the allowed list
        """
        target_predicate = target_predicate.lower()
        
        # Direct substring matches
        for pred in self.predicates:
            if target_predicate in pred.lower() or pred.lower() in target_predicate:
                return pred
        
        # Common mappings
        predicate_mappings = {
            'chasing': 'near',  # chasing -> near
            'running after': 'near',
            'following': 'behind',
            'catching': 'holding',
            'grabbing': 'holding',
            'playing with': 'playing',
            'interacting with': 'near',
            'next to': 'near',
            'beside': 'near',
            'close to': 'near'
        }
        
        if target_predicate in predicate_mappings:
            return predicate_mappings[target_predicate]
        
        # Default fallback
        return 'near'
    
    def send_request(self, image_path: str) -> Dict[str, Any]:
        """
        Send request to Gemini API with image and scene graph prompt
        
        Args:
            image_path: Path to the image file
            
        Returns:
            API response as dictionary
        """
        # Encode image
        encoded_image = self.encode_image(image_path)
        
        # Get image format
        image_format = image_path.split('.')[-1].lower()
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        # Create request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": self.create_scene_graph_prompt()
                        },
                        {
                            "inline_data": {
                                "mime_type": f"image/{image_format}",
                                "data": encoded_image
                            }
                        }
                    ]
                }
            ]
        }
        
        # Send request
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def parse_scene_graph_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Gemini response and extract scene graph data
        
        Args:
            response: Raw API response
            
        Returns:
            Parsed scene graph data
        """
        try:
            # Extract text content from response
            content = response['candidates'][0]['content']['parts'][0]['text']
            
            # Find JSON content (remove any markdown formatting)
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            scene_graph = json.loads(content)
            
            # Validate and fix the scene graph
            scene_graph = self.validate_and_fix_scene_graph(scene_graph)
            
            return scene_graph
            
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return {"objects": [], "relationships": []}
    
    def format_output_like_sgg_model(self, scene_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format output to match traditional SGG model output format
        
        Args:
            scene_graph: Parsed scene graph from Gemini
            
        Returns:
            Formatted output similar to SGG models
        """
        objects = scene_graph.get("objects", [])
        relationships = scene_graph.get("relationships", [])
        
        # Format similar to your SGG model output
        formatted_output = {
            "num_objects": len(objects),
            "object_classes": [obj.get("class", "unknown") for obj in objects],
            "object_boxes": [[0, 0, 100, 100] for _ in objects],  # Default placeholder boxes
            "object_scores": [1.0 for _ in objects],  # Default confidence scores
            "num_relations": len(relationships),
            "relation_triplets": [],
            "relation_scores": [1.0 for _ in relationships],  # Default confidence scores
            "predicate_classes": [rel.get("predicate", "unknown") for rel in relationships]
        }
        
        # Create relation triplets [subject_id, predicate_id, object_id]
        for rel in relationships:
            subject_id = rel.get("subject_id", 0)
            object_id = rel.get("object_id", 0)
            predicate = rel.get("predicate", "unknown")
            
            # Try to map predicate to VG predicate ID
            predicate_id = 0
            if predicate in self.predicates:
                predicate_id = self.predicates.index(predicate)
            
            formatted_output["relation_triplets"].append([subject_id, predicate_id, object_id])
        
        return formatted_output
    
    def generate_scene_graph(self, image_path: str) -> Dict[str, Any]:
        """
        Main method to generate scene graph from image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Formatted scene graph output
        """
        print(f"Generating scene graph for: {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Send request to API
        print("Analyzing image...")
        response = self.send_request(image_path)
        
        # Parse response
        print("Parsing scene graph response...")
        scene_graph = self.parse_scene_graph_response(response)
        
        # Format output
        print("Formatting output...")
        formatted_output = self.format_output_like_sgg_model(scene_graph)
        
        return formatted_output
    
    def print_scene_graph(self, scene_graph: Dict[str, Any]):
        """
        Print scene graph in a readable format
        
        Args:
            scene_graph: Formatted scene graph output
        """
        print("\n=== SCENE GRAPH RESULTS ===")
        print(f"Detected {scene_graph['num_objects']} objects:")
        
        for i, obj_class in enumerate(scene_graph['object_classes']):
            print(f"  Object {i}: {obj_class}")
        
        print(f"\nDetected {scene_graph['num_relations']} relationships:")
        
        for i, (triplet, predicate) in enumerate(zip(
            scene_graph['relation_triplets'],
            scene_graph['predicate_classes']
        )):
            subj_id, pred_id, obj_id = triplet
            subj_name = scene_graph['object_classes'][subj_id] if subj_id < len(scene_graph['object_classes']) else f"obj_{subj_id}"
            obj_name = scene_graph['object_classes'][obj_id] if obj_id < len(scene_graph['object_classes']) else f"obj_{obj_id}"
            
            print(f"  Relation {i}: {subj_name} -> {predicate} -> {obj_name}")


def main():
    """
    Example usage of Scene Graph Generator
    """
    # Initialize generator
    generator = GeminiSceneGraphGenerator()
    
    # Process only one test image
    image_path = "E:\\VS Code\\CV Project\\test_image1.png"
    
    if os.path.exists(image_path):
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {os.path.basename(image_path)}")
            print(f"{'='*50}")
            
            # Generate scene graph
            result = generator.generate_scene_graph(image_path)
            
            # Print results
            generator.print_scene_graph(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
