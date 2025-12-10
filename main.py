import cv2
import mediapipe as mp
import math
import tkinter as tk
from PIL import Image, ImageTk
import os

MONKEY_THINK_PATH = "monkeyPictures/monkeyThink.png"  # Use forward slash for cross-platform
MONKEY_IDEA_PATH = "monkeyPictures/monkeyIdea.png"
MONKEY_SHOCKED_PATH = "monkeyPictures/monkeyShocked.png"
MONKEY_STARE_PATH = "monkeyPictures/monkeyStare.PNG"
MONKEY_MIDDLE_FINGER_PATH = "monkeyPictures/monkeyMiddleFinger.PNG"
MONKEY_THUMBS_UP_PATH = "monkeyPictures/monkeyThumbsUp.png"
MONKEY_TONGUE_PATH = "monkeyPictures/monkeyTongue.png"


class PoseMonkeyApp:
    # Debug mode: Set to True to show landmarks, circles, and rectangles on the video feed
    DEBUG = False  # Change to False to hide all debugging visuals
    
    # Detection thresholds (adjust these if detection is too sensitive/not sensitive enough)
    FINGER_MOUTH_DISTANCE_THRESHOLD = 0.08  # Normalized distance (0-1), smaller = more strict
    HEAD_TILT_UP_THRESHOLD = 0.05  # How much higher nose should be than shoulders (normalized)
    FINGER_POINT_UP_THRESHOLD = 0.15  # Distance threshold for finger pointing up detection
    HAND_CHEST_DISTANCE_THRESHOLD = 0.15  # Distance for hands on chest area (broader than heart)
    MOUTH_OPEN_THRESHOLD = 0.02  # Distance between upper and lower lip for open mouth
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Monkey")

        # --- Video label ---
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # --- Status text ---
        self.status_var = tk.StringVar(value="Do a pose: thinking, idea, shocked, stare, middle finger, thumbs up, or tongue! 🐒")
        tk.Label(root, textvariable=self.status_var).pack(pady=5)

        # --- Load monkey images ---
        # Try different path formats for thinking monkey
        think_path = MONKEY_THINK_PATH if os.path.exists(MONKEY_THINK_PATH) else MONKEY_THINK_PATH.replace('/', '\\')
        self.monkey_think_img = cv2.imread(think_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_think_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_THINK_PATH}")
        print(f"Thinking monkey image loaded: {self.monkey_think_img.shape}")
        
        # Load idea monkey image
        idea_path = MONKEY_IDEA_PATH if os.path.exists(MONKEY_IDEA_PATH) else MONKEY_IDEA_PATH.replace('/', '\\')
        self.monkey_idea_img = cv2.imread(idea_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_idea_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_IDEA_PATH}")
        print(f"Idea monkey image loaded: {self.monkey_idea_img.shape}")
        
        # Load shocked monkey image
        shocked_path = MONKEY_SHOCKED_PATH if os.path.exists(MONKEY_SHOCKED_PATH) else MONKEY_SHOCKED_PATH.replace('/', '\\')
        self.monkey_shocked_img = cv2.imread(shocked_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_shocked_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_SHOCKED_PATH}")
        print(f"Shocked monkey image loaded: {self.monkey_shocked_img.shape}")
        
        # Load stare monkey image
        stare_path = MONKEY_STARE_PATH if os.path.exists(MONKEY_STARE_PATH) else MONKEY_STARE_PATH.replace('/', '\\')
        self.monkey_stare_img = cv2.imread(stare_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_stare_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_STARE_PATH}")
        print(f"Stare monkey image loaded: {self.monkey_stare_img.shape}")
        
        # Load middle finger monkey image
        middle_finger_path = MONKEY_MIDDLE_FINGER_PATH if os.path.exists(MONKEY_MIDDLE_FINGER_PATH) else MONKEY_MIDDLE_FINGER_PATH.replace('/', '\\')
        self.monkey_middle_finger_img = cv2.imread(middle_finger_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_middle_finger_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_MIDDLE_FINGER_PATH}")
        print(f"Middle finger monkey image loaded: {self.monkey_middle_finger_img.shape}")
        
        # Load thumbs up monkey image
        thumbs_up_path = MONKEY_THUMBS_UP_PATH if os.path.exists(MONKEY_THUMBS_UP_PATH) else MONKEY_THUMBS_UP_PATH.replace('/', '\\')
        self.monkey_thumbs_up_img = cv2.imread(thumbs_up_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_thumbs_up_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_THUMBS_UP_PATH}")
        print(f"Thumbs up monkey image loaded: {self.monkey_thumbs_up_img.shape}")
        
        # Load tongue monkey image
        tongue_path = MONKEY_TONGUE_PATH if os.path.exists(MONKEY_TONGUE_PATH) else MONKEY_TONGUE_PATH.replace('/', '\\')
        self.monkey_tongue_img = cv2.imread(tongue_path, cv2.IMREAD_UNCHANGED)
        if self.monkey_tongue_img is None:
            raise FileNotFoundError(f"Could not load {MONKEY_TONGUE_PATH}")
        print(f"Tongue monkey image loaded: {self.monkey_tongue_img.shape}")

        # --- MediaPipe Hands for finger detection ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # --- MediaPipe Pose for head orientation ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # --- MediaPipe Face Mesh for mouth detection ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Webcam ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # keep reference to PhotoImage
        self.photo = None

        # handle closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # start loop
        self.update_frame()

    @staticmethod
    def distance_normalized(p1, p2):
        """Calculate distance between two normalized coordinate points (0-1 range)."""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def is_looking_up(self, pose_landmarks):
        """Check if head is tilted up by comparing nose position to shoulders."""
        if not pose_landmarks:
            return False
        
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Check visibility
        if nose.visibility < 0.5 or left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return False
        
        # Average shoulder Y position
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # If nose is significantly higher (smaller Y value) than shoulders, head is up
        nose_is_higher = nose.y < (avg_shoulder_y - self.HEAD_TILT_UP_THRESHOLD)
        
        return nose_is_higher

    def is_finger_at_mouth(self, hand_landmarks, pose_landmarks, w, h):
        """Check if index finger tip is near the mouth area."""
        if not hand_landmarks or not pose_landmarks:
            return False
        
        # Get index finger tip (landmark 8)
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Get mouth area from pose (use nose as approximation, or mouth if available)
        pose_landmark_list = pose_landmarks.landmark
        nose = pose_landmark_list[self.mp_pose.PoseLandmark.NOSE]
        
        # Mouth is typically below nose, so we'll use nose + offset
        # For "finger at mouth" pose, finger should be near the mouth/nose area
        mouth_ref_x = nose.x
        mouth_ref_y = nose.y + 0.03  # Mouth is slightly below nose
        
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        mouth_ref = Point(mouth_ref_x, mouth_ref_y)
        
        # Calculate distance from index finger tip to mouth
        distance = self.distance_normalized(mouth_ref, index_tip)
        
        # Check if finger is close enough to mouth
        finger_near_mouth = distance < self.FINGER_MOUTH_DISTANCE_THRESHOLD
        
        # Also check that finger is roughly at face level (not too high or low)
        finger_at_face_level = abs(index_tip.y - mouth_ref.y) < 0.1
        
        return finger_near_mouth and finger_at_face_level

    def is_finger_pointing_up(self, hand_landmarks, pose_landmarks):
        """Check if index finger is pointing upward (idea pose)."""
        if not hand_landmarks or not pose_landmarks:
            return False
        
        # Get finger landmarks
        landmarks = hand_landmarks.landmark
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]  # Middle joint
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Base joint
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Check if index finger is extended (tip is higher than middle joint, which is higher than base)
        index_extended = (index_tip.y < index_pip.y < index_mcp.y)
        
        # Check if other fingers are curled (their tips are below their PIP joints)
        middle_curled = middle_tip.y > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_curled = ring_tip.y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_curled = pinky_tip.y > landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y
        
        # Check if thumb is not extended (thumb is curled or neutral)
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_curled = thumb_tip.x > thumb_ip.x or thumb_tip.y > thumb_ip.y
        
        # Index finger should be pointing up (tip is significantly above base)
        finger_pointing_up = index_tip.y < (index_mcp.y - 0.1)
        
        # Get head position to ensure finger is above head/shoulders
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        if nose.visibility > 0.5:
            finger_above_head = index_tip.y < nose.y - 0.05
        else:
            finger_above_head = True  # If we can't see head, assume it's valid
        
        return (index_extended and middle_curled and ring_curled and pinky_curled and 
                finger_pointing_up and finger_above_head)

    def is_hands_on_chest(self, hand_results, pose_landmarks):
        """Check if hands are on the chest area."""
        if not pose_landmarks or not hand_results.multi_hand_landmarks:
            return False
        
        # Need at least one hand detected
        if len(hand_results.multi_hand_landmarks) < 1:
            return False
        
        pose_landmark_list = pose_landmarks.landmark
        
        # Get chest area reference (between shoulders, covering chest width)
        left_shoulder = pose_landmark_list[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmark_list[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return False
        
        # Chest area spans from shoulder to shoulder, and extends lower on the torso
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        # Start chest area lower (below shoulders) and extend much further down
        chest_top_y = shoulder_center_y + 0.10  # Start 10% below shoulders
        chest_bottom_y = shoulder_center_y + 0.35  # Extend to 35% below shoulders (larger area)
        
        # Check each detected hand
        hands_on_chest = 0
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Use wrist as reference point
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Check if hand is within chest horizontal bounds (wider margin for bigger area)
            chest_width = abs(left_shoulder.x - right_shoulder.x)
            # Allow larger margin beyond shoulders for hands on chest (bigger detection zone)
            horizontal_margin = chest_width * 0.5  # Increased from 0.3 to 0.5 for wider detection
            within_chest_width = (
                wrist.x >= min(left_shoulder.x, right_shoulder.x) - horizontal_margin and
                wrist.x <= max(left_shoulder.x, right_shoulder.x) + horizontal_margin
            )
            
            # Check if hand is at chest vertical level (lower and bigger range)
            at_chest_level = (
                wrist.y >= chest_top_y and  # Start from lower point
                wrist.y <= chest_bottom_y  # Down to lower chest/torso area
            )
            
            if within_chest_width and at_chest_level:
                hands_on_chest += 1
        
        # At least one hand should be on the chest
        return hands_on_chest >= 1

    def is_mouth_open(self, face_landmarks):
        """Check if mouth is open using face mesh landmarks."""
        if not face_landmarks:
            return False
        
        # Face mesh landmark indices for mouth
        # Upper lip inner: 13, 82, 81, 80, 78
        # Lower lip inner: 14, 88, 95, 96, 85
        # Using key points: upper inner (13) and lower inner (14)
        upper_lip = face_landmarks.landmark[13]  # Upper lip inner
        lower_lip = face_landmarks.landmark[14]  # Lower lip inner
        
        # Calculate vertical distance between upper and lower lip
        mouth_open_distance = abs(upper_lip.y - lower_lip.y)
        
        # Mouth is open if the distance exceeds threshold
        return mouth_open_distance > self.MOUTH_OPEN_THRESHOLD

    def is_tongue_out(self, face_landmarks):
        """Check if tongue is sticking out using face mesh landmarks."""
        if not face_landmarks:
            return False
        
        # Face mesh landmarks for mouth and tongue (with refine_landmarks=True)
        # Lower lip inner: 14
        # Upper lip inner: 13
        # Tongue landmarks (when refine_landmarks=True): 12, 15, 16, 17, 18
        # Landmark 12 is typically the tongue tip when visible
        
        lower_lip = face_landmarks.landmark[14]  # Lower lip inner
        upper_lip = face_landmarks.landmark[13]  # Upper lip inner
        
        # Check if mouth is open enough (required for tongue to be out)
        mouth_open_distance = abs(upper_lip.y - lower_lip.y)
        # Require mouth to be open (but not as strict - 1.3x threshold)
        if mouth_open_distance < self.MOUTH_OPEN_THRESHOLD * 1.3:
            return False
        
        # Check tongue landmarks - when tongue is out, these should be visible and below lower lip
        # Use multiple tongue landmarks for more reliable detection
        tongue_landmarks = [12, 15, 16, 17, 18]  # Tongue outline landmarks
        
        tongue_visible_below_lip = 0
        tongue_tip_visible = False
        
        for lm_idx in tongue_landmarks:
            if lm_idx < len(face_landmarks.landmark):
                tongue_point = face_landmarks.landmark[lm_idx]
                # Check if tongue point is below lower lip (more lenient threshold: 0.01)
                if tongue_point.y > lower_lip.y + 0.01:  # Below lower lip
                    tongue_visible_below_lip += 1
                    # Check if tongue tip (landmark 12) is visible
                    if lm_idx == 12:
                        tongue_tip_visible = True
        
        # Require at least 2 tongue landmarks to be below lip (more lenient)
        # AND tongue tip must be visible
        return tongue_visible_below_lip >= 2 and tongue_tip_visible

    def is_thumbs_up(self, hand_results, pose_landmarks):
        """Check if user is doing thumbs up at chest level."""
        if not hand_results.multi_hand_landmarks or not pose_landmarks:
            return False
        
        pose_landmark_list = pose_landmarks.landmark
        
        # Get chest area reference
        left_shoulder = pose_landmark_list[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmark_list[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return False
        
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        chest_top_y = shoulder_center_y + 0.10
        chest_bottom_y = shoulder_center_y + 0.35
        chest_width = abs(left_shoulder.x - right_shoulder.x)
        horizontal_margin = chest_width * 0.5
        
        # Check each detected hand for thumbs up
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get thumb landmarks
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
            thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
            wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
            
            # Get finger landmarks
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
            
            # Thumb should be extended (thumb tip is above thumb IP, which is above thumb MCP)
            thumb_extended = (thumb_tip.y < thumb_ip.y < thumb_mcp.y)
            
            # Other fingers should be curled (tips are below their PIP joints)
            index_curled = index_tip.y > index_pip.y
            middle_curled = middle_tip.y > middle_pip.y
            ring_curled = ring_tip.y > ring_pip.y
            pinky_curled = pinky_tip.y > pinky_pip.y
            
            # Check if hand is at chest level
            within_chest_width = (
                wrist.x >= min(left_shoulder.x, right_shoulder.x) - horizontal_margin and
                wrist.x <= max(left_shoulder.x, right_shoulder.x) + horizontal_margin
            )
            
            at_chest_level = (
                wrist.y >= chest_top_y and
                wrist.y <= chest_bottom_y
            )
            
            # Thumbs up detected if thumb extended, other fingers curled, and at chest level
            if (thumb_extended and index_curled and middle_curled and ring_curled and pinky_curled and
                within_chest_width and at_chest_level):
                return True
        
        return False

    def is_middle_finger_pose(self, hand_results):
        """Check if both middle fingers are extended (other fingers curled)."""
        if not hand_results.multi_hand_landmarks:
            return False
        
        # Need exactly 2 hands for this pose
        if len(hand_results.multi_hand_landmarks) != 2:
            return False
        
        both_middle_fingers_extended = True
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get finger landmarks
            middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            
            ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
            
            pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
            
            # Check if middle finger is extended (tip is above PIP, which is above MCP)
            middle_extended = (middle_tip.y < middle_pip.y < middle_mcp.y)
            
            # Check if other fingers are curled (tips are below their PIP joints)
            index_curled = index_tip.y > index_pip.y
            ring_curled = ring_tip.y > ring_pip.y
            pinky_curled = pinky_tip.y > pinky_pip.y
            
            # Check if thumb is curled (thumb tip is below thumb IP)
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
            thumb_curled = thumb_tip.y > thumb_ip.y
            
            # All conditions must be met for this hand
            if not (middle_extended and index_curled and ring_curled and pinky_curled):
                both_middle_fingers_extended = False
                break
        
        return both_middle_fingers_extended

    def is_stare_pose(self, pose_results, hand_results, face_results):
        """Check if user is just staring (face visible, no hand gestures, neutral expression)."""
        # Face and pose must be detected
        if not face_results.multi_face_landmarks or not pose_results or not pose_results.pose_landmarks:
            return False
        
        # Face should be roughly centered and facing forward (nose visible)
        nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        if nose.visibility < 0.5:
            return False
        
        # Mouth should be closed (neutral expression)
        face_landmarks = face_results.multi_face_landmarks[0]
        if self.is_mouth_open(face_landmarks):
            return False
        
        # If hands are detected, check that they are NOT near the face (reject active gestures)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Reject if hand/finger is near face area (indicates active gesture like finger at mouth)
                # Check if wrist or index finger is close to nose/face area
                face_area_x = nose.x
                face_area_y = nose.y + 0.05  # Slightly below nose for mouth area
                
                class Point:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                
                face_ref = Point(face_area_x, face_area_y)
                wrist_dist = self.distance_normalized(face_ref, wrist)
                finger_dist = self.distance_normalized(face_ref, index_tip)
                
                # If hand or finger is too close to face, this is NOT a stare pose
                if wrist_dist < 0.15 or finger_dist < 0.15:
                    return False
        
        # Check if shoulders are visible (may not be when user is further away)
        left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulders_visible = (left_shoulder.visibility >= 0.5 and right_shoulder.visibility >= 0.5)
        
        # If shoulders are visible, check head alignment (but make it more lenient)
        if shoulders_visible:
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            # More lenient head alignment check (0.35 instead of 0.25)
            head_straight = abs(nose.y - avg_shoulder_y) < 0.35
            
            if not head_straight:
                return False
        else:
            # If shoulders not visible (user further away), just check that head isn't tilted extremely
            # Use a more lenient check - just ensure nose is roughly in middle of frame vertically
            # If nose is very high or very low, user is probably not staring straight
            head_straight = 0.1 < nose.y < 0.6  # Nose should be roughly in middle third of frame
            if not head_straight:
                return False
        
        # If no hands detected, this is likely a stare pose (very common when further away)
        if not hand_results.multi_hand_landmarks:
            return True
        
        # If hands are detected, they should be in neutral position (not gesturing)
        # When user is further away, be more lenient about hand positions
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            if shoulders_visible:
                # If shoulders visible, check hand is below shoulders
                avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                # More lenient - allow hands to be closer to shoulder level
                if wrist.y < avg_shoulder_y - 0.05:  # Hand is significantly above shoulders (gesturing)
                    return False
            else:
                # If shoulders not visible, just check hands aren't too high in frame
                # Hands should be in lower portion of frame (not raised up gesturing)
                if wrist.y < 0.3:  # Hand is in upper third of frame (likely gesturing)
                    return False
        
        # All conditions met - this is a stare pose
        return True

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        # Mirror for natural viewing
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands, pose, and face
        hand_results = self.hands.process(rgb)
        pose_results = self.pose.process(rgb)
        face_results = self.face_mesh.process(rgb)

        pose_detected = False
        finger_at_mouth = False
        looking_up = False
        finger_pointing_up = False
        hands_on_chest = False
        mouth_open = False
        middle_finger_pose = False
        stare_pose = False
        thumbs_up = False
        tongue_out = False
        current_pose = None  # "think", "idea", "shocked", "stare", "middle_finger", "thumbs_up", or "tongue"

        # Draw hand landmarks and check for poses
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if self.DEBUG:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if pose_results.pose_landmarks:
                    # Check for thinking pose (finger at mouth)
                    if self.is_finger_at_mouth(hand_landmarks, pose_results.pose_landmarks, w, h):
                        finger_at_mouth = True
                        if self.DEBUG:
                            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            tip_x, tip_y = int(index_tip.x * w), int(index_tip.y * h)
                            cv2.circle(frame, (tip_x, tip_y), 15, (0, 255, 255), 3)
                    
                    # Check for idea pose (finger pointing up)
                    if self.is_finger_pointing_up(hand_landmarks, pose_results.pose_landmarks):
                        finger_pointing_up = True
                        if self.DEBUG:
                            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            tip_x, tip_y = int(index_tip.x * w), int(index_tip.y * h)
                            cv2.circle(frame, (tip_x, tip_y), 15, (255, 0, 255), 3)  # Magenta for idea pose
            
            # Check for middle finger pose (both hands)
            middle_finger_pose = self.is_middle_finger_pose(hand_results)
            if middle_finger_pose and self.DEBUG:
                # Highlight middle fingers
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    tip_x, tip_y = int(middle_tip.x * w), int(middle_tip.y * h)
                    cv2.circle(frame, (tip_x, tip_y), 15, (255, 165, 0), 3)  # Orange for middle finger

        # Draw pose landmarks and check head orientation
        if pose_results.pose_landmarks:
            pose_detected = True
            if self.DEBUG:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Check if looking up (for thinking pose)
            looking_up = self.is_looking_up(pose_results.pose_landmarks)
            
            # Visual debugging: draw nose
            if self.DEBUG:
                nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                cv2.circle(frame, (nose_x, nose_y), 8, (0, 255, 0), 2)  # Green for nose
            
            # Check for hands on chest
            if hand_results.multi_hand_landmarks:
                hands_on_chest = self.is_hands_on_chest(hand_results, pose_results.pose_landmarks)
                if hands_on_chest and self.DEBUG:
                    # Draw chest area reference (bigger, lower area)
                    left_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                    chest_top_y = shoulder_center_y + 0.10
                    chest_bottom_y = shoulder_center_y + 0.35
                    chest_width = abs(left_shoulder.x - right_shoulder.x)
                    horizontal_margin = chest_width * 0.5
                    
                    cv2.rectangle(frame, 
                                 (int((min(left_shoulder.x, right_shoulder.x) - horizontal_margin) * w), 
                                  int(chest_top_y * h)),
                                 (int((max(left_shoulder.x, right_shoulder.x) + horizontal_margin) * w), 
                                  int(chest_bottom_y * h)),
                                 (0, 0, 255), 2)  # Red rectangle for chest area
                
                # Check for thumbs up at chest level
                thumbs_up = self.is_thumbs_up(hand_results, pose_results.pose_landmarks)
                if thumbs_up and self.DEBUG:
                    # Highlight thumbs
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        tip_x, tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        cv2.circle(frame, (tip_x, tip_y), 15, (0, 255, 0), 3)  # Green for thumbs up
        
        # Check for open mouth using face mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh (optional, can be removed for cleaner view)
                # self.mp_drawing.draw_landmarks(
                #     frame,
                #     face_landmarks,
                #     self.mp_face_mesh.FACEMESH_CONTOURS,
                #     None,
                #     self.mp_drawing_styles.get_default_face_mesh_contours_style()
                # )
                
                mouth_open = self.is_mouth_open(face_landmarks)
                if mouth_open and self.DEBUG:
                    # Draw mouth indicator
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]
                    mouth_x = int((upper_lip.x + lower_lip.x) / 2 * w)
                    mouth_y = int((upper_lip.y + lower_lip.y) / 2 * h)
                    cv2.circle(frame, (mouth_x, mouth_y), 10, (0, 255, 255), 2)  # Cyan for open mouth
                
                # Check for tongue out
                tongue_out = self.is_tongue_out(face_landmarks)
                
                # Visual debugging for tongue detection
                if self.DEBUG:
                    lower_lip = face_landmarks.landmark[14]
                    lower_lip_x = int(lower_lip.x * w)
                    lower_lip_y = int(lower_lip.y * h)
                    # Draw lower lip reference point
                    cv2.circle(frame, (lower_lip_x, lower_lip_y), 8, (0, 255, 0), 2)  # Green for lower lip
                    
                    # Draw all tongue landmarks for debugging
                    tongue_landmarks = [12, 15, 16, 17, 18]
                    for lm_idx in tongue_landmarks:
                        if lm_idx < len(face_landmarks.landmark):
                            tongue_point = face_landmarks.landmark[lm_idx]
                            tongue_x = int(tongue_point.x * w)
                            tongue_y = int(tongue_point.y * h)
                            # Check if this point is below lower lip
                            is_below = tongue_point.y > lower_lip.y + 0.01
                            color = (255, 0, 0) if is_below else (128, 128, 128)  # Blue if below, gray if not
                            cv2.circle(frame, (tongue_x, tongue_y), 6, color, 2)
                    
                    if tongue_out:
                        # Draw tongue tip indicator when detected
                        if 12 < len(face_landmarks.landmark):
                            tongue_tip = face_landmarks.landmark[12]
                            tongue_x = int(tongue_tip.x * w)
                            tongue_y = int(tongue_tip.y * h)
                            cv2.circle(frame, (tongue_x, tongue_y), 15, (255, 0, 255), 3)  # Magenta for detected tongue
        
        # Determine which ACTIVE pose is detected (check these first)
        thinking_pose = pose_detected and finger_at_mouth and looking_up
        idea_pose = pose_detected and finger_pointing_up
        shocked_pose = pose_detected and hands_on_chest and mouth_open
        
        # Tongue pose only if tongue is out AND hands are NOT on chest
        # If hands are on chest, show shocked instead (even if tongue is also out)
        tongue_pose = tongue_out and not hands_on_chest
        
        # Check for stare pose ONLY if no active gestures are detected
        # This prevents stare from overriding when user is actively gesturing
        any_active_gesture = (thinking_pose or idea_pose or shocked_pose or middle_finger_pose or thumbs_up or tongue_pose)
        if not any_active_gesture:
            stare_pose = self.is_stare_pose(pose_results, hand_results, face_results)
        else:
            stare_pose = False

        if shocked_pose:
            # Shocked has priority - if hands on chest + mouth open, show shocked (even if tongue also out)
            current_pose = "shocked"
            self.status_var.set("Shocked pose detected! 😱🐒")
            monkey_img = self.monkey_shocked_img
        elif tongue_pose:
            # Tongue only shows if tongue is out AND hands are NOT on chest
            current_pose = "tongue"
            self.status_var.set("Tongue out pose detected! 👅🐒")
            monkey_img = self.monkey_tongue_img
        elif middle_finger_pose:
            current_pose = "middle_finger"
            self.status_var.set("Middle finger pose detected! 🖕🐒")
            monkey_img = self.monkey_middle_finger_img
        elif thumbs_up:
            current_pose = "thumbs_up"
            self.status_var.set("Thumbs up pose detected! 👍🐒")
            monkey_img = self.monkey_thumbs_up_img
        elif idea_pose:
            current_pose = "idea"
            self.status_var.set("Idea pose detected! 💡🐒")
            monkey_img = self.monkey_idea_img
        elif thinking_pose:
            current_pose = "think"
            self.status_var.set("Thinking pose detected! 🧠🐒")
            monkey_img = self.monkey_think_img
        elif stare_pose:
            # Stare pose only triggers if no active gestures detected
            current_pose = "stare"
            self.status_var.set("Stare pose detected! 👀🐒")
            monkey_img = self.monkey_stare_img
        else:
            # Update status with helpful feedback
            status_parts = []
            if not pose_detected:
                status_parts.append("Show your face")
            elif stare_pose:
                # Stare pose should be detected, but if we're here something went wrong
                status_parts.append("Keep staring blankly")
            elif tongue_out and hands_on_chest:
                status_parts.append("Tongue detected but hands on chest = shocked pose")
            elif tongue_out and not hands_on_chest:
                status_parts.append("Tongue out!")
            elif thumbs_up:
                status_parts.append("Thumbs up at chest!")
            elif middle_finger_pose:
                status_parts.append("Both middle fingers!")
            elif hands_on_chest and not mouth_open:
                status_parts.append("Open mouth")
            elif mouth_open and not hands_on_chest:
                status_parts.append("Hands on chest")
            elif finger_pointing_up:
                status_parts.append("Keep finger pointing up!")
            elif finger_at_mouth and not looking_up:
                status_parts.append("Look up more")
            elif looking_up and not finger_at_mouth:
                status_parts.append("Finger at mouth")
            elif pose_detected and not hand_results.multi_hand_landmarks:
                # Face detected but no hands - suggest stare pose
                status_parts.append("Keep staring (or try other poses)")
            elif pose_detected:
                status_parts.append("Try: thinking, idea, shocked, stare, middle finger, thumbs up, or tongue")
            else:
                status_parts.append("Show your face and try a pose")
            
            if status_parts:
                self.status_var.set("Need: " + " + ".join(status_parts))
            else:
                self.status_var.set("Do a pose: thinking, idea, shocked, stare, middle finger, thumbs up, or tongue! 🐒")
            monkey_img = None

        # Overlay monkey image if pose detected
        if monkey_img is not None:
            mh, mw = monkey_img.shape[:2]
            scale = 0.35
            new_w = int(w * scale)
            new_h = int(mh * (new_w / mw))
            
            # Make sure we don't exceed frame boundaries
            new_w = min(new_w, w)
            new_h = min(new_h, h)
            
            # Resize monkey image
            resized_monkey = cv2.resize(monkey_img, (new_w, new_h))
            
            # Handle image with or without alpha channel
            if len(resized_monkey.shape) == 2 or resized_monkey.shape[2] == 1:
                # Grayscale - convert to BGR
                resized_monkey = cv2.cvtColor(resized_monkey, cv2.COLOR_GRAY2BGR)
            elif resized_monkey.shape[2] == 4:
                # Has alpha channel - need to blend
                alpha = resized_monkey[:, :, 3] / 255.0
                rgb_monkey = resized_monkey[:, :, :3]
                # Simple overlay (replace pixels)
                frame[0:new_h, 0:new_w] = (1 - alpha[:, :, None]) * frame[0:new_h, 0:new_w].astype(float) + alpha[:, :, None] * rgb_monkey.astype(float)
                frame = frame.astype('uint8')
            else:
                # Standard BGR image
                frame[0:new_h, 0:new_w] = resized_monkey

        # Convert BGR -> RGB for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self.photo)

        # Schedule next frame
        self.root.after(10, self.update_frame)

    def on_close(self):
        self.cap.release()
        self.hands.close()
        self.pose.close()
        self.face_mesh.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseMonkeyApp(root)
    root.mainloop()
