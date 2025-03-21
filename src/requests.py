import requests
import json

# API connection parameters
api_ip_port = "http://localhost:5000"  # Default value, change as needed


# Dictionary mapping gesture names to hex codes
# Load gesture codes from JSON file
def load_gesture_codes():
    try:
        with open("../arduino/GestureControl/codes.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading gesture codes: {e}")
        # Fallback to default codes if file can't be loaded
        return {
            "up": "0xe41b7f00",
            "down": "0xe41b7f01",
            "left": "0xe41b7f02",
            "right": "0xe41b7f03",
            "ok": "0xe41b7f04",
            "fist": "0xe41b7f05",
        }


gesture_codes = load_gesture_codes()


def send_gesture_code(gesture_name):
    """
    Send the hex code corresponding to the gesture name to the API.

    Args:
        gesture_name (str): The name of the gesture.

    Returns:
        dict: The response from the API.
    """
    if gesture_name not in gesture_codes:
        print(f"Unknown gesture: {gesture_name}")
        return None

    code = gesture_codes[gesture_name]
    payload = {"code": code}

    try:
        response = requests.post(f"{api_ip_port}/send", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None


def set_api_endpoint(ip, port):
    """
    Set the API endpoint IP and port.

    Args:
        ip (str): The IP address of the API.
        port (int): The port number of the API.
    """
    global api_ip_port
    api_ip_port = f"http://{ip}:{port}"
    print(f"API endpoint set to {api_ip_port}")


if __name__ == "__main__":
    # Example usage
    set_api_endpoint("localhost", 5000)

    # Send a gesture
    response = send_gesture_code("up")
    if response:
        print(f"Response: {response}")
